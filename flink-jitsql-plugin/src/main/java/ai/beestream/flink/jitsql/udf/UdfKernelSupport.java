/*
 * Copyright (c) 2026 Beestream Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ai.beestream.flink.jitsql.udf;

import org.apache.flink.table.accelerator.AccelUdfCall;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.reflect.Method;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

/**
 * Turns the name of a user's function into a static method a kernel can call, or into a reason.
 *
 * <p>This is the whole of what the provider does with {@link AccelUdfCall}. The planner named a
 * method; here the class is resolved against the user code classloader, the body is analysed, and
 * if it is admissible it is lowered to a static copy. Everything that could go wrong -- a class
 * that is not on this TaskManager, a body that allocates, an overload that moved -- comes back as a
 * sentence rather than an exception, because a provider that throws takes down a job with a
 * perfectly good CPU operator waiting.
 *
 * <p>Results are cached per TaskManager. Reading bytecode and defining a class is cheap next to a
 * device compilation but it is not free, and a query evaluates the same UDF on every batch.
 */
public final class UdfKernelSupport {

    private static final Logger LOG = LoggerFactory.getLogger(UdfKernelSupport.class);

    private static final ConcurrentMap<String, Resolution> CACHE = new ConcurrentHashMap<>();

    private UdfKernelSupport() {}

    /** Either a static method a kernel can call, or why there is none. */
    public static final class Resolution {

        private final Method lowered;
        private final Construct construct;
        private final String because;

        private Resolution(Method lowered, Construct construct, String because) {
            this.lowered = lowered;
            this.construct = construct;
            this.because = because;
        }

        static Resolution lowered(Method method) {
            return new Resolution(method, null, null);
        }

        static Resolution refused(Construct construct, String because) {
            return new Resolution(null, construct, because);
        }

        public boolean usable() {
            return lowered != null;
        }

        public Optional<Method> method() {
            return Optional.ofNullable(lowered);
        }

        public Optional<Construct> construct() {
            return Optional.ofNullable(construct);
        }

        public Optional<String> because() {
            return Optional.ofNullable(because);
        }

        @Override
        public String toString() {
            return usable() ? "lowered to " + lowered : construct + ": " + because;
        }
    }

    /** Resolves a call, using {@code loader} to find the user's class. Never throws. */
    public static Resolution resolve(AccelUdfCall call, ClassLoader loader) {
        String key =
                call.className()
                        + "#"
                        + call.methodName()
                        + call.methodDescriptor()
                        + "@"
                        + System.identityHashCode(loader);
        return CACHE.computeIfAbsent(key, ignored -> resolveUncached(call, loader));
    }

    private static Resolution resolveUncached(AccelUdfCall call, ClassLoader loader) {
        Method declared;
        try {
            Class<?> owner = Class.forName(call.className(), false, loader);
            declared = find(owner, call.methodName(), call.methodDescriptor());
        } catch (ClassNotFoundException | LinkageError e) {
            return Resolution.refused(
                    Construct.UNREADABLE, call.className() + " is not on this TaskManager: " + e);
        }
        if (declared == null) {
            return Resolution.refused(
                    Construct.UNREADABLE,
                    call.methodName()
                            + call.methodDescriptor()
                            + " is no longer declared by "
                            + call.className());
        }
        UdfEligibility eligibility = UdfEligibilityAnalyzer.analyse(declared);
        if (!eligibility.eligible()) {
            LOG.info("declining the user function {}: {}", call.className(), eligibility);
            return Resolution.refused(
                    eligibility.construct().orElse(Construct.UNREADABLE),
                    eligibility.detail().orElse("no reason given")
                            + (eligibility.line() >= 0
                                    ? " (line " + eligibility.line() + ")"
                                    : ""));
        }
        try {
            return Resolution.lowered(StaticUdfLowering.lower(declared));
        } catch (RuntimeException e) {
            return Resolution.refused(
                    Construct.UNREADABLE, "could not lower " + call.className() + ": " + e);
        }
    }

    /** Matches by descriptor, because a user function may have several {@code eval} overloads. */
    private static Method find(Class<?> owner, String name, String descriptor) {
        for (Method candidate : owner.getMethods()) {
            if (candidate.getName().equals(name) && descriptorOf(candidate).equals(descriptor)) {
                return candidate;
            }
        }
        return null;
    }

    private static String descriptorOf(Method method) {
        StringBuilder out = new StringBuilder("(");
        for (Class<?> parameter : method.getParameterTypes()) {
            append(out, parameter);
        }
        out.append(')');
        append(out, method.getReturnType());
        return out.toString();
    }

    private static void append(StringBuilder out, Class<?> type) {
        if (type.isArray()) {
            out.append('[');
            append(out, type.getComponentType());
        } else if (type == boolean.class) {
            out.append('Z');
        } else if (type == byte.class) {
            out.append('B');
        } else if (type == char.class) {
            out.append('C');
        } else if (type == short.class) {
            out.append('S');
        } else if (type == int.class) {
            out.append('I');
        } else if (type == long.class) {
            out.append('J');
        } else if (type == float.class) {
            out.append('F');
        } else if (type == double.class) {
            out.append('D');
        } else if (type == void.class) {
            out.append('V');
        } else {
            out.append('L').append(type.getName().replace('.', '/')).append(';');
        }
    }
}
