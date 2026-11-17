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

import org.objectweb.asm.ClassReader;
import org.objectweb.asm.Opcodes;
import org.objectweb.asm.Type;
import org.objectweb.asm.tree.AbstractInsnNode;
import org.objectweb.asm.tree.ClassNode;
import org.objectweb.asm.tree.FieldInsnNode;
import org.objectweb.asm.tree.JumpInsnNode;
import org.objectweb.asm.tree.LabelNode;
import org.objectweb.asm.tree.LdcInsnNode;
import org.objectweb.asm.tree.LineNumberNode;
import org.objectweb.asm.tree.MethodInsnNode;
import org.objectweb.asm.tree.MethodNode;

import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Method;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;

/**
 * Decides whether a user's {@code ScalarFunction.eval} body can become a kernel.
 *
 * <h2>Why bytecode at all</h2>
 *
 * <p>Because the thing that sends these functions to the CPU is structural. {@code AccelFunction}
 * is a closed set of scalar SQL built-ins, and an expression tree over them cannot hold a loop, an
 * accumulator or an early return no matter how many functions are added to it. A method body can.
 * That is the same reason the RAPIDS {@code udf-compiler} declines a looping UDF -- it builds
 * Catalyst expressions, and an expression tree cannot hold a loop either.
 *
 * <p>So this reads the body. A loop is not a refusal here; it is the reason to be here.
 *
 * <h2>Why refuse while planning</h2>
 *
 * <p>Planning-time refusal is not executor-time fallback. Once a plan holds a kernel, a failure
 * during execution fails the task -- there is no CPU implementation sitting beside a kernel that
 * was substituted into the plan. So a function is never sent to a kernel on the hope that it will
 * work: anything this analyser is unsure of is refused, and the query keeps the plan it had.
 */
public final class UdfEligibilityAnalyzer {

    /** {@code java.lang.Math} entry points with an exact device equivalent. */
    private static final Set<String> SUPPORTED_MATH =
            Set.of(
                    "abs",
                    "sqrt",
                    "exp",
                    "log",
                    "log10",
                    "pow",
                    "sin",
                    "cos",
                    "tan",
                    "asin",
                    "acos",
                    "atan",
                    "atan2",
                    "tanh",
                    "sinh",
                    "cosh",
                    "floor",
                    "ceil",
                    "min",
                    "max",
                    "signum",
                    "cbrt",
                    "hypot",
                    "toRadians",
                    "toDegrees");

    /** Descriptors a kernel argument or result can have. */
    private static final Set<String> SUPPORTED_TYPES =
            Set.of("I", "J", "F", "D", "B", "S", "Z", "[I", "[J", "[F", "[D", "[B");

    private UdfEligibilityAnalyzer() {}

    /** Analyses {@code method}, which must be an {@code eval} of a user's scalar function. */
    public static UdfEligibility analyse(Method method) {
        Class<?> owner = method.getDeclaringClass();
        UdfEligibility signature = checkSignature(method);
        if (signature != null) {
            return signature;
        }
        ClassNode node;
        try {
            node = read(owner);
        } catch (IOException | RuntimeException e) {
            return UdfEligibility.refused(
                    Construct.UNREADABLE,
                    "the bytecode of " + owner.getName() + " could not be read: " + e,
                    -1);
        }
        String descriptor = Type.getMethodDescriptor(method);
        MethodNode body = find(node, method.getName(), descriptor);
        if (body == null) {
            return UdfEligibility.refused(
                    Construct.UNREADABLE,
                    method.getName() + descriptor + " is not in " + owner.getName(),
                    -1);
        }
        return walk(node, body, new LinkedHashSet<>(), 0);
    }

    private static UdfEligibility checkSignature(Method method) {
        for (Class<?> parameter : method.getParameterTypes()) {
            String descriptor = Type.getDescriptor(parameter);
            if (!SUPPORTED_TYPES.contains(descriptor)) {
                return UdfEligibility.refused(
                        Construct.UNSUPPORTED_TYPE,
                        "the parameter of type "
                                + parameter.getSimpleName()
                                + " has no device representation",
                        -1);
            }
        }
        String returned = Type.getDescriptor(method.getReturnType());
        if (!SUPPORTED_TYPES.contains(returned) || returned.startsWith("[")) {
            return UdfEligibility.refused(
                    Construct.UNSUPPORTED_TYPE,
                    "the return type "
                            + method.getReturnType().getSimpleName()
                            + " is not a primitive a kernel can write",
                    -1);
        }
        return null;
    }

    /** Walks one method body, following static helpers in the same class. */
    private static UdfEligibility walk(
            ClassNode owner, MethodNode body, Set<String> visiting, int depth) {
        if (depth > 8) {
            return UdfEligibility.refused(
                    Construct.RECURSION, "helper methods nest more than eight deep", -1);
        }
        String self = body.name + body.desc;
        if (!visiting.add(self)) {
            return UdfEligibility.refused(
                    Construct.RECURSION, body.name + " calls itself", lineNear(body, null));
        }
        if (body.tryCatchBlocks != null && !body.tryCatchBlocks.isEmpty()) {
            return UdfEligibility.refused(
                    Construct.EXCEPTION,
                    "the body has an exception handler, and a kernel has nowhere to unwind to",
                    lineNear(body, null));
        }
        int loops = 0;
        int arrayReads = 0;
        Map<LabelNode, Integer> positions = positionsOf(body);
        for (AbstractInsnNode insn = body.instructions.getFirst();
                insn != null;
                insn = insn.getNext()) {
            int op = insn.getOpcode();
            switch (op) {
                case Opcodes.NEW:
                case Opcodes.ANEWARRAY:
                case Opcodes.MULTIANEWARRAY:
                case Opcodes.NEWARRAY:
                    return UdfEligibility.refused(
                            Construct.OBJECT_ALLOCATION,
                            "the body allocates, and a kernel has no heap",
                            lineNear(body, insn));
                case Opcodes.ATHROW:
                    return UdfEligibility.refused(
                            Construct.EXCEPTION, "the body throws", lineNear(body, insn));
                case Opcodes.MONITORENTER:
                case Opcodes.MONITOREXIT:
                    return UdfEligibility.refused(
                            Construct.SYNCHRONIZATION,
                            "the body synchronises, and there is nothing to synchronise with",
                            lineNear(body, insn));
                case Opcodes.GETFIELD:
                case Opcodes.PUTFIELD:
                case Opcodes.PUTSTATIC:
                    return UdfEligibility.refused(
                            Construct.OBJECT_ACCESS,
                            "the body reads or writes the field "
                                    + ((FieldInsnNode) insn).name
                                    + ", which no kernel argument carries",
                            lineNear(body, insn));
                case Opcodes.GETSTATIC:
                    // A static final primitive is a constant and perfectly kernel-safe; anything
                    // else is state the device has no copy of.
                    FieldInsnNode field = (FieldInsnNode) insn;
                    if (!SUPPORTED_TYPES.contains(field.desc)) {
                        return UdfEligibility.refused(
                                Construct.OBJECT_ACCESS,
                                "the body reads the static field "
                                        + field.name
                                        + " of type "
                                        + field.desc,
                                lineNear(body, insn));
                    }
                    break;
                case Opcodes.IALOAD:
                case Opcodes.LALOAD:
                case Opcodes.FALOAD:
                case Opcodes.DALOAD:
                case Opcodes.BALOAD:
                case Opcodes.SALOAD:
                case Opcodes.CALOAD:
                    arrayReads++;
                    break;
                case Opcodes.AALOAD:
                case Opcodes.AASTORE:
                    return UdfEligibility.refused(
                            Construct.OBJECT_ACCESS,
                            "the body indexes an array of objects",
                            lineNear(body, insn));
                case Opcodes.INVOKEDYNAMIC:
                    return UdfEligibility.refused(
                            Construct.UNSUPPORTED_CALL,
                            "the body uses invokedynamic -- a lambda, a method reference or"
                                    + " string concatenation",
                            lineNear(body, insn));
                case Opcodes.INVOKEVIRTUAL:
                case Opcodes.INVOKEINTERFACE:
                case Opcodes.INVOKESPECIAL:
                case Opcodes.INVOKESTATIC:
                    UdfEligibility call =
                            checkCall(owner, body, (MethodInsnNode) insn, visiting, depth);
                    if (call != null && !call.eligible()) {
                        return call;
                    }
                    if (call != null) {
                        loops += call.loops();
                        arrayReads += call.arrayReads();
                    }
                    break;
                default:
                    break;
            }
            if (insn instanceof LdcInsnNode && ((LdcInsnNode) insn).cst instanceof String) {
                return UdfEligibility.refused(
                        Construct.STRING, "the body mentions a string", lineNear(body, insn));
            }
            if (insn instanceof JumpInsnNode) {
                Integer target = positions.get(((JumpInsnNode) insn).label);
                Integer here = body.instructions.indexOf(insn);
                if (target != null && target < here) {
                    // A backward jump is a loop. Counted rather than refused: this is the
                    // construct an expression tree cannot hold, and the reason to read bytecode.
                    loops++;
                }
            }
        }
        visiting.remove(self);
        return UdfEligibility.eligible(loops, arrayReads);
    }

    /** Math goes to a built-in; a static helper in the same class is followed; the rest refuse. */
    private static UdfEligibility checkCall(
            ClassNode owner,
            MethodNode body,
            MethodInsnNode call,
            Set<String> visiting,
            int depth) {
        if ("java/lang/Math".equals(call.owner) || "java/lang/StrictMath".equals(call.owner)) {
            if (SUPPORTED_MATH.contains(call.name)) {
                return null;
            }
            return UdfEligibility.refused(
                    Construct.UNSUPPORTED_MATH,
                    "Math." + call.name + " has no exact device equivalent",
                    lineNear(body, call));
        }
        if (call.owner.equals(owner.name) && call.getOpcode() == Opcodes.INVOKESTATIC) {
            MethodNode helper = find(owner, call.name, call.desc);
            if (helper == null) {
                return UdfEligibility.refused(
                        Construct.UNSUPPORTED_CALL,
                        call.name + " is not in this class after all",
                        lineNear(body, call));
            }
            return walk(owner, helper, visiting, depth + 1);
        }
        if (call.getOpcode() == Opcodes.INVOKESPECIAL && "<init>".equals(call.name)) {
            return UdfEligibility.refused(
                    Construct.OBJECT_ALLOCATION,
                    "the body constructs a " + call.owner,
                    lineNear(body, call));
        }
        return UdfEligibility.refused(
                Construct.UNSUPPORTED_CALL,
                "the body calls "
                        + call.owner.replace('/', '.')
                        + "."
                        + call.name
                        + ", which this analyser will not follow",
                lineNear(body, call));
    }

    private static ClassNode read(Class<?> type) throws IOException {
        String resource = type.getName().replace('.', '/') + ".class";
        try (InputStream in = type.getClassLoader().getResourceAsStream(resource)) {
            if (in == null) {
                throw new IOException("no class file for " + type.getName());
            }
            ClassNode node = new ClassNode();
            new ClassReader(in).accept(node, ClassReader.SKIP_FRAMES);
            return node;
        }
    }

    private static MethodNode find(ClassNode owner, String name, String descriptor) {
        for (MethodNode method : owner.methods) {
            if (method.name.equals(name) && method.desc.equals(descriptor)) {
                return method;
            }
        }
        return null;
    }

    private static Map<LabelNode, Integer> positionsOf(MethodNode body) {
        Map<LabelNode, Integer> positions = new HashMap<>();
        for (AbstractInsnNode insn = body.instructions.getFirst();
                insn != null;
                insn = insn.getNext()) {
            if (insn instanceof LabelNode) {
                positions.put((LabelNode) insn, body.instructions.indexOf(insn));
            }
        }
        return positions;
    }

    /** The source line covering an instruction, or -1 when the class has no line numbers. */
    private static int lineNear(MethodNode body, AbstractInsnNode at) {
        int line = -1;
        Set<AbstractInsnNode> seen = new HashSet<>();
        for (AbstractInsnNode insn = body.instructions.getFirst();
                insn != null && seen.add(insn);
                insn = insn.getNext()) {
            if (insn instanceof LineNumberNode) {
                line = ((LineNumberNode) insn).line;
            }
            if (insn == at) {
                return line;
            }
        }
        return at == null ? line : -1;
    }
}
