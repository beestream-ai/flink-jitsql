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
import org.objectweb.asm.ClassWriter;
import org.objectweb.asm.Opcodes;
import org.objectweb.asm.Type;
import org.objectweb.asm.tree.AbstractInsnNode;
import org.objectweb.asm.tree.ClassNode;
import org.objectweb.asm.tree.IincInsnNode;
import org.objectweb.asm.tree.MethodInsnNode;
import org.objectweb.asm.tree.MethodNode;
import org.objectweb.asm.tree.VarInsnNode;

import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;

/**
 * Turns a user's {@code eval} into a static method, so TornadoVM can compile it.
 *
 * <h2>Why this and not a lowering to kernel source</h2>
 *
 * <p>TornadoVM is already a Java-to-GPU compiler, and the generated kernels in this project are
 * generated <em>Java</em> that it then compiles. So a UDF does not need translating into anything:
 * it needs to be reachable from a kernel, and the only thing stopping it is that {@code eval} is an
 * instance method. A device has no object to call it on.
 *
 * <p>The transform is therefore mechanical rather than a decompilation. The body is copied
 * instruction for instruction into a static method of the same descriptor minus the receiver, and
 * every local variable index is shifted down by one to close the hole where {@code this} used to
 * sit. Nothing is interpreted, nothing is re-expressed, and the arithmetic cannot drift because it
 * is the same bytecode.
 *
 * <p>This is only sound because {@link UdfEligibilityAnalyzer} has already refused every body that
 * touches {@code this}: no {@code GETFIELD}, no {@code PUTFIELD}, no call on the receiver. Slot
 * zero is therefore dead, and removing it changes nothing. Lowering a method that was not analysed
 * first would silently renumber a live variable.
 */
public final class StaticUdfLowering {

    private static final String SUFFIX = "$jitsqlStatic";

    private StaticUdfLowering() {}

    /** A loader that only ever holds generated trampolines, so nothing of ours leaks into it. */
    public static final class TrampolineLoader extends ClassLoader {

        TrampolineLoader(ClassLoader parent) {
            super(parent);
        }

        Class<?> define(String name, byte[] bytes) {
            return defineClass(name, bytes, 0, bytes.length);
        }
    }

    /**
     * Lowers {@code eval} into a static method with the same body.
     *
     * @throws IllegalArgumentException if the method was not found eligible first
     */
    public static Method lower(Method eval) {
        UdfEligibility eligibility = UdfEligibilityAnalyzer.analyse(eval);
        if (!eligibility.eligible()) {
            throw new IllegalArgumentException(
                    "refusing to lower a body that was not admitted: " + eligibility);
        }
        Class<?> owner = eval.getDeclaringClass();
        try {
            ClassNode source = read(owner);
            String descriptor = Type.getMethodDescriptor(eval);
            MethodNode body = find(source, eval.getName(), descriptor);
            if (body == null) {
                throw new IllegalStateException(eval + " vanished between analysis and lowering");
            }
            String generatedName = owner.getName().replace('.', '/') + SUFFIX;
            byte[] bytes = emit(source, body, generatedName);
            LoweredClasses.publish(generatedName, bytes);
            Class<?> trampoline =
                    new TrampolineLoader(owner.getClassLoader())
                            .define(generatedName.replace('/', '.'), bytes);
            return trampoline.getMethod(eval.getName(), eval.getParameterTypes());
        } catch (IOException | ReflectiveOperationException e) {
            throw new IllegalStateException("could not lower " + eval, e);
        }
    }

    private static byte[] emit(ClassNode source, MethodNode body, String generatedName) {
        ClassNode generated = new ClassNode();
        generated.version = Math.max(source.version, Opcodes.V17);
        generated.access = Opcodes.ACC_PUBLIC | Opcodes.ACC_FINAL | Opcodes.ACC_SUPER;
        generated.name = generatedName;
        generated.superName = "java/lang/Object";
        generated.methods = new ArrayList<>();
        generated.methods.add(staticCopy(body, generatedName, true));
        // Static helpers the body calls come with it, under the new owner, so the trampoline is
        // self-contained and TornadoVM never has to reach back into the user's class.
        for (MethodNode method : source.methods) {
            if (method == body) {
                continue;
            }
            if ((method.access & Opcodes.ACC_STATIC) != 0 && !"<clinit>".equals(method.name)) {
                generated.methods.add(staticCopy(method, generatedName, false));
            }
        }
        ClassWriter writer = new ClassWriter(ClassWriter.COMPUTE_FRAMES | ClassWriter.COMPUTE_MAXS);
        generated.accept(writer);
        return writer.toByteArray();
    }

    /**
     * Copies a method body, dropping the receiver slot when there was one.
     *
     * @param dropReceiver true for the instance {@code eval}, false for a helper already static
     */
    private static MethodNode staticCopy(
            MethodNode body, String generatedName, boolean dropReceiver) {
        MethodNode copy = new MethodNode();
        copy.access = Opcodes.ACC_PUBLIC | Opcodes.ACC_STATIC;
        copy.name = body.name;
        copy.desc = body.desc;
        copy.exceptions = new ArrayList<>();
        copy.tryCatchBlocks = new ArrayList<>();
        copy.instructions = body.instructions;
        copy.maxLocals = Math.max(0, body.maxLocals - (dropReceiver ? 1 : 0));
        copy.maxStack = body.maxStack;

        List<AbstractInsnNode> all = new ArrayList<>();
        for (AbstractInsnNode insn = copy.instructions.getFirst();
                insn != null;
                insn = insn.getNext()) {
            all.add(insn);
        }
        for (AbstractInsnNode insn : all) {
            if (dropReceiver && insn instanceof VarInsnNode) {
                VarInsnNode var = (VarInsnNode) insn;
                if (var.var >= 1) {
                    var.var -= 1;
                }
            } else if (dropReceiver && insn instanceof IincInsnNode) {
                IincInsnNode inc = (IincInsnNode) insn;
                if (inc.var >= 1) {
                    inc.var -= 1;
                }
            }
            if (insn instanceof MethodInsnNode) {
                MethodInsnNode call = (MethodInsnNode) insn;
                // A helper that travelled with us is now a method of the trampoline.
                if (call.getOpcode() == Opcodes.INVOKESTATIC && !call.owner.startsWith("java/")) {
                    call.owner = generatedName;
                }
            }
        }
        // Local variable tables name slots that have just moved, and nothing reads them on a
        // device. Dropping them is cheaper than renumbering them correctly.
        copy.localVariables = null;
        copy.visibleLocalVariableAnnotations = null;
        copy.invisibleLocalVariableAnnotations = null;
        copy.parameters = null;
        return copy;
    }

    /** The binary name a generated kernel writes to call this function. */
    public static String trampolineName(Class<?> owner) {
        return owner.getName() + SUFFIX;
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
}
