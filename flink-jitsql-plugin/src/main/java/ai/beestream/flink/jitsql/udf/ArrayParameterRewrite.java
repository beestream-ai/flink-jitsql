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

import org.objectweb.asm.ClassWriter;
import org.objectweb.asm.Opcodes;
import org.objectweb.asm.Type;
import org.objectweb.asm.tree.AbstractInsnNode;
import org.objectweb.asm.tree.ClassNode;
import org.objectweb.asm.tree.IincInsnNode;
import org.objectweb.asm.tree.InsnList;
import org.objectweb.asm.tree.InsnNode;
import org.objectweb.asm.tree.MethodInsnNode;
import org.objectweb.asm.tree.MethodNode;
import org.objectweb.asm.tree.VarInsnNode;
import org.objectweb.asm.tree.analysis.Analyzer;
import org.objectweb.asm.tree.analysis.AnalyzerException;
import org.objectweb.asm.tree.analysis.Frame;
import org.objectweb.asm.tree.analysis.SourceInterpreter;
import org.objectweb.asm.tree.analysis.SourceValue;

import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Rewrites a {@code double[]} parameter into the flat buffer and bounds a device actually has.
 *
 * <h2>The problem</h2>
 *
 * <p>An {@code ARRAY} column is not an array of arrays on a device. It is one flat run of elements
 * with offsets saying where each row's run begins -- the cuDF {@code LIST} layout, which is also
 * what {@code ArrayColumnGather} stages into. So the row a kernel is working on does not have a
 * {@code double[]}; it has a buffer, a start and an end.
 *
 * <p>A UDF written against {@code double[]} therefore cannot be called as written, however
 * admissible its body. Each array parameter becomes three: the buffer, the start and the end.
 * {@code a.length} becomes {@code end - start}, and {@code a[i]} becomes {@code buffer.get(start +
 * i)}.
 *
 * <h2>Why this needs stack analysis rather than pattern matching</h2>
 *
 * <p>{@code ARRAYLENGTH} always sits immediately after the reference that feeds it, so that one
 * could be matched by looking backwards. {@code DALOAD} cannot: between the reference and the load
 * sits the index expression, which is arbitrary. Finding which local produced the reference is a
 * dataflow question, and ASM answers it -- {@link SourceInterpreter} reports, for every stack
 * value, the instructions that produced it.
 *
 * <p>Anything the analysis cannot place is refused rather than guessed. If an array reference
 * reaches a load from anywhere but a parameter -- stored into a local, chosen by a branch, handed
 * to a call -- there is no single buffer and offset to rewrite it to, and emitting something
 * plausible would compute over the wrong elements.
 */
public final class ArrayParameterRewrite {

    private static final String SUFFIX = "$jitsqlFlat";

    private ArrayParameterRewrite() {}

    /** The TornadoVM array type that carries elements of a given array descriptor. */
    private static String bufferTypeOf(String elementDescriptor) {
        switch (elementDescriptor) {
            case "[I":
                return "uk/ac/manchester/tornado/api/types/arrays/IntArray";
            case "[J":
                return "uk/ac/manchester/tornado/api/types/arrays/LongArray";
            case "[F":
                return "uk/ac/manchester/tornado/api/types/arrays/FloatArray";
            case "[D":
                return "uk/ac/manchester/tornado/api/types/arrays/DoubleArray";
            default:
                return null;
        }
    }

    private static String getterDescriptorOf(String elementDescriptor) {
        switch (elementDescriptor) {
            case "[I":
                return "(I)I";
            case "[J":
                return "(I)J";
            case "[F":
                return "(I)F";
            case "[D":
                return "(I)D";
            default:
                return null;
        }
    }

    /** The result: either a rewritten method, or why the body could not be rewritten. */
    public static final class Rewritten {

        private final Method method;
        private final Construct construct;
        private final String because;

        private Rewritten(Method method, Construct construct, String because) {
            this.method = method;
            this.construct = construct;
            this.because = because;
        }

        public boolean usable() {
            return method != null;
        }

        public Method method() {
            return method;
        }

        public Construct construct() {
            return construct;
        }

        public String because() {
            return because;
        }

        @Override
        public String toString() {
            return usable() ? "rewritten to " + method : construct + ": " + because;
        }
    }

    /**
     * Rewrites a lowered static method so each array parameter becomes buffer, start and end.
     *
     * @param lowered the output of {@link StaticUdfLowering}, which is already static and already
     *     known not to touch a receiver
     */
    public static Rewritten rewrite(Method lowered) {
        Class<?>[] parameters = lowered.getParameterTypes();
        List<Integer> arrayParams = new ArrayList<>();
        for (int i = 0; i < parameters.length; i++) {
            if (parameters[i].isArray()) {
                if (bufferTypeOf(Type.getDescriptor(parameters[i])) == null) {
                    return new Rewritten(
                            null,
                            Construct.UNSUPPORTED_TYPE,
                            "an array of "
                                    + parameters[i].getComponentType().getSimpleName()
                                    + " has no device buffer type");
                }
                arrayParams.add(i);
            }
        }
        if (arrayParams.isEmpty()) {
            return new Rewritten(lowered, null, null);
        }
        try {
            ClassNode source = readDeclaring(lowered);
            MethodNode body = findMethod(source, lowered);
            if (body == null) {
                return new Rewritten(null, Construct.UNREADABLE, "the lowered method went missing");
            }
            return emit(lowered, source, body, parameters, arrayParams);
        } catch (Exception e) {
            return new Rewritten(
                    null, Construct.UNREADABLE, "could not rewrite " + lowered + ": " + e);
        }
    }

    private static Rewritten emit(
            Method lowered,
            ClassNode source,
            MethodNode body,
            Class<?>[] parameters,
            List<Integer> arrayParams)
            throws AnalyzerException, ReflectiveOperationException {

        // Where each original parameter's locals move to, once array parameters take three slots.
        Map<Integer, Integer> slotOf = new HashMap<>();
        Map<Integer, int[]> arraySlots = new HashMap<>();
        List<Type> newParameterTypes = new ArrayList<>();
        int oldSlot = 0;
        int newSlot = 0;
        for (int i = 0; i < parameters.length; i++) {
            slotOf.put(oldSlot, newSlot);
            Type original = Type.getType(parameters[i]);
            if (parameters[i].isArray()) {
                String buffer = bufferTypeOf(original.getDescriptor());
                newParameterTypes.add(Type.getObjectType(buffer));
                newParameterTypes.add(Type.INT_TYPE);
                newParameterTypes.add(Type.INT_TYPE);
                arraySlots.put(oldSlot, new int[] {newSlot, newSlot + 1, newSlot + 2});
                newSlot += 3;
            } else {
                newParameterTypes.add(original);
                newSlot += original.getSize();
            }
            oldSlot += original.getSize();
        }
        // Body locals follow the parameters and shift by however much the parameters grew.
        final int shift = newSlot - oldSlot;

        Analyzer<SourceValue> analyzer = new Analyzer<>(new SourceInterpreter());
        Frame<SourceValue>[] frames = analyzer.analyze(source.name, body);

        InsnList instructions = body.instructions;
        List<AbstractInsnNode> all = new ArrayList<>();
        Map<AbstractInsnNode, Integer> indexOf = new HashMap<>();
        for (AbstractInsnNode insn = instructions.getFirst(); insn != null; insn = insn.getNext()) {
            indexOf.put(insn, all.size());
            all.add(insn);
        }

        // Decide everything against the original frames, then apply. Rewriting as we walk would
        // renumber the instruction list under the analysis that is still being consulted, and the
        // second array access onwards would be read against the wrong frame -- which shows up not
        // as a crash but as a refusal, or worse, as a trace to the wrong parameter.
        Map<AbstractInsnNode, InsnList> replacements = new HashMap<>();
        for (AbstractInsnNode insn : all) {
            int index = indexOf.get(insn);
            int op = insn.getOpcode();

            if (op == Opcodes.ARRAYLENGTH) {
                Integer param = parameterBehind(frames[index], 0, arraySlots.keySet());
                if (param == null) {
                    return new Rewritten(
                            null,
                            Construct.UNPROVABLE_BOUNDS,
                            "an array length is taken of something that is not a parameter");
                }
                int[] slots = arraySlots.get(param);
                InsnList replacement = new InsnList();
                // The reference that fed ARRAYLENGTH is left on the stack by its ALOAD, which
                // becomes a buffer load below; pop it, then compute end - start.
                replacement.add(new InsnNode(Opcodes.POP));
                replacement.add(new VarInsnNode(Opcodes.ILOAD, slots[2]));
                replacement.add(new VarInsnNode(Opcodes.ILOAD, slots[1]));
                replacement.add(new InsnNode(Opcodes.ISUB));
                replacements.put(insn, replacement);
                continue;
            }

            if (isArrayLoad(op)) {
                Integer param = parameterBehind(frames[index], 1, arraySlots.keySet());
                if (param == null) {
                    return new Rewritten(
                            null,
                            Construct.UNPROVABLE_BOUNDS,
                            "an array element is read through a reference this analysis cannot"
                                    + " trace to a parameter");
                }
                int[] slots = arraySlots.get(param);
                String descriptor =
                        Type.getType(parameters[parameterIndexOf(param, parameters)])
                                .getDescriptor();
                InsnList replacement = new InsnList();
                // stack: buffer, index -> buffer, start + index -> buffer.get(start + index)
                replacement.add(new VarInsnNode(Opcodes.ILOAD, slots[1]));
                replacement.add(new InsnNode(Opcodes.IADD));
                replacement.add(
                        new MethodInsnNode(
                                Opcodes.INVOKEVIRTUAL,
                                bufferTypeOf(descriptor),
                                "get",
                                getterDescriptorOf(descriptor),
                                false));
                replacements.put(insn, replacement);
                continue;
            }

            if (isArrayStore(op)) {
                return new Rewritten(
                        null,
                        Construct.OBJECT_ACCESS,
                        "the body writes to an array argument, which a device view is not for");
            }
        }

        for (Map.Entry<AbstractInsnNode, InsnList> entry : replacements.entrySet()) {
            instructions.insert(entry.getKey(), entry.getValue());
            instructions.remove(entry.getKey());
        }

        // The array reference loads are now dead pushes feeding an IADD that wants an int, so
        // every ALOAD of an array parameter is replaced by the buffer it became; the verifier
        // then sees buffer, index -> get(index).
        for (AbstractInsnNode insn : all) {
            if (insn instanceof VarInsnNode) {
                VarInsnNode var = (VarInsnNode) insn;
                if (var.getOpcode() == Opcodes.ALOAD && arraySlots.containsKey(var.var)) {
                    var.var = arraySlots.get(var.var)[0];
                } else {
                    var.var = remap(var.var, slotOf, oldSlot, shift);
                }
            } else if (insn instanceof IincInsnNode) {
                IincInsnNode inc = (IincInsnNode) insn;
                inc.var = remap(inc.var, slotOf, oldSlot, shift);
            }
        }

        String descriptor =
                Type.getMethodDescriptor(
                        Type.getType(lowered.getReturnType()),
                        newParameterTypes.toArray(new Type[0]));
        String className = source.name + SUFFIX;

        MethodNode rewritten = new MethodNode();
        rewritten.access = Opcodes.ACC_PUBLIC | Opcodes.ACC_STATIC;
        rewritten.name = lowered.getName();
        rewritten.desc = descriptor;
        rewritten.exceptions = new ArrayList<>();
        rewritten.tryCatchBlocks = new ArrayList<>();
        rewritten.instructions = instructions;
        rewritten.maxLocals = body.maxLocals + shift + 3;
        rewritten.maxStack = body.maxStack + 3;

        ClassNode generated = new ClassNode();
        generated.version = Math.max(source.version, Opcodes.V17);
        generated.access = Opcodes.ACC_PUBLIC | Opcodes.ACC_FINAL | Opcodes.ACC_SUPER;
        generated.name = className;
        generated.superName = "java/lang/Object";
        generated.methods = new ArrayList<>();
        generated.methods.add(rewritten);
        for (MethodNode method : source.methods) {
            if (method != body && (method.access & Opcodes.ACC_STATIC) != 0) {
                generated.methods.add(method);
            }
        }

        ClassWriter writer = new ClassWriter(ClassWriter.COMPUTE_FRAMES | ClassWriter.COMPUTE_MAXS);
        generated.accept(writer);
        byte[] bytes = writer.toByteArray();

        // Written as well as defined, for the same reason the trampoline is: the generated kernel
        // is Java source, and javac resolves a name on a classpath rather than in a classloader.
        LoweredClasses.publish(className, bytes);
        Class<?> defined =
                new StaticUdfLowering.TrampolineLoader(lowered.getDeclaringClass().getClassLoader())
                        .define(className.replace('/', '.'), bytes);
        Class<?>[] signature = new Class<?>[newParameterTypes.size()];
        for (int i = 0; i < signature.length; i++) {
            signature[i] = classOf(newParameterTypes.get(i), defined.getClassLoader());
        }
        return new Rewritten(defined.getMethod(lowered.getName(), signature), null, null);
    }

    private static int parameterIndexOf(int slot, Class<?>[] parameters) {
        int at = 0;
        for (int i = 0; i < parameters.length; i++) {
            if (at == slot) {
                return i;
            }
            at += Type.getType(parameters[i]).getSize();
        }
        throw new IllegalStateException("no parameter at slot " + slot);
    }

    private static int remap(int var, Map<Integer, Integer> slotOf, int parameterSlots, int shift) {
        Integer mapped = slotOf.get(var);
        if (mapped != null) {
            return mapped;
        }
        return var >= parameterSlots ? var + shift : var;
    }

    /** The array parameter whose reference sits {@code below} slots under the top of the stack. */
    private static Integer parameterBehind(
            Frame<SourceValue> frame, int below, java.util.Set<Integer> arraySlots) {
        if (frame == null || frame.getStackSize() <= below) {
            return null;
        }
        SourceValue value = frame.getStack(frame.getStackSize() - 1 - below);
        if (value.insns.size() != 1) {
            // Produced on more than one path: there is no single buffer to rewrite it to.
            return null;
        }
        AbstractInsnNode producer = value.insns.iterator().next();
        if (!(producer instanceof VarInsnNode) || producer.getOpcode() != Opcodes.ALOAD) {
            return null;
        }
        int slot = ((VarInsnNode) producer).var;
        return arraySlots.contains(slot) ? slot : null;
    }

    private static boolean isArrayLoad(int op) {
        return op == Opcodes.IALOAD
                || op == Opcodes.LALOAD
                || op == Opcodes.FALOAD
                || op == Opcodes.DALOAD;
    }

    private static boolean isArrayStore(int op) {
        return op == Opcodes.IASTORE
                || op == Opcodes.LASTORE
                || op == Opcodes.FASTORE
                || op == Opcodes.DASTORE;
    }

    private static Class<?> classOf(Type type, ClassLoader loader) throws ClassNotFoundException {
        switch (type.getSort()) {
            case Type.INT:
                return int.class;
            case Type.LONG:
                return long.class;
            case Type.FLOAT:
                return float.class;
            case Type.DOUBLE:
                return double.class;
            default:
                return Class.forName(type.getClassName(), false, loader);
        }
    }

    /**
     * Reads the declaring class's bytes.
     *
     * <p>The method handed here has usually been lowered already, so its class exists in a {@code
     * TrampolineLoader} and in the directory {@link StaticUdfLowering} wrote it to -- and not as a
     * resource of any classloader, which is where the obvious lookup goes first and finds nothing.
     * The directory is the authority for a generated class; a classloader resource is the authority
     * for an ordinary one.
     */
    private static ClassNode readDeclaring(Method method) throws Exception {
        Class<?> owner = method.getDeclaringClass();
        String path = owner.getName().replace('.', '/') + ".class";
        java.nio.file.Path written = LoweredClasses.outputDirectory().resolve(path);
        if (java.nio.file.Files.exists(written)) {
            try (java.io.InputStream in = java.nio.file.Files.newInputStream(written)) {
                return parse(in);
            }
        }
        try (java.io.InputStream in = owner.getClassLoader().getResourceAsStream(path)) {
            if (in == null) {
                throw new java.io.IOException("no class file for " + owner.getName());
            }
            return parse(in);
        }
    }

    private static ClassNode parse(java.io.InputStream in) throws java.io.IOException {
        ClassNode node = new ClassNode();
        new org.objectweb.asm.ClassReader(in).accept(node, 0);
        return node;
    }

    private static MethodNode findMethod(ClassNode owner, Method method) {
        String descriptor = Type.getMethodDescriptor(method);
        for (MethodNode candidate : owner.methods) {
            if (candidate.name.equals(method.getName()) && candidate.desc.equals(descriptor)) {
                return candidate;
            }
        }
        return null;
    }
}
