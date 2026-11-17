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
package ai.beestream.flink.jitsql.operator;

import ai.beestream.flink.jitsql.codegen.AccelKernelGenerator;
import ai.beestream.flink.jitsql.codegen.GpuKernelSource;

import org.apache.flink.table.accelerator.AccelInput;
import org.apache.flink.table.accelerator.AccelInputRef;
import org.apache.flink.table.accelerator.AccelLiteral;
import org.apache.flink.table.accelerator.AccelNode;
import org.apache.flink.table.accelerator.AccelProject;
import org.apache.flink.table.accelerator.AccelUdfCall;
import org.apache.flink.table.functions.ScalarFunction;
import org.apache.flink.table.types.logical.ArrayType;
import org.apache.flink.table.types.logical.DoubleType;
import org.apache.flink.table.types.logical.RowType;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Optional;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * A user's function reaching a kernel: the last link of B5.
 *
 * <p>What this proves without a GPU is the part that would otherwise only fail on one. The
 * generated kernel is Java source compiled by {@code javac}, and a call to a lowered trampoline
 * resolves only if the trampoline's class file is on the classpath the kernel is compiled with.
 * That wiring is invisible until something compiles -- so something compiles here.
 */
class UdfKernelGenerationTest {

    /** A loop and an accumulator: what an expression tree cannot hold and a method can. */
    public static final class Decay extends ScalarFunction {
        public double eval(double x, double rate) {
            double total = 0;
            for (int i = 0; i < 8; i++) {
                total += x * Math.exp(-rate * i);
            }
            return total;
        }
    }

    public static final class Allocating extends ScalarFunction {
        public double eval(double x) {
            double[] scratch = new double[2];
            scratch[0] = x;
            return scratch[0];
        }
    }

    public static final class TakingAnArray extends ScalarFunction {
        public double eval(double[] values) {
            double total = 0;
            for (int i = 0; i < values.length; i++) {
                total += values[i];
            }
            return total;
        }
    }

    private static final RowType INPUT = RowType.of(new DoubleType(false), new DoubleType(false));

    private static AccelNode projectionCalling(
            Class<?> udf,
            String descriptor,
            List<org.apache.flink.table.accelerator.AccelExpression> operands) {
        return new AccelProject(
                List.of(
                        new AccelUdfCall(
                                udf.getName(),
                                "eval",
                                descriptor,
                                operands,
                                new DoubleType(false))),
                new AccelInput(INPUT),
                RowType.of(new DoubleType(false)));
    }

    private static AccelNode decayProjection() {
        return projectionCalling(
                Decay.class,
                "(DD)D",
                List.of(
                        new AccelInputRef(0, new DoubleType(false)),
                        new AccelInputRef(1, new DoubleType(false))));
    }

    @Test
    void aUserFunctionIsNamedInTheGeneratedKernel() {
        Optional<GpuKernelSource> kernel = AccelKernelGenerator.generate(decayProjection(), "udf");
        assertThat(kernel).isPresent();
        assertThat(kernel.get().source())
                .contains(Decay.class.getName() + "$jitsqlStatic.eval(")
                .contains("@Parallel");
    }

    /**
     * The acceptance test: the kernel that calls it compiles.
     *
     * <p>This is the assertion the whole chain hangs on. {@code javac} resolves {@code
     * Decay$jitsqlStatic} only because {@code StaticUdfLowering} wrote its class file to a
     * directory and {@code GeneratedKernel} put that directory on the compile classpath. Get either
     * wrong and this fails here rather than on a device.
     */
    @Test
    void theKernelThatCallsItCompiles() throws Exception {
        GpuKernelSource kernel =
                AccelKernelGenerator.generate(decayProjection(), "udfcompile").orElseThrow();
        try (GeneratedKernelEngine.Compiled compiled =
                GeneratedKernelEngine.compileStandalone(kernel)) {
            assertThat(compiled).isNotNull();
        }
    }

    @Test
    void aFunctionThatCannotBeLoweredDeclinesTheWholeKernel() {
        AccelNode projection =
                projectionCalling(
                        Allocating.class,
                        "(D)D",
                        List.of(new AccelInputRef(0, new DoubleType(false))));
        assertThat(AccelKernelGenerator.generate(projection, "udfbad")).isEmpty();
    }

    /**
     * An {@code ARRAY} column reaching a kernel: the last link of B4.
     *
     * <p>The row does not hold a {@code double[]} on a device; it holds a flat child and two
     * offsets. So the kernel takes the column as {@code l0_off} and {@code l0_val} and calls the
     * rewritten function with the buffer and this row's bounds. That it compiles is the proof that
     * the signature, the rewrite and the call all agree.
     */
    @Test
    void anArrayColumnBecomesOffsetsAndAFlatChild() throws Exception {
        RowType withArray =
                RowType.of(new ArrayType(false, new DoubleType(false)), new DoubleType(false));
        AccelNode projection =
                new AccelProject(
                        List.of(
                                new AccelUdfCall(
                                        TakingAnArray.class.getName(),
                                        "eval",
                                        "([D)D",
                                        List.of(
                                                new AccelInputRef(
                                                        0,
                                                        new ArrayType(
                                                                false, new DoubleType(false)))),
                                        new DoubleType(false))),
                        new AccelInput(withArray),
                        RowType.of(new DoubleType(false)));

        GpuKernelSource kernel = AccelKernelGenerator.generate(projection, "udflist").orElseThrow();
        assertThat(kernel.source())
                .contains("IntArray l0_off")
                .contains("DoubleArray l0_val")
                .contains("l0_val, l0_off.get(i), l0_off.get(i + 1)");

        try (GeneratedKernelEngine.Compiled compiled =
                GeneratedKernelEngine.compileStandalone(kernel)) {
            assertThat(compiled).isNotNull();
        }
    }

    /** A computed array has no staged buffer behind it, so it is refused rather than guessed at. */
    @Test
    void anArrayArgumentThatIsNotAStagedColumnIsRefused() {
        AccelNode projection =
                new AccelProject(
                        List.of(
                                new AccelUdfCall(
                                        TakingAnArray.class.getName(),
                                        "eval",
                                        "([D)D",
                                        List.of(
                                                new AccelLiteral(
                                                        1.0,
                                                        new ArrayType(
                                                                false, new DoubleType(false)))),
                                        new DoubleType(false))),
                        new AccelInput(INPUT),
                        RowType.of(new DoubleType(false)));
        assertThat(AccelKernelGenerator.generate(projection, "udflistbad")).isEmpty();
    }
}
