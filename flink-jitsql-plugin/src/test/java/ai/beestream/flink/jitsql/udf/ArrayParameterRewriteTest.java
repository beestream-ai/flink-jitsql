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

import org.apache.flink.table.functions.ScalarFunction;

import org.junit.jupiter.api.Test;
import uk.ac.manchester.tornado.api.types.arrays.DoubleArray;

import java.lang.reflect.Method;
import java.util.Random;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * A UDF written against {@code double[]}, run against the flat buffer a device actually has.
 *
 * <p>Every assertion is {@code isEqualTo} on a {@code double}. The rewrite reorders no arithmetic
 * -- it only changes where the operands are fetched from -- so anything less than exact equality
 * would mean it had changed the computation, which is the one thing it must not do.
 *
 * <p>The rows are placed at varying offsets inside one shared buffer on purpose. An off-by-one in
 * the start offset is invisible when every row begins at zero, and is the most likely way for this
 * to be wrong.
 */
class ArrayParameterRewriteTest {

    public static final class SumOf extends ScalarFunction {
        public double eval(double[] a) {
            double total = 0;
            for (int i = 0; i < a.length; i++) {
                total += a[i] * (i + 1);
            }
            return total;
        }
    }

    public static final class CosineSim extends ScalarFunction {
        public double eval(double[] a, double[] b) {
            double dot = 0;
            double na = 0;
            double nb = 0;
            for (int i = 0; i < a.length; i++) {
                dot += a[i] * b[i];
                na += a[i] * a[i];
                nb += b[i] * b[i];
            }
            if (na == 0 || nb == 0) {
                return 0.0;
            }
            return dot / (Math.sqrt(na) * Math.sqrt(nb));
        }
    }

    public static final class Mixed extends ScalarFunction {
        public double eval(double[] a, double scale) {
            double total = 0;
            for (int i = 0; i < a.length; i++) {
                total += Math.sqrt(Math.abs(a[i])) * scale;
            }
            return total + a.length;
        }
    }

    private static Method evalOf(Class<?> type, Class<?>... parameters) {
        try {
            return type.getMethod("eval", parameters);
        } catch (NoSuchMethodException e) {
            throw new AssertionError(e);
        }
    }

    private static Method flattened(Class<?> type, Class<?>... parameters) {
        Method lowered = StaticUdfLowering.lower(evalOf(type, parameters));
        ArrayParameterRewrite.Rewritten rewritten = ArrayParameterRewrite.rewrite(lowered);
        assertThat(rewritten.usable()).as("%s", rewritten).isTrue();
        return rewritten.method();
    }

    private static DoubleArray buffer(double[] values) {
        DoubleArray array = new DoubleArray(values.length);
        for (int i = 0; i < values.length; i++) {
            array.set(i, values[i]);
        }
        return array;
    }

    @Test
    void theSignatureBecomesBufferStartAndEnd() {
        Method flat = flattened(SumOf.class, double[].class);
        assertThat(flat.getParameterTypes())
                .containsExactly(DoubleArray.class, int.class, int.class);
        assertThat(flat.getReturnType()).isEqualTo(double.class);
    }

    @Test
    void aRowReadFromTheMiddleOfABufferIsTheSameAsTheRowOnItsOwn() throws Exception {
        SumOf original = new SumOf();
        Method flat = flattened(SumOf.class, double[].class);
        Random random = new Random(99L);

        // One flat run holding many ragged rows, exactly as ArrayColumnGather stages them.
        double[] flatValues = new double[4096];
        for (int i = 0; i < flatValues.length; i++) {
            flatValues[i] = random.nextGaussian();
        }
        DoubleArray buffer = buffer(flatValues);

        int start = 0;
        for (int row = 0; row < 150 && start < flatValues.length - 40; row++) {
            int length = random.nextInt(32);
            int end = start + length;
            double[] asArray = new double[length];
            System.arraycopy(flatValues, start, asArray, 0, length);

            double expected = original.eval(asArray);
            double actual = (double) flat.invoke(null, buffer, start, end);
            assertThat(actual).as("row %s at [%s, %s)", row, start, end).isEqualTo(expected);
            start = end;
        }
    }

    @Test
    void twoArrayParametersBothGetTheirOwnBounds() throws Exception {
        CosineSim original = new CosineSim();
        Method flat = flattened(CosineSim.class, double[].class, double[].class);
        assertThat(flat.getParameterTypes())
                .containsExactly(
                        DoubleArray.class,
                        int.class,
                        int.class,
                        DoubleArray.class,
                        int.class,
                        int.class);

        Random random = new Random(1234L);
        double[] flatValues = new double[2048];
        for (int i = 0; i < flatValues.length; i++) {
            flatValues[i] = random.nextGaussian();
        }
        DoubleArray buffer = buffer(flatValues);

        for (int trial = 0; trial < 150; trial++) {
            int width = 1 + random.nextInt(24);
            int aStart = random.nextInt(flatValues.length - 2 * width - 1);
            int bStart = aStart + width;
            double[] a = new double[width];
            double[] b = new double[width];
            System.arraycopy(flatValues, aStart, a, 0, width);
            System.arraycopy(flatValues, bStart, b, 0, width);

            double expected = original.eval(a, b);
            double actual =
                    (double)
                            flat.invoke(
                                    null,
                                    buffer,
                                    aStart,
                                    aStart + width,
                                    buffer,
                                    bStart,
                                    bStart + width);
            assertThat(actual).as("trial %s", trial).isEqualTo(expected);
        }
    }

    @Test
    void aScalarParameterBesideAnArrayKeepsItsPlace() throws Exception {
        Mixed original = new Mixed();
        Method flat = flattened(Mixed.class, double[].class, double.class);
        assertThat(flat.getParameterTypes())
                .containsExactly(DoubleArray.class, int.class, int.class, double.class);

        Random random = new Random(5L);
        double[] flatValues = new double[512];
        for (int i = 0; i < flatValues.length; i++) {
            flatValues[i] = random.nextGaussian() * 5;
        }
        DoubleArray buffer = buffer(flatValues);

        int start = 0;
        for (int row = 0; row < 60 && start < 400; row++) {
            int length = random.nextInt(8);
            double scale = random.nextGaussian();
            double[] asArray = new double[length];
            System.arraycopy(flatValues, start, asArray, 0, length);
            assertThat((double) flat.invoke(null, buffer, start, start + length, scale))
                    .as("row %s", row)
                    .isEqualTo(original.eval(asArray, scale));
            start += length;
        }
    }

    @Test
    void anEmptyRowRunsZeroIterationsAndTouchesNothing() throws Exception {
        Method flat = flattened(SumOf.class, double[].class);
        DoubleArray buffer = buffer(new double[] {1.0, 2.0, 3.0});
        assertThat((double) flat.invoke(null, buffer, 2, 2))
                .isEqualTo(new SumOf().eval(new double[0]));
    }

    @Test
    void aFunctionWithNoArrayParametersIsHandedBackUnchanged() {
        Method lowered =
                StaticUdfLowering.lower(
                        evalOf(
                                StaticUdfLoweringTest.ManyLocals.class,
                                double.class,
                                long.class,
                                int.class));
        ArrayParameterRewrite.Rewritten rewritten = ArrayParameterRewrite.rewrite(lowered);
        assertThat(rewritten.usable()).isTrue();
        assertThat(rewritten.method()).isSameAs(lowered);
    }

    public static final class WritingToItsArgument extends ScalarFunction {
        public double eval(double[] a) {
            a[0] = 1.0;
            return a[0];
        }
    }

    /** A device view is read-only to a kernel, and writing through one is refused by name. */
    @Test
    void writingToAnArrayArgumentIsRefused() {
        Method lowered =
                StaticUdfLowering.lower(evalOf(WritingToItsArgument.class, double[].class));
        ArrayParameterRewrite.Rewritten rewritten = ArrayParameterRewrite.rewrite(lowered);
        assertThat(rewritten.usable()).isFalse();
        assertThat(rewritten.construct()).isEqualTo(Construct.OBJECT_ACCESS);
        assertThat(rewritten.because()).contains("writes to an array argument");
    }
}
