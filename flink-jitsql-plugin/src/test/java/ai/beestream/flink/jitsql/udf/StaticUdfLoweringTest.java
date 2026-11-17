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

import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.Random;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * The lowered method must compute exactly what the user's method computes.
 *
 * <p>"Exactly" is the word that matters and it is checkable here without a device, because the
 * transform is a bytecode copy rather than a translation: the same instructions in the same order
 * over the same values. Every assertion below is {@code isEqualTo} on a {@code double} and none is
 * a tolerance, which is the whole claim. If a renumbered local ever went wrong, the arithmetic
 * would change and these would fail long before a GPU was involved.
 */
class StaticUdfLoweringTest {

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

    public static final class ManyLocals extends ScalarFunction {
        public double eval(double x, long n, int k) {
            double acc = 0;
            long guard = n;
            int steps = k;
            for (int i = 0; i < steps; i++) {
                double term = Math.sin(x + i) * Math.cos(x - i);
                acc += term / (1.0 + i);
                guard += i;
            }
            return acc + guard;
        }
    }

    public static final class WithStaticHelper extends ScalarFunction {
        private static double weight(double v, int i) {
            return v / (1.0 + i);
        }

        public double eval(double x, int k) {
            double total = 0;
            for (int i = 0; i < k; i++) {
                total += weight(Math.sqrt(x + i), i);
            }
            return total;
        }
    }

    private static Method eval(Class<?> type, Class<?>... parameters) {
        try {
            return type.getMethod("eval", parameters);
        } catch (NoSuchMethodException e) {
            throw new AssertionError(e);
        }
    }

    @Test
    void theLoweredMethodIsStaticAndKeepsItsSignature() {
        Method lowered =
                StaticUdfLowering.lower(eval(CosineSim.class, double[].class, double[].class));
        assertThat(Modifier.isStatic(lowered.getModifiers())).isTrue();
        assertThat(lowered.getName()).isEqualTo("eval");
        assertThat(lowered.getParameterTypes()).containsExactly(double[].class, double[].class);
        assertThat(lowered.getReturnType()).isEqualTo(double.class);
        assertThat(lowered.getDeclaringClass()).isNotEqualTo(CosineSim.class);
    }

    /** The acceptance test: same answer, to the last bit, over data it has not seen. */
    @Test
    void aLoopOverArraysComputesTheSameDoubleToTheLastBit() throws Exception {
        CosineSim original = new CosineSim();
        Method lowered =
                StaticUdfLowering.lower(eval(CosineSim.class, double[].class, double[].class));
        Random random = new Random(20261117L);
        for (int trial = 0; trial < 200; trial++) {
            int width = 1 + random.nextInt(64);
            double[] a = new double[width];
            double[] b = new double[width];
            for (int i = 0; i < width; i++) {
                a[i] = random.nextGaussian();
                b[i] = random.nextGaussian();
            }
            double expected = original.eval(a, b);
            double actual = (double) lowered.invoke(null, a, b);
            assertThat(actual).as("trial %s, width %s", trial, width).isEqualTo(expected);
        }
    }

    @Test
    void theEmptyArrayCaseIsCarriedOverToo() throws Exception {
        CosineSim original = new CosineSim();
        Method lowered =
                StaticUdfLowering.lower(eval(CosineSim.class, double[].class, double[].class));
        double[] empty = new double[0];
        assertThat((double) lowered.invoke(null, empty, empty))
                .isEqualTo(original.eval(empty, empty));
    }

    /**
     * Several locals of several widths, which is where a renumbering bug would show.
     *
     * <p>A {@code long} occupies two slots and a {@code double} two more, so shifting indices down
     * by one is only correct if every reference shifts together. Mixed widths and an {@code iinc}
     * in the loop header are the shape that catches a half-done shift.
     */
    @Test
    void localsOfMixedWidthsSurviveTheShift() throws Exception {
        ManyLocals original = new ManyLocals();
        Method lowered =
                StaticUdfLowering.lower(
                        eval(ManyLocals.class, double.class, long.class, int.class));
        Random random = new Random(7L);
        for (int trial = 0; trial < 200; trial++) {
            double x = random.nextGaussian() * 10;
            long n = random.nextLong() % 1000;
            int k = random.nextInt(32);
            assertThat((double) lowered.invoke(null, x, n, k))
                    .as("x=%s n=%s k=%s", x, n, k)
                    .isEqualTo(original.eval(x, n, k));
        }
    }

    @Test
    void aStaticHelperTravelsWithTheBodyAndStillComputesTheSame() throws Exception {
        WithStaticHelper original = new WithStaticHelper();
        Method lowered =
                StaticUdfLowering.lower(eval(WithStaticHelper.class, double.class, int.class));
        for (int k = 0; k < 20; k++) {
            assertThat((double) lowered.invoke(null, 3.25, k)).isEqualTo(original.eval(3.25, k));
        }
        // The helper came with it, so nothing on a device has to reach back into the user's class.
        assertThat(lowered.getDeclaringClass().getDeclaredMethods())
                .extracting(Method::getName)
                .contains("weight");
    }

    /**
     * Nothing is lowered that was not admitted first.
     *
     * <p>The shift is sound only because the analyser has already refused every body that touches
     * {@code this}. Lowering an unanalysed method would renumber a live slot and change the
     * arithmetic quietly, so the gate is an exception rather than a comment.
     */
    @Test
    void anIneligibleBodyIsNotLoweredAtAll() {
        assertThatThrownBy(
                        () ->
                                StaticUdfLowering.lower(
                                        eval(
                                                UdfEligibilityAnalyzerTest.Stateful.class,
                                                double.class)))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("not admitted")
                .hasMessageContaining("OBJECT_ACCESS");
    }

    @Test
    void loweringTwiceGivesTwoWorkingCopies() throws Exception {
        Method first =
                StaticUdfLowering.lower(eval(CosineSim.class, double[].class, double[].class));
        Method second =
                StaticUdfLowering.lower(eval(CosineSim.class, double[].class, double[].class));
        double[] a = {1.0, 2.0, 3.0};
        double[] b = {4.0, 5.0, 6.0};
        assertThat((double) second.invoke(null, a, b)).isEqualTo((double) first.invoke(null, a, b));
    }
}
