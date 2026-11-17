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

import static org.assertj.core.api.Assertions.assertThat;

/**
 * What a user's {@code ScalarFunction} may contain, decided from its bytecode.
 *
 * <p>The fixtures are ordinary Flink UDFs, written the way somebody would write them. None of them
 * mentions a device, an annotation or a hint -- which is the point: the query stays standard SQL
 * and the function stays the Java the user already had.
 */
class UdfEligibilityAnalyzerTest {

    private static Method eval(Class<?> type, Class<?>... parameters) {
        try {
            return type.getMethod("eval", parameters);
        } catch (NoSuchMethodException e) {
            throw new AssertionError(e);
        }
    }

    // ------------------------------------------------------------------------------------
    // The case this whole task exists for.
    // ------------------------------------------------------------------------------------

    /** A loop over an array: exactly what makes an expression tree unable to hold this. */
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

    @Test
    void aLoopOverAnArrayIsTheReasonToBeHereAndNotARefusal() {
        UdfEligibility result =
                UdfEligibilityAnalyzer.analyse(
                        eval(CosineSim.class, double[].class, double[].class));
        assertThat(result.eligible()).as("%s", result).isTrue();
        assertThat(result.loops()).isGreaterThanOrEqualTo(1);
        assertThat(result.arrayReads()).isGreaterThanOrEqualTo(3);
    }

    public static final class Branching extends ScalarFunction {
        public double eval(double x, double y) {
            if (x > y) {
                return Math.sqrt(x - y);
            } else if (x < y) {
                return Math.exp(y - x);
            }
            return 0.0;
        }
    }

    @Test
    void branchesAndSupportedMathAreFine() {
        assertThat(
                        UdfEligibilityAnalyzer.analyse(
                                        eval(Branching.class, double.class, double.class))
                                .eligible())
                .isTrue();
    }

    public static final class WithHelper extends ScalarFunction {
        private static double square(double v) {
            return v * v;
        }

        public double eval(double x) {
            double total = 0;
            for (int i = 0; i < 4; i++) {
                total += square(x + i);
            }
            return total;
        }
    }

    @Test
    void aStaticHelperInTheSameClassIsFollowed() {
        UdfEligibility result =
                UdfEligibilityAnalyzer.analyse(eval(WithHelper.class, double.class));
        assertThat(result.eligible()).as("%s", result).isTrue();
        assertThat(result.loops()).isGreaterThanOrEqualTo(1);
    }

    // ------------------------------------------------------------------------------------
    // Refusals. Each names the construct, and the query keeps the plan it had.
    // ------------------------------------------------------------------------------------

    public static final class Allocating extends ScalarFunction {
        public double eval(double x) {
            double[] scratch = new double[4];
            scratch[0] = x;
            return scratch[0];
        }
    }

    @Test
    void allocationIsRefusedByName() {
        UdfEligibility result =
                UdfEligibilityAnalyzer.analyse(eval(Allocating.class, double.class));
        assertThat(result.eligible()).isFalse();
        assertThat(result.construct()).get().isEqualTo(Construct.OBJECT_ALLOCATION);
        assertThat(result.detail()).get().asString().contains("no heap");
    }

    public static final class Stateful extends ScalarFunction {
        private double offset = 1.0;

        public double eval(double x) {
            return x + offset;
        }
    }

    @Test
    void readingInstanceStateIsRefusedByName() {
        UdfEligibility result = UdfEligibilityAnalyzer.analyse(eval(Stateful.class, double.class));
        assertThat(result.construct()).get().isEqualTo(Construct.OBJECT_ACCESS);
        assertThat(result.detail()).get().asString().contains("offset");
    }

    public static final class UsingUnsupportedMath extends ScalarFunction {
        public double eval(double x) {
            return Math.ulp(x);
        }
    }

    @Test
    void mathWithNoDeviceEquivalentIsRefusedByName() {
        UdfEligibility result =
                UdfEligibilityAnalyzer.analyse(eval(UsingUnsupportedMath.class, double.class));
        assertThat(result.construct()).get().isEqualTo(Construct.UNSUPPORTED_MATH);
        assertThat(result.detail()).get().asString().contains("ulp");
    }

    public static final class Throwing extends ScalarFunction {
        public double eval(double x) {
            if (x < 0) {
                throw new IllegalArgumentException();
            }
            return x;
        }
    }

    @Test
    void throwingIsRefusedBecauseAKernelCannotUnwind() {
        UdfEligibility result = UdfEligibilityAnalyzer.analyse(eval(Throwing.class, double.class));
        assertThat(result.eligible()).isFalse();
        assertThat(result.construct()).get().isIn(Construct.EXCEPTION, Construct.OBJECT_ALLOCATION);
    }

    public static final class Catching extends ScalarFunction {
        public double eval(double x) {
            try {
                return Math.sqrt(x);
            } catch (RuntimeException e) {
                return 0.0;
            }
        }
    }

    @Test
    void anExceptionHandlerIsRefusedByName() {
        UdfEligibility result = UdfEligibilityAnalyzer.analyse(eval(Catching.class, double.class));
        assertThat(result.construct()).get().isEqualTo(Construct.EXCEPTION);
        assertThat(result.detail()).get().asString().contains("unwind");
    }

    public static final class CallingOut extends ScalarFunction {
        public double eval(double x) {
            return Double.parseDouble(Double.toString(x));
        }
    }

    @Test
    void callingAnotherClassIsRefusedByName() {
        UdfEligibility result =
                UdfEligibilityAnalyzer.analyse(eval(CallingOut.class, double.class));
        assertThat(result.eligible()).isFalse();
        assertThat(result.construct()).get().isIn(Construct.UNSUPPORTED_CALL, Construct.STRING);
    }

    public static final class Recursive extends ScalarFunction {
        private static double down(double v) {
            return v <= 0 ? 0 : down(v - 1) + 1;
        }

        public double eval(double x) {
            return down(x);
        }
    }

    @Test
    void recursionIsRefusedByName() {
        UdfEligibility result = UdfEligibilityAnalyzer.analyse(eval(Recursive.class, double.class));
        assertThat(result.construct()).get().isEqualTo(Construct.RECURSION);
    }

    public static final class TakingAString extends ScalarFunction {
        public double eval(String s) {
            return s.length();
        }
    }

    @Test
    void aParameterWithNoDeviceRepresentationIsRefusedBeforeTheBodyIsRead() {
        UdfEligibility result =
                UdfEligibilityAnalyzer.analyse(eval(TakingAString.class, String.class));
        assertThat(result.construct()).get().isEqualTo(Construct.UNSUPPORTED_TYPE);
        assertThat(result.detail()).get().asString().contains("String");
    }

    public static final class ReturningAnArray extends ScalarFunction {
        public double[] eval(double x) {
            return null;
        }
    }

    @Test
    void returningAnArrayIsRefused() {
        UdfEligibility result =
                UdfEligibilityAnalyzer.analyse(eval(ReturningAnArray.class, double.class));
        assertThat(result.construct()).get().isEqualTo(Construct.UNSUPPORTED_TYPE);
    }

    /** Every refusal reads as a sentence, because somebody has to act on it. */
    @Test
    void everyRefusalSaysSomethingUseful() {
        for (UdfEligibility result :
                new UdfEligibility[] {
                    UdfEligibilityAnalyzer.analyse(eval(Allocating.class, double.class)),
                    UdfEligibilityAnalyzer.analyse(eval(Stateful.class, double.class)),
                    UdfEligibilityAnalyzer.analyse(eval(Catching.class, double.class)),
                    UdfEligibilityAnalyzer.analyse(eval(Recursive.class, double.class))
                }) {
            assertThat(result.eligible()).isFalse();
            assertThat(result.construct()).isPresent();
            assertThat(result.detail()).isPresent().get().asString().isNotBlank();
            assertThat(result.toString()).isNotBlank();
        }
    }
}
