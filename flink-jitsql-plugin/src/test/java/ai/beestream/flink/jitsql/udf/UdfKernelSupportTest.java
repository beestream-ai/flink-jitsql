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
import org.apache.flink.table.types.logical.DoubleType;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * The planner names a method; this resolves it, or says why it cannot.
 *
 * <p>Every failure here has to be a sentence rather than an exception. This runs while a task is
 * starting, beside a perfectly good CPU operator, so a provider that throws takes down a job that
 * had a correct answer waiting.
 */
class UdfKernelSupportTest {

    private static AccelUdfCall call(Class<?> owner, String descriptor) {
        return new AccelUdfCall(
                owner.getName(), "eval", descriptor, List.of(), new DoubleType(false));
    }

    private static ClassLoader loader() {
        return UdfKernelSupportTest.class.getClassLoader();
    }

    @Test
    void anAdmissibleFunctionIsLoweredToAStaticMethod() throws Exception {
        AccelUdfCall udf = call(StaticUdfLoweringTest.CosineSim.class, "([D[D)D");
        UdfKernelSupport.Resolution resolution = UdfKernelSupport.resolve(udf, loader());
        assertThat(resolution.usable()).as("%s", resolution).isTrue();
        double[] a = {1.0, 2.0, 3.0};
        double[] b = {4.0, 5.0, 6.0};
        assertThat((double) resolution.method().orElseThrow().invoke(null, a, b))
                .isEqualTo(new StaticUdfLoweringTest.CosineSim().eval(a, b));
    }

    @Test
    void anInadmissibleBodyComesBackAsAReasonAndNotAnException() {
        UdfKernelSupport.Resolution resolution =
                UdfKernelSupport.resolve(
                        call(UdfEligibilityAnalyzerTest.Allocating.class, "(D)D"), loader());
        assertThat(resolution.usable()).isFalse();
        assertThat(resolution.construct()).get().isEqualTo(Construct.OBJECT_ALLOCATION);
        assertThat(resolution.because()).get().asString().contains("no heap");
    }

    @Test
    void aClassThatIsNotHereComesBackAsAReason() {
        AccelUdfCall missing =
                new AccelUdfCall(
                        "com.example.NotHere", "eval", "(D)D", List.of(), new DoubleType(false));
        UdfKernelSupport.Resolution resolution = UdfKernelSupport.resolve(missing, loader());
        assertThat(resolution.usable()).isFalse();
        assertThat(resolution.construct()).get().isEqualTo(Construct.UNREADABLE);
        assertThat(resolution.because()).get().asString().contains("not on this TaskManager");
    }

    /** A descriptor that no longer matches means the user recompiled; it is not a crash. */
    @Test
    void aDescriptorThatNoLongerMatchesComesBackAsAReason() {
        UdfKernelSupport.Resolution resolution =
                UdfKernelSupport.resolve(
                        call(StaticUdfLoweringTest.CosineSim.class, "(IIII)D"), loader());
        assertThat(resolution.usable()).isFalse();
        assertThat(resolution.because()).get().asString().contains("no longer declared");
    }

    @Test
    void theRightOverloadIsPickedByDescriptor() {
        UdfKernelSupport.Resolution byArray =
                UdfKernelSupport.resolve(
                        call(StaticUdfLoweringTest.CosineSim.class, "([D[D)D"), loader());
        UdfKernelSupport.Resolution byScalars =
                UdfKernelSupport.resolve(
                        call(StaticUdfLoweringTest.ManyLocals.class, "(DJI)D"), loader());
        assertThat(byArray.usable()).isTrue();
        assertThat(byScalars.usable()).isTrue();
        assertThat(byArray.method().orElseThrow().getParameterTypes())
                .containsExactly(double[].class, double[].class);
        assertThat(byScalars.method().orElseThrow().getParameterTypes())
                .containsExactly(double.class, long.class, int.class);
    }

    @Test
    void resolvingTwiceDoesTheWorkOnce() {
        AccelUdfCall udf = call(StaticUdfLoweringTest.CosineSim.class, "([D[D)D");
        UdfKernelSupport.Resolution first = UdfKernelSupport.resolve(udf, loader());
        UdfKernelSupport.Resolution second = UdfKernelSupport.resolve(udf, loader());
        assertThat(second).isSameAs(first);
    }
}
