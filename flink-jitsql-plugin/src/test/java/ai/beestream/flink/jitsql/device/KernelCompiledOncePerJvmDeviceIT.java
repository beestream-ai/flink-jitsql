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
package ai.beestream.flink.jitsql.device;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.TornadoExecutionResult;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.enums.ProfilerMode;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * That <em>TornadoVM's own</em> compilation of a kernel happens once per JVM, not once per subtask.
 *
 * <h2>Why this exists beside {@code KernelCompilationCacheTest}</h2>
 *
 * <p>That test counts {@code javac}: it proves the generated Java is compiled once and the class
 * shared. It says nothing about the step after it, where TornadoVM lowers that class to device code
 * — which is the expensive half, and which a host-side test cannot see at all.
 *
 * <p>The two are independent. The class could be cached perfectly while every subtask's execution
 * plan still drove a fresh device compilation, and the start-up cost P5.2 is about would be
 * untouched. So this measures the second half directly, on a device, through TornadoVM's own
 * profiler rather than through anything this project counts for itself.
 *
 * <h2>What is asserted, and why it is a ratio</h2>
 *
 * <p>Several execution plans are built from one kernel, as several subtasks in one TaskManager
 * would. The first pays for compilation; the rest must not. Measured on an RTX 4070 the first plan
 * reports ~186 ms of TornadoVM compiler time and the rest ~0.2 ms, about 770x less — the residue
 * being cache lookup and bookkeeping, not compilation.
 *
 * <p>The assertion is a ratio rather than a wall-clock bound because the absolute figures belong to
 * one card and one driver, and a test that pins those is a test that fails on the next machine for
 * no reason worth knowing. What does not vary is the shape: compile once, then nothing.
 */
final class KernelCompiledOncePerJvmDeviceIT {

    /** How many subtasks' worth of plans to build from the one kernel. */
    private static final int PLANS = 5;

    /**
     * The most a reused kernel may cost, as a fraction of a fresh compilation measured beside it.
     *
     * <p>Generous on purpose: a reuse that genuinely recompiled would land near 1.0, so a quarter
     * fails only on a real regression, not on a slow or busy machine.
     */
    private static final double TOLERANCE = 0.25;

    public static void twice(FloatArray in, FloatArray out) {
        for (@Parallel int i = 0; i < in.getSize(); i++) {
            out.set(i, in.get(i) * 2.0f);
        }
    }

    /** A different kernel, so its first compilation is a fresh one measured in this same JVM. */
    public static void thrice(FloatArray in, FloatArray out) {
        for (@Parallel int i = 0; i < in.getSize(); i++) {
            out.set(i, in.get(i) * 3.0f + 1.0f);
        }
    }

    @Test
    @DisplayName("the first plan compiles the kernel and the rest reuse it")
    void compilesOncePerJvm() throws Exception {
        DeviceAssumptions.requireDevice();

        final long[] compilerTime = new long[PLANS];
        for (int p = 0; p < PLANS; p++) {
            final int n = 4096;
            final FloatArray in = new FloatArray(n);
            final FloatArray out = new FloatArray(n);
            for (int i = 0; i < n; i++) {
                in.set(i, i);
            }
            final TaskGraph graph =
                    new TaskGraph("once")
                            .transferToDevice(DataTransferMode.EVERY_EXECUTION, in)
                            .task("twice", KernelCompiledOncePerJvmDeviceIT::twice, in, out)
                            .transferToHost(DataTransferMode.EVERY_EXECUTION, out);
            try (TornadoExecutionPlan plan =
                    new TornadoExecutionPlan(graph.snapshot()).withProfiler(ProfilerMode.SILENT)) {
                final TornadoExecutionResult result = plan.execute();
                compilerTime[p] = result.getProfilerResult().getTornadoCompilerTime();
                // The answer still has to be right; a cache that returned the wrong kernel would
                // otherwise sail through a timing assertion.
                for (int i = 0; i < n; i++) {
                    assertThat(out.get(i)).isEqualTo(in.get(i) * 2.0f);
                }
            }
            // The log the acceptance asks for: the numbers, not just a verdict.
            System.out.printf("  plan %d: TornadoVM compiler time %,d ns%n", p, compilerTime[p]);
        }

        // A fresh compilation, right here, in whatever state this JVM is now in.
        final long freshCompile = compileFresh();
        System.out.printf("  a different kernel, compiled fresh: %,d ns%n", freshCompile);

        assertThat(compilerTime[0])
                .as("the first plan reported no compilation at all, so this measured nothing")
                .isPositive();
        assertThat(freshCompile)
                .as("the baseline kernel reported no compilation, so there is nothing to compare")
                .isPositive();

        final long budget = (long) (freshCompile * TOLERANCE);
        for (int p = 1; p < PLANS; p++) {
            assertThat(compilerTime[p])
                    .as(
                            "plan %d spent %,d ns against the %,d ns a fresh compilation costs in"
                                    + " this same JVM: the kernel is being compiled again per plan,"
                                    + " so at parallelism p a TaskManager pays for it p times",
                            p, compilerTime[p], freshCompile)
                    .isLessThanOrEqualTo(budget);
        }
    }

    /** Compiles a kernel this JVM has not seen, and reports what TornadoVM spent doing it. */
    private static long compileFresh() throws Exception {
        final int n = 4096;
        final FloatArray in = new FloatArray(n);
        final FloatArray out = new FloatArray(n);
        for (int i = 0; i < n; i++) {
            in.set(i, i);
        }
        final TaskGraph graph =
                new TaskGraph("fresh")
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION, in)
                        .task("thrice", KernelCompiledOncePerJvmDeviceIT::thrice, in, out)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, out);
        try (TornadoExecutionPlan plan =
                new TornadoExecutionPlan(graph.snapshot()).withProfiler(ProfilerMode.SILENT)) {
            return plan.execute().getProfilerResult().getTornadoCompilerTime();
        }
    }
}
