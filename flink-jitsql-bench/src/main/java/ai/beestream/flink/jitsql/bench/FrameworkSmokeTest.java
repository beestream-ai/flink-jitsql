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
package ai.beestream.flink.jitsql.bench;

import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * Whether TornadoVM works on this machine at all — nothing of this project involved.
 *
 * <pre>
 *   flink run FrameworkSmokeTest.jar      # or: java @$TORNADOVM_HOME/tornado-argfile ...
 *   ... ends: SMOKE: OK
 * </pre>
 *
 * <h2>Why this is separate from the integration check</h2>
 *
 * <p>When {@link IntegrationCheck} reports everything running on the CPU, there are two very
 * different explanations and its output cannot tell them apart: TornadoVM is not working on this
 * machine, or TornadoVM is fine and this project is not reaching it. The first is far more common —
 * a driver, a JDK, a module path, an SDK built for another backend — and it is not a defect in
 * anything here.
 *
 * <p>So this runs one trivial kernel: fill an array with {@code a[i] * 2}. No Flink, no
 * TableEnvironment, no accelerator SPI, no cuDF, and no class of this project's except this one. If
 * it fails, the problem is the environment, and {@code scripts/preflight.sh} will usually name it.
 * If it passes and the integration check still says {@code cpu}, the problem is here and worth
 * reporting.
 *
 * <p>Borrowed from spark-jitudf's {@code RapidsSmokeTest}, which exists for the same reason and is
 * the first thing its README tells you to run when something is wrong.
 */
public final class FrameworkSmokeTest {

    private FrameworkSmokeTest() {}

    /** The whole kernel. Deliberately the least interesting one that can still be wrong. */
    public static void twice(FloatArray in, FloatArray out) {
        for (@Parallel int i = 0; i < in.getSize(); i++) {
            out.set(i, in.get(i) * 2.0f);
        }
    }

    public static void main(String[] args) {
        final int n = 1024;
        final FloatArray in = new FloatArray(n);
        final FloatArray out = new FloatArray(n);
        for (int i = 0; i < n; i++) {
            in.set(i, i);
        }

        System.out.println("flink-jitsql framework smoke test");
        try {
            final TaskGraph graph =
                    new TaskGraph("smoke")
                            .transferToDevice(DataTransferMode.FIRST_EXECUTION, in)
                            .task("twice", FrameworkSmokeTest::twice, in, out)
                            .transferToHost(DataTransferMode.EVERY_EXECUTION, out);
            final ImmutableTaskGraph snapshot = graph.snapshot();
            try (TornadoExecutionPlan plan = new TornadoExecutionPlan(snapshot)) {
                plan.execute();
            }
        } catch (Throwable t) {
            System.out.println("  the kernel did not run: " + t);
            System.out.println();
            System.out.println("SMOKE: FAILED");
            System.out.println(
                    "TornadoVM is not working on this machine, which is not a defect in"
                            + " flink-jitsql. Run scripts/preflight.sh.");
            System.exit(1);
            return;
        }

        int wrong = 0;
        for (int i = 0; i < n; i++) {
            if (out.get(i) != in.get(i) * 2.0f) {
                wrong++;
            }
        }
        if (wrong > 0) {
            System.out.printf("  the kernel ran and %d of %d values are wrong%n", wrong, n);
            System.out.println();
            System.out.println("SMOKE: FAILED");
            System.exit(1);
            return;
        }

        System.out.printf("  one kernel over %d floats, every value correct%n", n);
        System.out.println();
        System.out.println("SMOKE: OK");
    }
}
