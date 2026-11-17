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

import ai.beestream.flink.jitsql.catalogue.Route;
import ai.beestream.flink.jitsql.catalogue.RoutingContext;
import ai.beestream.flink.jitsql.codegen.AccelKernelGenerator;
import ai.beestream.flink.jitsql.codegen.GpuKernelSource;
import ai.beestream.flink.jitsql.device.BoundView;
import ai.beestream.flink.jitsql.device.CachedPlan;
import ai.beestream.flink.jitsql.device.CrossRuntimeOrdering;
import ai.beestream.flink.jitsql.device.DeviceBinding;
import ai.beestream.flink.jitsql.device.ElementKind;
import ai.beestream.flink.jitsql.device.ExecutionPlanCache;
import ai.beestream.flink.jitsql.device.PlanShape;
import ai.beestream.flink.jitsql.device.StageBuffers;
import ai.beestream.flink.jitsql.operator.GeneratedKernelEngine;
import ai.beestream.flink.jitsql.region.RegionAssembler;
import ai.beestream.flink.jitsql.region.RegionPlan;
import ai.beestream.flink.jitsql.region.Stage;

import org.apache.flink.table.accelerator.AccelAggCall;
import org.apache.flink.table.accelerator.AccelAggFunction;
import org.apache.flink.table.accelerator.AccelAggregate;
import org.apache.flink.table.accelerator.AccelCall;
import org.apache.flink.table.accelerator.AccelFunction;
import org.apache.flink.table.accelerator.AccelInput;
import org.apache.flink.table.accelerator.AccelInputRef;
import org.apache.flink.table.accelerator.AccelNode;
import org.apache.flink.table.accelerator.AccelProject;
import org.apache.flink.table.accelerator.AccelWorkProfile;
import org.apache.flink.table.types.logical.DoubleType;
import org.apache.flink.table.types.logical.RowType;

import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.DoubleArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.cudf.Cudf;

import java.lang.reflect.Method;
import java.util.List;
import java.util.Random;

/**
 * The three acceptance criteria that can only be settled on a device.
 *
 * <ol>
 *   <li><b>B2</b> -- a generated kernel reads a buffer this code did not allocate, with no copy.
 *   <li><b>A2</b> -- a shape nobody wrote a region for runs on the device, with the same answers.
 *   <li><b>B1</b> -- the plan cache holds real execution plans, builds one per shape, and the
 *       answers are right when a cached plan is reused.
 * </ol>
 *
 * <p>Each prints its own verdict and the process fails if any does.
 */
public final class DeviceRegionVerification {

    private static final int ROWS = 8192;
    private static final int SHORT_ROWS = 777;

    private DeviceRegionVerification() {}

    public static void main(String[] args) throws Throwable {
        String libcudart =
                System.getProperty(
                        "jitsql.libcudart",
                        "/home/mary/gpu-bench-data/flink-llm/cuda13/targets/x86_64-linux/lib/"
                                + "libcudart.so.13");

        boolean b2 = verifyForeignBuffer(libcudart);
        boolean a2 = verifyAssembledRegion();
        boolean b1 = verifyPlanCache();

        System.out.println();
        System.out.println("B2 foreign buffer bound and read: " + verdict(b2));
        System.out.println("A2 assembled region on device:    " + verdict(a2));
        System.out.println("B1 plan cache with real plans:    " + verdict(b1));
        System.out.println(CrossRuntimeOrdering.line());
        boolean passed = b2 && a2 && b1;
        System.out.println(passed ? "DEVICE VERIFICATION: PASS" : "DEVICE VERIFICATION: FAIL");
        if (!passed) {
            System.exit(1);
        }
    }

    private static String verdict(boolean ok) {
        return ok ? "PASS" : "FAIL";
    }

    // ---------------------------------------------------------------------------------------
    // B2
    // ---------------------------------------------------------------------------------------

    /**
     * Binds a kernel to memory {@code cudaMalloc} gave us and nothing here allocated.
     *
     * <p>Argument order follows the generated signature, which is inputs, then outputs, then the
     * row count -- with list columns appended after the outputs. Getting it wrong binds the view to
     * the output parameter, and TornadoVM says so rather than silently overwriting the foreign
     * buffer: "the task writes to this device view, but it is bound read-only".
     *
     * <p>Nothing is transferred to the device for the input: the view carries the foreign address,
     * and the task graph is never told to copy anything into it. That the kernel's output is right
     * is the proof it read the foreign bytes where they already were.
     */
    private static boolean verifyForeignBuffer(String libcudart) throws Throwable {
        double[] host = new double[ROWS];
        Random random = new Random(11L);
        for (int i = 0; i < ROWS; i++) {
            host[i] = Math.abs(random.nextGaussian()) * 100.0;
        }

        GpuKernelSource source = sqrtTimesTwo();
        System.out.println(
                "B2: kernel signature: "
                        + source.source()
                                .lines()
                                .filter(l -> l.contains("static void"))
                                .findFirst()
                                .orElse("?"));
        try (ForeignDeviceBuffer foreign = ForeignDeviceBuffer.of(host, libcudart);
                GeneratedKernelEngine.Compiled compiled =
                        GeneratedKernelEngine.compileStandalone(source);
                StageBuffers buffers = new StageBuffers()) {

            BoundView input =
                    buffers.view(
                            ai.beestream.flink.jitsql.device.BufferKey.output(0),
                            ElementKind.DOUBLE,
                            ROWS);
            DoubleArray out = new DoubleArray(ROWS);
            IntArray rows = new IntArray(1);
            rows.set(0, ROWS);

            TaskGraph graph =
                    new TaskGraph("foreign")
                            .transferToDevice(DataTransferMode.EVERY_EXECUTION, out, rows)
                            .task("kernel", compiled.entry(), input.argument(), out, rows)
                            .transferToHost(DataTransferMode.EVERY_EXECUTION, out);

            // The @Parallel bound is rows.get(0), which is not a compile-time constant, so the
            // launch domain has to come from a grid rather than be inferred.
            GridScheduler scheduler = new GridScheduler("foreign.kernel", new WorkerGrid1D(ROWS));
            try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
                DeviceBinding binding =
                        DeviceBinding.describe(
                                "cudaMalloc",
                                foreign.address(),
                                foreign.byteLength(),
                                ROWS,
                                ElementKind.DOUBLE);
                buffers.bind(
                        plan,
                        ai.beestream.flink.jitsql.device.BufferKey.output(0),
                        binding,
                        true,
                        CrossRuntimeOrdering.IN_GRAPH);
                System.out.printf(
                        "B2: bound a view to 0x%x (%d bytes) that this code did not allocate%n",
                        foreign.address(), foreign.byteLength());
                plan.withGridScheduler(scheduler).execute();
                buffers.unbindAll(plan);
            }

            int mismatches = 0;
            for (int i = 0; i < ROWS; i++) {
                double expected = Math.sqrt(host[i]) * 2.0;
                if (Double.compare(expected, out.get(i)) != 0) {
                    if (mismatches == 0) {
                        System.out.printf("B2: row %d cpu %s device %s%n", i, expected, out.get(i));
                    }
                    mismatches++;
                }
            }
            System.out.printf("B2: %d rows, %d mismatches%n", ROWS, mismatches);
            return mismatches == 0;
        }
    }

    // ---------------------------------------------------------------------------------------
    // A2
    // ---------------------------------------------------------------------------------------

    /**
     * Runs a shape nobody wrote a region class for, with the stage list coming from the assembler.
     *
     * <p>A projection feeding an ungrouped sum over a staged input. Every hand-written region is
     * rooted at a scan and reads its own Parquet; none of them covers this, and no class is added
     * here to make it work -- the graph below is built by walking {@link RegionPlan#stages()} and
     * asking each stage what it is.
     */
    private static boolean verifyAssembledRegion() throws Exception {
        RowType input = RowType.of(new DoubleType(false));
        AccelNode projection =
                new AccelProject(
                        List.of(
                                new AccelCall(
                                        AccelFunction.SQRT,
                                        List.of(new AccelInputRef(0, new DoubleType(false))),
                                        new DoubleType(false))),
                        new AccelInput(input),
                        RowType.of(new DoubleType(false)));
        AccelNode subtree =
                new AccelAggregate(
                        new int[] {},
                        List.of(new AccelAggCall(AccelAggFunction.SUM, 0, new DoubleType(false))),
                        projection,
                        RowType.of(new DoubleType(false)));

        int[] opsPerRow = new int[AccelFunction.values().length];
        opsPerRow[AccelFunction.SQRT.ordinal()] = 1;
        AccelWorkProfile work = new AccelWorkProfile(opsPerRow, 8, 8, ROWS);

        RegionPlan region =
                RegionAssembler.assemble(
                        subtree, work, RoutingContext.fullyAvailable(1L << 30), ROWS);
        if (!region.accepted()) {
            System.out.println("A2: the assembler declined: " + region);
            return false;
        }
        System.out.println("A2: assembled stages:");
        for (Stage stage : region.stages()) {
            System.out.println("    " + stage);
        }

        double[] host = new double[ROWS];
        Random random = new Random(22L);
        for (int i = 0; i < ROWS; i++) {
            host[i] = Math.abs(random.nextGaussian()) * 10.0;
        }

        DoubleArray in = new DoubleArray(ROWS);
        for (int i = 0; i < ROWS; i++) {
            in.set(i, host[i]);
        }
        DoubleArray computed = new DoubleArray(ROWS);
        DoubleArray sum = new DoubleArray(1);
        IntArray rows = new IntArray(1);
        rows.set(0, ROWS);

        GpuKernelSource kernelSource =
                AccelKernelGenerator.generate(projection, "assembled")
                        .orElseThrow(() -> new IllegalStateException("no kernel for the stage"));

        try (GeneratedKernelEngine.Compiled compiled =
                GeneratedKernelEngine.compileStandalone(kernelSource)) {
            TaskGraph graph =
                    new TaskGraph("assembled")
                            .transferToDevice(DataTransferMode.EVERY_EXECUTION, in, rows)
                            .transferToDevice(DataTransferMode.EVERY_EXECUTION, computed, sum);
            // One task per stage, in the order the assembler put them.
            for (Stage stage : region.stages()) {
                if (stage.route().kind() == Route.Kind.KERNEL) {
                    graph = graph.task("s" + stage.index(), compiled.entry(), in, computed, rows);
                } else {
                    String primitive = stage.route().primitive().orElseThrow().name();
                    if (!"reduce".equals(primitive)) {
                        System.out.println("A2: no emitter for " + primitive);
                        return false;
                    }
                    graph =
                            graph.libraryTask(
                                    "s" + stage.index(), Cudf::reduce, ROWS, 0, computed, sum);
                }
            }
            graph = graph.transferToHost(DataTransferMode.EVERY_EXECUTION, computed, sum);
            GridScheduler scheduler = new GridScheduler();
            for (Stage stage : region.stages()) {
                if (stage.route().kind() == Route.Kind.KERNEL) {
                    scheduler.addWorkerGrid("assembled.s" + stage.index(), new WorkerGrid1D(ROWS));
                }
            }
            try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
                plan.withGridScheduler(scheduler).execute();
            }
        }

        int mismatches = 0;
        double cpuSum = 0;
        for (int i = 0; i < ROWS; i++) {
            double expected = Math.sqrt(host[i]);
            cpuSum += expected;
            if (Double.compare(expected, computed.get(i)) != 0) {
                mismatches++;
            }
        }
        double ulps =
                Math.abs(cpuSum - sum.get(0))
                        / Math.ulp(Math.max(Math.abs(cpuSum), Math.abs(sum.get(0))));
        System.out.printf(
                "A2: %d rows, %d elementwise mismatches; sum cpu %s device %s (%.1f ulps)%n",
                ROWS, mismatches, cpuSum, sum.get(0), ulps);
        return mismatches == 0 && ulps <= 256;
    }

    // ---------------------------------------------------------------------------------------
    // B1
    // ---------------------------------------------------------------------------------------

    /**
     * Drives real execution plans through the cache at two row counts.
     *
     * <p>The second query of a shape must build nothing, and the short final batch must build its
     * own plan rather than reuse the full one -- which is the whole reason the row count is in the
     * key. Both answers are checked, because a cache that returns the wrong plan does not fail; it
     * computes over the wrong number of rows.
     */
    private static boolean verifyPlanCache() throws Exception {
        ExecutionPlanCache.releaseThread();
        GpuKernelSource source = sqrtTimesTwo();
        try (GeneratedKernelEngine.Compiled compiled =
                GeneratedKernelEngine.compileStandalone(source)) {
            Method entry = compiled.entry();
            long missesBefore = ExecutionPlanCache.misses();

            boolean ok = true;
            ok &= runThroughCache(entry, ROWS);
            ok &= runThroughCache(entry, ROWS);
            ok &= runThroughCache(entry, SHORT_ROWS);
            ok &= runThroughCache(entry, SHORT_ROWS);

            long built = ExecutionPlanCache.misses() - missesBefore;
            System.out.printf(
                    "B1: four executions at two row counts built %d plans (want 2)%n", built);
            ExecutionPlanCache.releaseThread();
            return ok && built == 2;
        }
    }

    private static boolean runThroughCache(Method entry, int rowCount) {
        CachedPlan cached =
                ExecutionPlanCache.get(
                        "sqrt-times-two",
                        PlanShape.ofRows(rowCount),
                        () -> buildPlan(entry, rowCount));
        StageBuffers buffers = cached.buffers();
        DoubleArray in =
                (DoubleArray)
                        buffers.argument(ai.beestream.flink.jitsql.device.BufferKey.output(0));
        DoubleArray out =
                (DoubleArray)
                        buffers.argument(ai.beestream.flink.jitsql.device.BufferKey.output(1));
        for (int i = 0; i < rowCount; i++) {
            in.set(i, i + 1.0);
        }
        cached.plan().execute();
        for (int i = 0; i < rowCount; i++) {
            double expected = Math.sqrt(i + 1.0) * 2.0;
            if (Double.compare(expected, out.get(i)) != 0) {
                System.out.printf(
                        "B1: at %d rows, row %d cpu %s device %s%n",
                        rowCount, i, expected, out.get(i));
                return false;
            }
        }
        return true;
    }

    private static CachedPlan buildPlan(Method entry, int rowCount) {
        StageBuffers buffers = new StageBuffers();
        DoubleArray in =
                (DoubleArray)
                        buffers.owned(
                                ai.beestream.flink.jitsql.device.BufferKey.output(0),
                                ElementKind.DOUBLE,
                                rowCount);
        DoubleArray out =
                (DoubleArray)
                        buffers.owned(
                                ai.beestream.flink.jitsql.device.BufferKey.output(1),
                                ElementKind.DOUBLE,
                                rowCount);
        IntArray rows =
                (IntArray)
                        buffers.owned(
                                ai.beestream.flink.jitsql.device.BufferKey.scratch(0, "rows"),
                                ElementKind.INT,
                                1);
        rows.set(0, rowCount);
        TaskGraph graph =
                new TaskGraph("cached" + rowCount)
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION, in, rows)
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION, out)
                        .task("kernel", entry, in, out, rows)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, out);
        GridScheduler scheduler =
                new GridScheduler("cached" + rowCount + ".kernel", new WorkerGrid1D(rowCount));
        return new CachedPlan(
                new TornadoExecutionPlan(graph.snapshot()).withGridScheduler(scheduler),
                buffers,
                PlanShape.ofRows(rowCount));
    }

    // ---------------------------------------------------------------------------------------

    /** {@code SELECT SQRT(c0) * 2}, as a generated kernel. */
    private static GpuKernelSource sqrtTimesTwo() {
        RowType input = RowType.of(new DoubleType(false));
        AccelNode projection =
                new AccelProject(
                        List.of(
                                new AccelCall(
                                        AccelFunction.TIMES,
                                        List.of(
                                                new AccelCall(
                                                        AccelFunction.SQRT,
                                                        List.of(
                                                                new AccelInputRef(
                                                                        0, new DoubleType(false))),
                                                        new DoubleType(false)),
                                                new org.apache.flink.table.accelerator.AccelLiteral(
                                                        2.0, new DoubleType(false))),
                                        new DoubleType(false))),
                        new AccelInput(input),
                        RowType.of(new DoubleType(false)));
        return AccelKernelGenerator.generate(projection, "sqrt2")
                .orElseThrow(() -> new IllegalStateException("no kernel generated"));
    }
}
