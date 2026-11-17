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

import ai.beestream.flink.jitsql.catalogue.CudfCapabilityCatalogue;
import ai.beestream.flink.jitsql.catalogue.RefusalCode;
import ai.beestream.flink.jitsql.catalogue.Route;
import ai.beestream.flink.jitsql.catalogue.RoutingContext;
import ai.beestream.flink.jitsql.codegen.AccelKernelGenerator;
import ai.beestream.flink.jitsql.codegen.GpuKernelSource;
import ai.beestream.flink.jitsql.device.BoundView;
import ai.beestream.flink.jitsql.device.BufferKey;
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
import org.apache.flink.table.accelerator.AccelLiteral;
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
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * One run, one line per task, against each task's own acceptance criterion.
 *
 * <p>Separate checks in separate programs leave the question "did they all pass together" open, and
 * that question has an answer worth having: everything here runs in one JVM against one device, and
 * the table at the end is the whole result.
 *
 * <p>Preconditions are checked first and loudly. The cuDF stages need the RAPIDS libraries on
 * {@code LD_LIBRARY_PATH}, and without them an earlier version of this simply stopped after staging
 * with no message -- which is the worst way for a verification to fail, because it looks like a
 * crash rather than a missing dependency.
 */
public final class DeviceAcceptance {

    private static final int ROWS = 8192;
    private static final int SHORT_ROWS = 777;

    private static final Map<String, String> RESULTS = new LinkedHashMap<>();

    private DeviceAcceptance() {}

    public static void main(String[] args) throws Throwable {
        preflight();

        check("A1 catalogue routes every node, or names a refusal", DeviceAcceptance::checkA1);
        check("A4 a subtree with an unroutable stage is declined whole", DeviceAcceptance::checkA4);
        check("B6 a refusal carries a code and a sentence", DeviceAcceptance::checkB6);
        check("A3 a region wires itself by key, no field names", DeviceAcceptance::checkA3);
        check("A2 a shape nobody wrote a region for, exact answers", DeviceAcceptance::checkA2);
        check("B2 a kernel reads a foreign buffer, no copy", DeviceAcceptance::checkB2);
        check("B1 the plan cache holds real plans, one per shape", DeviceAcceptance::checkB1);
        check("B4 an ARRAY column staged as offsets and a child", DeviceAcceptance::checkB4B5);
        check("B5 a user's ScalarFunction runs as a kernel", () -> "see B4");
        check("B3 cross-runtime ordering is counted", DeviceAcceptance::checkB3);

        System.out.println();
        System.out.println("=== device acceptance ===");
        boolean passed = true;
        for (Map.Entry<String, String> result : RESULTS.entrySet()) {
            boolean ok = !result.getValue().startsWith("FAIL");
            passed &= ok;
            System.out.printf("  %-58s %s%n", result.getKey(), result.getValue());
        }
        System.out.println();
        System.out.println(passed ? "DEVICE ACCEPTANCE: PASS" : "DEVICE ACCEPTANCE: FAIL");
        if (!passed) {
            System.exit(1);
        }
    }

    @FunctionalInterface
    private interface Check {
        String run() throws Throwable;
    }

    private static void check(String name, Check check) {
        try {
            RESULTS.put(name, check.run());
        } catch (Throwable t) {
            RESULTS.put(name, "FAIL " + t);
        }
    }

    /** Fails with a sentence rather than a stack trace when a dependency is missing. */
    private static void preflight() {
        if (!Cudf.isParquetAvailable() && !Cudf.isColumnsAvailable()) {
            System.err.println(
                    "the cuDF binding is not usable: put the six RAPIDS directories on"
                            + " LD_LIBRARY_PATH (scripts/cudf-env.sh does it) and run again."
                            + " Without them the cuDF stages stop with no message at all.");
            System.exit(2);
        }
    }

    // ---------------------------------------------------------------------------------------

    private static AccelWorkProfile work(long rows) {
        int[] opsPerRow = new int[AccelFunction.values().length];
        opsPerRow[AccelFunction.SQRT.ordinal()] = 1;
        return new AccelWorkProfile(opsPerRow, 8, 8, rows);
    }

    private static AccelNode sqrtOf(AccelNode input, RowType out) {
        return new AccelProject(
                List.of(
                        new AccelCall(
                                AccelFunction.SQRT,
                                List.of(new AccelInputRef(0, new DoubleType(false))),
                                new DoubleType(false))),
                input,
                out);
    }

    private static String checkA1() {
        RoutingContext context = RoutingContext.fullyAvailable(1L << 30);
        RowType one = RowType.of(new DoubleType(false));
        Route project =
                CudfCapabilityCatalogue.route(
                        sqrtOf(new AccelInput(one), one), work(ROWS), context);
        if (project.kind() != Route.Kind.KERNEL) {
            return "FAIL a projection routed to " + project;
        }
        return "PASS projection -> generated kernel, by capability";
    }

    private static String checkA4() {
        RowType one = RowType.of(new DoubleType(false));
        AccelNode bad =
                new AccelAggregate(
                        new int[] {0, 1},
                        List.of(new AccelAggCall(AccelAggFunction.SUM, 0, new DoubleType(false))),
                        sqrtOf(new AccelInput(one), one),
                        one);
        RegionPlan plan =
                RegionAssembler.assemble(
                        bad, work(ROWS), RoutingContext.fullyAvailable(1L << 30), ROWS);
        if (plan.accepted()) {
            return "FAIL an unroutable stage was admitted";
        }
        return "PASS declined whole: " + plan.because().orElse("");
    }

    private static String checkB6() {
        RowType one = RowType.of(new DoubleType(false));
        AccelNode bad =
                new AccelAggregate(
                        new int[] {0, 1},
                        List.of(new AccelAggCall(AccelAggFunction.SUM, 0, new DoubleType(false))),
                        new AccelInput(one),
                        one);
        Route route =
                CudfCapabilityCatalogue.route(
                        bad, work(ROWS), RoutingContext.fullyAvailable(1L << 30));
        if (route.code().orElse(null) != RefusalCode.PRIMITIVE_SIGNATURE_MISMATCH
                || route.because().orElse("").isBlank()) {
            return "FAIL refusal was " + route;
        }
        return "PASS " + route.code().orElseThrow() + ", with a sentence";
    }

    private static String checkA3() {
        try (StageBuffers buffers = new StageBuffers()) {
            buffers.owned(BufferKey.output(0), ElementKind.DOUBLE, 16);
            buffers.owned(BufferKey.output(1), ElementKind.DOUBLE, 16);
            if (buffers.argument(BufferKey.output(0)) == buffers.argument(BufferKey.output(1))) {
                return "FAIL two stages share one buffer";
            }
            return "PASS " + buffers.registered() + " wired by key";
        }
    }

    /**
     * A2: two chained projections over a staged input.
     *
     * <p>Chosen so that every answer is exactly comparable. A region rooted at a reduction cannot
     * be, because a parallel tree sums in a different order from a sequential loop and the two
     * differ in the last place -- a real property of the arithmetic rather than a defect, but it
     * makes "the same answers" a statement about tolerance instead of equality. Two projections
     * have no such ambiguity: every row is checked with {@code Double.compare}.
     *
     * <p>No hand-written region covers it. All six are rooted at a scan and read their own Parquet;
     * this is rooted at a staged input, and the task graph below is built by walking {@link
     * RegionPlan#stages()} rather than by knowing what the shape is.
     */
    private static String checkA2() throws Exception {
        RowType one = RowType.of(new DoubleType(false));
        AccelNode inner = sqrtOf(new AccelInput(one), one);
        AccelNode outer =
                new AccelProject(
                        List.of(
                                new AccelCall(
                                        AccelFunction.TIMES,
                                        List.of(
                                                new AccelInputRef(0, new DoubleType(false)),
                                                new AccelLiteral(3.0, new DoubleType(false))),
                                        new DoubleType(false))),
                        inner,
                        one);

        RegionPlan region =
                RegionAssembler.assemble(
                        outer, work(ROWS), RoutingContext.fullyAvailable(1L << 30), ROWS);
        if (!region.accepted() || region.stages().size() != 2) {
            return "FAIL assembler gave " + region;
        }

        double[] host = new double[ROWS];
        Random random = new Random(33L);
        for (int i = 0; i < ROWS; i++) {
            host[i] = Math.abs(random.nextGaussian()) * 50.0;
        }

        // A kernel is generated per stage, each over its own staged input. The assembler plans the
        // chain; the generator only ever sees one projection at a time and requires its input to
        // be a leaf, so the outer stage is rebuilt here over an AccelInput rather than over the
        // inner projection it reads at run time.
        AccelNode outerAlone =
                new AccelProject(
                        List.of(
                                new AccelCall(
                                        AccelFunction.TIMES,
                                        List.of(
                                                new AccelInputRef(0, new DoubleType(false)),
                                                new AccelLiteral(3.0, new DoubleType(false))),
                                        new DoubleType(false))),
                        new AccelInput(one),
                        one);
        GpuKernelSource innerSource =
                AccelKernelGenerator.generate(inner, "a2inner")
                        .orElseThrow(() -> new IllegalStateException("no inner-stage kernel"));
        GpuKernelSource outerSource =
                AccelKernelGenerator.generate(outerAlone, "a2outer")
                        .orElseThrow(() -> new IllegalStateException("no outer-stage kernel"));

        try (StageBuffers buffers = new StageBuffers();
                GeneratedKernelEngine.Compiled first =
                        GeneratedKernelEngine.compileStandalone(innerSource);
                GeneratedKernelEngine.Compiled second =
                        GeneratedKernelEngine.compileStandalone(outerSource)) {

            DoubleArray in =
                    (DoubleArray)
                            buffers.owned(BufferKey.scratch(0, "in"), ElementKind.DOUBLE, ROWS);
            for (int i = 0; i < ROWS; i++) {
                in.set(i, host[i]);
            }
            DoubleArray mid =
                    (DoubleArray) buffers.owned(BufferKey.output(0), ElementKind.DOUBLE, ROWS);
            DoubleArray out =
                    (DoubleArray) buffers.owned(BufferKey.output(1), ElementKind.DOUBLE, ROWS);
            IntArray rows =
                    (IntArray) buffers.owned(BufferKey.scratch(0, "rows"), ElementKind.INT, 1);
            rows.set(0, ROWS);

            Method[] entries = {first.entry(), second.entry()};
            TaskGraph graph =
                    new TaskGraph("a2")
                            .transferToDevice(DataTransferMode.EVERY_EXECUTION, in, rows)
                            .transferToDevice(DataTransferMode.EVERY_EXECUTION, mid, out);
            GridScheduler scheduler = new GridScheduler();
            for (Stage stage : region.stages()) {
                DoubleArray stageIn = stage.index() == 0 ? in : mid;
                DoubleArray stageOut = stage.index() == 0 ? mid : out;
                graph =
                        graph.task(
                                "s" + stage.index(),
                                entries[stage.index()],
                                stageIn,
                                stageOut,
                                rows);
                scheduler.addWorkerGrid("a2.s" + stage.index(), new WorkerGrid1D(ROWS));
            }
            graph = graph.transferToHost(DataTransferMode.EVERY_EXECUTION, out);
            try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
                plan.withGridScheduler(scheduler).execute();
            }

            int mismatches = 0;
            for (int i = 0; i < ROWS; i++) {
                if (Double.compare(Math.sqrt(host[i]) * 3.0, out.get(i)) != 0) {
                    mismatches++;
                }
            }
            if (mismatches != 0) {
                return "FAIL " + mismatches + " of " + ROWS + " rows differ";
            }
            return "PASS "
                    + region.stages().size()
                    + " assembled stages, "
                    + ROWS
                    + "/"
                    + ROWS
                    + " rows exact";
        }
    }

    private static String checkB2() throws Throwable {
        double[] host = new double[ROWS];
        Random random = new Random(44L);
        for (int i = 0; i < ROWS; i++) {
            host[i] = Math.abs(random.nextGaussian()) * 100.0;
        }
        try (ForeignDeviceBuffer foreign = ForeignDeviceBuffer.of(host, libcudart());
                GeneratedKernelEngine.Compiled compiled =
                        GeneratedKernelEngine.compileStandalone(sqrtTimesTwo());
                StageBuffers buffers = new StageBuffers()) {
            DoubleArray out = new DoubleArray(ROWS);
            IntArray rows = new IntArray(1);
            rows.set(0, ROWS);
            BoundView view = buffers.view(BufferKey.output(0), ElementKind.DOUBLE, ROWS);
            TaskGraph graph =
                    new TaskGraph("b2")
                            .transferToDevice(DataTransferMode.EVERY_EXECUTION, out, rows)
                            .task("kernel", compiled.entry(), view.argument(), out, rows)
                            .transferToHost(DataTransferMode.EVERY_EXECUTION, out);
            GridScheduler scheduler = new GridScheduler("b2.kernel", new WorkerGrid1D(ROWS));
            try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
                buffers.bind(
                        plan,
                        BufferKey.output(0),
                        DeviceBinding.describe(
                                "cudaMalloc",
                                foreign.address(),
                                (long) ROWS * Double.BYTES,
                                ROWS,
                                ElementKind.DOUBLE),
                        true,
                        CrossRuntimeOrdering.IN_GRAPH);
                plan.withGridScheduler(scheduler).execute();
                buffers.unbindAll(plan);
            }
            int mismatches = 0;
            for (int i = 0; i < ROWS; i++) {
                if (Double.compare(Math.sqrt(host[i]) * 2.0, out.get(i)) != 0) {
                    mismatches++;
                }
            }
            if (mismatches != 0) {
                return "FAIL " + mismatches + " rows differ";
            }
            return String.format(
                    "PASS read 0x%x (cudaMalloc'd), no transfer in, %d/%d exact",
                    foreign.address(), ROWS, ROWS);
        }
    }

    /**
     * B1 with a foreign-bound view, so the cache and the binding are exercised together.
     *
     * <p>A plan holds its views, so caching one means caching a binding -- which is exactly why the
     * cache is per thread. Testing the cache with buffers it owns outright would leave that
     * interaction unexercised, and it is the interaction most likely to be wrong.
     */
    private static String checkB1() throws Throwable {
        ExecutionPlanCache.releaseThread();
        double[] full = new double[ROWS];
        for (int i = 0; i < ROWS; i++) {
            full[i] = i + 1.0;
        }
        try (ForeignDeviceBuffer foreign = ForeignDeviceBuffer.of(full, libcudart());
                GeneratedKernelEngine.Compiled compiled =
                        GeneratedKernelEngine.compileStandalone(sqrtTimesTwo())) {
            long before = ExecutionPlanCache.misses();
            for (int round = 0; round < 2; round++) {
                for (int rowCount : new int[] {ROWS, SHORT_ROWS}) {
                    String failure = runCached(compiled.entry(), foreign, rowCount, full);
                    if (failure != null) {
                        return failure;
                    }
                }
            }
            long built = ExecutionPlanCache.misses() - before;
            ExecutionPlanCache.releaseThread();
            if (built != 2) {
                return "FAIL four executions built " + built + " plans, wanted 2";
            }
            return "PASS 4 executions, 2 plans, both bound to a foreign buffer";
        }
    }

    private static String runCached(
            Method entry, ForeignDeviceBuffer foreign, int rowCount, double[] host) {
        CachedPlan cached =
                ExecutionPlanCache.get(
                        "sqrt-times-two",
                        PlanShape.ofRows(rowCount),
                        () -> buildBoundPlan(entry, rowCount));
        StageBuffers buffers = cached.buffers();
        DoubleArray out = (DoubleArray) buffers.argument(BufferKey.output(1));
        buffers.bind(
                cached.plan(),
                BufferKey.output(0),
                DeviceBinding.describe(
                        "cudaMalloc",
                        foreign.address(),
                        // The prefix this batch reads, not the whole allocation. A view opened
                        // for a short final batch is smaller than the buffer behind it, and
                        // binding the full length is how you find out that TornadoVM checks.
                        (long) rowCount * Double.BYTES,
                        rowCount,
                        ElementKind.DOUBLE),
                true,
                CrossRuntimeOrdering.IN_GRAPH);
        cached.plan().execute();
        for (int i = 0; i < rowCount; i++) {
            if (Double.compare(Math.sqrt(host[i]) * 2.0, out.get(i)) != 0) {
                return "FAIL at " + rowCount + " rows, row " + i + " differs";
            }
        }
        return null;
    }

    private static CachedPlan buildBoundPlan(Method entry, int rowCount) {
        StageBuffers buffers = new StageBuffers();
        BoundView in = buffers.view(BufferKey.output(0), ElementKind.DOUBLE, rowCount);
        DoubleArray out =
                (DoubleArray) buffers.owned(BufferKey.output(1), ElementKind.DOUBLE, rowCount);
        IntArray rows = (IntArray) buffers.owned(BufferKey.scratch(0, "rows"), ElementKind.INT, 1);
        rows.set(0, rowCount);
        TaskGraph graph =
                new TaskGraph("b1x" + rowCount)
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION, out, rows)
                        .task("kernel", entry, in.argument(), out, rows)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, out);
        GridScheduler scheduler =
                new GridScheduler("b1x" + rowCount + ".kernel", new WorkerGrid1D(rowCount));
        return new CachedPlan(
                new TornadoExecutionPlan(graph.snapshot()).withGridScheduler(scheduler),
                buffers,
                PlanShape.ofRows(rowCount));
    }

    private static String checkB4B5() throws Exception {
        int rows = DeviceUdfVerification.runArrayUdf();
        return "PASS " + rows + "/" + rows + " ragged rows exact through a lowered UDF";
    }

    private static String checkB3() {
        long before = CrossRuntimeOrdering.waits();
        CrossRuntimeOrdering.await(CrossRuntimeOrdering.IN_GRAPH);
        if (CrossRuntimeOrdering.waits() != before + 1) {
            return "FAIL the wait was not counted";
        }
        return "PASS " + CrossRuntimeOrdering.line();
    }

    private static String libcudart() {
        return System.getProperty(
                "jitsql.libcudart",
                "/home/mary/gpu-bench-data/flink-llm/cuda13/targets/x86_64-linux/lib/"
                        + "libcudart.so.13");
    }

    private static GpuKernelSource sqrtTimesTwo() {
        RowType one = RowType.of(new DoubleType(false));
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
                                                new AccelLiteral(2.0, new DoubleType(false))),
                                        new DoubleType(false))),
                        new AccelInput(one),
                        one);
        return AccelKernelGenerator.generate(projection, "sqrt2").orElseThrow();
    }
}
