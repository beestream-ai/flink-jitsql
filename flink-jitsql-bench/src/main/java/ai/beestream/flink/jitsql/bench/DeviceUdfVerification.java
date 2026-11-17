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

import ai.beestream.flink.jitsql.codegen.AccelKernelGenerator;
import ai.beestream.flink.jitsql.codegen.GpuKernelSource;
import ai.beestream.flink.jitsql.device.CrossRuntimeOrdering;
import ai.beestream.flink.jitsql.device.ElementKind;
import ai.beestream.flink.jitsql.gather.ArrayColumnGather;
import ai.beestream.flink.jitsql.operator.GeneratedKernelEngine;

import org.apache.flink.table.accelerator.AccelInput;
import org.apache.flink.table.accelerator.AccelInputRef;
import org.apache.flink.table.accelerator.AccelNode;
import org.apache.flink.table.accelerator.AccelProject;
import org.apache.flink.table.accelerator.AccelUdfCall;
import org.apache.flink.table.data.GenericArrayData;
import org.apache.flink.table.data.GenericRowData;
import org.apache.flink.table.data.RowData;
import org.apache.flink.table.functions.ScalarFunction;
import org.apache.flink.table.types.logical.ArrayType;
import org.apache.flink.table.types.logical.DoubleType;
import org.apache.flink.table.types.logical.RowType;

import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.DoubleArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.cudf.Cudf;

import java.lang.reflect.Method;
import java.util.List;
import java.util.Random;

/**
 * Runs a user's {@code ScalarFunction} over an {@code ARRAY} column on an actual GPU.
 *
 * <p>Everything in the UDF path is checked on the CPU by unit tests -- that the lowered method
 * computes the same doubles as the user's, that the flattened one does too when reading from the
 * middle of a shared buffer, that the generated kernel compiles. None of that can tell you whether
 * the kernel computes the right thing <em>on a device</em>, which is what this does.
 *
 * <p>The comparison is exact. A device is allowed to be faster than a CPU and is not allowed to be
 * different, so a tolerance here would hide the only failure worth catching.
 *
 * <p>Run it with the TornadoVM argfile:
 *
 * <pre>
 *   java @$TORNADO_SDK/tornado-argfile -Dtornado.enable.fma=false \
 *        -Dtornado.cuda.compile.profile=repro -cp ... \
 *        ai.beestream.flink.jitsql.bench.DeviceUdfVerification
 * </pre>
 */
public final class DeviceUdfVerification {

    /** A loop over a ragged array: the shape an expression tree cannot hold. */
    public static final class WeightedSum extends ScalarFunction {
        public double eval(double[] values) {
            double total = 0;
            for (int i = 0; i < values.length; i++) {
                total += values[i] * Math.sqrt(i + 1.0);
            }
            return total;
        }
    }

    private static final int ROWS = 4096;
    private static final int MAX_WIDTH = 24;

    private DeviceUdfVerification() {}

    /**
     * Runs the ARRAY-column UDF chain and returns the row count, throwing on any mismatch.
     *
     * <p>Exposed so {@link DeviceAcceptance} can report B4 and B5 in the same run as everything
     * else, rather than leaving "did they all pass together" open.
     */
    public static int runArrayUdf() throws Exception {
        int mismatches = runAndCount();
        if (mismatches != 0) {
            throw new IllegalStateException(mismatches + " rows differ from the CPU");
        }
        return ROWS;
    }

    public static void main(String[] args) throws Exception {
        runAndCount();
    }

    private static int runAndCount() throws Exception {
        RowType input = RowType.of(new ArrayType(false, new DoubleType(false)));
        AccelNode projection =
                new AccelProject(
                        List.of(
                                new AccelUdfCall(
                                        WeightedSum.class.getName(),
                                        "eval",
                                        "([D)D",
                                        List.of(
                                                new AccelInputRef(
                                                        0,
                                                        new ArrayType(
                                                                false, new DoubleType(false)))),
                                        new DoubleType(false))),
                        new AccelInput(input),
                        RowType.of(new DoubleType(false)));

        GpuKernelSource source =
                AccelKernelGenerator.generate(projection, "deviceudf")
                        .orElseThrow(() -> new IllegalStateException("no kernel was generated"));
        System.out.println("generated kernel:");
        System.out.println(source.source());

        // Stage ragged rows exactly as the operator would.
        ArrayColumnGather gather =
                new ArrayColumnGather(0, ElementKind.DOUBLE, ROWS, ROWS * MAX_WIDTH);
        Random random = new Random(20261117L);
        double[][] rows = new double[ROWS][];
        for (int row = 0; row < ROWS; row++) {
            int width = random.nextInt(MAX_WIDTH);
            rows[row] = new double[width];
            Object[] boxed = new Object[width];
            for (int i = 0; i < width; i++) {
                rows[row][i] = random.nextGaussian() * 100.0;
                boxed[i] = rows[row][i];
            }
            RowData data = GenericRowData.of(new GenericArrayData(boxed));
            gather.accept(data, row);
        }

        int[] offsetsHost = gather.offsets();
        IntArray offsets = new IntArray(offsetsHost.length);
        for (int i = 0; i < offsetsHost.length; i++) {
            offsets.set(i, offsetsHost[i]);
        }
        DoubleArray values = (DoubleArray) gather.values();
        DoubleArray out = new DoubleArray(ROWS);
        IntArray rowCount = new IntArray(1);
        rowCount.set(0, ROWS);

        System.out.printf(
                "staged %d rows holding %d elements between them%n",
                gather.rows(), gather.elements());

        try (GeneratedKernelEngine.Compiled compiled =
                GeneratedKernelEngine.compileStandalone(source)) {
            Method entry = compiled.entry();

            WorkerGrid worker = new WorkerGrid1D(ROWS);
            GridScheduler scheduler = new GridScheduler("udf.kernel", worker);

            TaskGraph graph =
                    new TaskGraph("udf")
                            .transferToDevice(
                                    DataTransferMode.EVERY_EXECUTION, offsets, values, rowCount)
                            .transferToDevice(DataTransferMode.EVERY_EXECUTION, out)
                            .task("kernel", entry, out, offsets, values, rowCount)
                            .transferToHost(DataTransferMode.EVERY_EXECUTION, out);

            ImmutableTaskGraph immutable = graph.snapshot();
            try (TornadoExecutionPlan plan = new TornadoExecutionPlan(immutable)) {
                plan.withGridScheduler(scheduler).execute();
            }
        }

        WeightedSum onTheCpu = new WeightedSum();
        int mismatches = 0;
        int firstMismatch = -1;
        for (int row = 0; row < ROWS; row++) {
            double expected = onTheCpu.eval(rows[row]);
            double actual = out.get(row);
            if (Double.compare(expected, actual) != 0) {
                if (firstMismatch < 0) {
                    firstMismatch = row;
                    System.out.printf(
                            "row %d (width %d): cpu %s, device %s%n",
                            row, rows[row].length, expected, actual);
                }
                mismatches++;
            }
        }

        // Phase two: the same kernel with a cuDF stage after it, in one task graph.
        //
        // This is the shape a region has -- cuDF and generated code interleaved -- and it is why
        // nothing here waits between them. TornadoVM sequences two tasks of one graph, so the
        // ordering barrier is IN_GRAPH and costs nothing, which the line below reports rather
        // than assumes.
        double deviceSum = reduceOnDevice(out, ROWS);
        double cpuSum = 0;
        for (int row = 0; row < ROWS; row++) {
            cpuSum += onTheCpu.eval(rows[row]);
        }
        // A reduction is the one result that must NOT be compared bit for bit.
        //
        // Floating-point addition is not associative, and a parallel tree reduction adds in a
        // different order from a sequential loop. The two answers differ in the last place or two
        // and neither is wrong; demanding equality here would be demanding that the device sum
        // sequentially, which is the one thing a device is for not doing. The elementwise results
        // above are a different matter and are checked exactly, because nothing reorders those.
        double scale = Math.max(Math.abs(cpuSum), Math.abs(deviceSum));
        double ulps = scale == 0 ? 0 : Math.abs(cpuSum - deviceSum) / Math.ulp(scale);
        boolean sumAgrees = ulps <= 64;
        System.out.printf(
                "%nreduce: cpu %s, device %s -> %.1f ulps, %s%n",
                cpuSum,
                deviceSum,
                ulps,
                sumAgrees ? "within summation-order tolerance" : "TOO FAR APART");
        System.out.println(CrossRuntimeOrdering.line());

        System.out.printf("%nCHECKED: %d rows, MISMATCHES: %d%n", ROWS, mismatches);
        boolean passed = mismatches == 0 && sumAgrees;
        System.out.println(passed ? "DEVICE VERIFICATION: PASS" : "DEVICE VERIFICATION: FAIL");
        return passed ? mismatches : Math.max(mismatches, 1);
    }

    /**
     * Sums the kernel's output with {@code Cudf.reduce}, in a graph of its own.
     *
     * <p>Separate from the kernel's graph only because the kernel's output has already come back to
     * the host for checking; in a region the two are tasks of one graph, which is the case the
     * ordering barrier is {@link CrossRuntimeOrdering#IN_GRAPH} for.
     */
    private static double reduceOnDevice(DoubleArray values, int rows) throws Exception {
        DoubleArray sum = new DoubleArray(1);
        TaskGraph graph =
                new TaskGraph("reduce")
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION, values)
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION, sum)
                        .libraryTask("sum", Cudf::reduce, rows, 0, values, sum)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, sum);
        CrossRuntimeOrdering.await(CrossRuntimeOrdering.IN_GRAPH);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            plan.execute();
        }
        return sum.get(0);
    }
}
