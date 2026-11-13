/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
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

import org.apache.flink.table.api.EnvironmentSettings;
import org.apache.flink.table.api.TableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.StringJoiner;

/**
 * The Gram-matrix query over Parquet, where the rows need never reach the host.
 *
 * <p>{@link FeatureGramBenchmark} runs the same query over CSV, which means Flink decodes every row
 * and the accelerated operator stages the columns it wants out of arriving {@code RowData}. There
 * is a second device path for exactly this shape — {@code DeviceParquetGramSource} — where cuDF
 * reads the file straight into device buffers, the generated kernel computes the features where
 * they lie, and cuBLAS contracts them, with the host never seeing a row.
 *
 * <p>The two paths compute the same matrix with the same library from the same SQL, and differ in
 * precisely the host-side row handling. That is what this example exists to measure.
 *
 * <h2>What the source path requires, and refuses</h2>
 *
 * <ul>
 *   <li>{@code 'format' = 'parquet'} — a CSV scan is declined, because the read is cuDF's.
 *   <li>Columns whose width <em>is</em> the contraction's width: a {@code FLOAT} query over {@code
 *       float} columns, or {@code DOUBLE} over {@code double}. A file of the other width is refused
 *       rather than cast, so it is a planning decision instead of a failure inside a running job.
 *   <li>{@code NOT NULL}, for the reason every device region here needs it.
 * </ul>
 *
 * <h2>Choosing the arm</h2>
 *
 * <p>Both arms are the same characters of SQL. What selects between them is a property the
 * TaskManager JVM is started with:
 *
 * <pre>
 *   (none)                                    the source path: cuDF reads the file
 *   -Dflink.accelerator.device-scan=off       the operator path: Flink reads, the operator stages
 *   -Dflink.accelerator.gram.contraction=kernel   a generated contraction instead of the GEMM
 * </pre>
 *
 * <pre>{@code
 * GramParquetSQLExample --data /home/mary/gpu-bench-data/gram-4m-f32 --cols 32 --type float
 * }</pre>
 */
public final class GramParquetSQLExample {

    public static void main(String[] args) throws Exception {
        final Args parsed = Args.parse(args);

        if (!parsed.explainOnly && !Files.isDirectory(Paths.get(parsed.data))) {
            throw new IOException(
                    parsed.data
                            + " does not exist. This example reads Parquet written elsewhere; it"
                            + " does not generate it, because the column width has to match the"
                            + " query's declared width exactly.");
        }

        System.out.printf(
                "data=%s  cols=%d  type=%s  offload=%s  fuse=%s  runs=%d  device-scan=%s"
                        + "  contraction=%s%n",
                parsed.data,
                parsed.cols,
                parsed.type,
                parsed.offload,
                parsed.fuseAggregate,
                parsed.runs,
                System.getProperty("flink.accelerator.device-scan", "on"),
                System.getProperty("flink.accelerator.gram.contraction", "cublas"));

        if (parsed.explainOnly) {
            gramExplain(parsed);
            return;
        }

        for (int run = 1; run <= parsed.runs; run++) {
            final long start = System.nanoTime();
            final double[] upper = gram(parsed, run);
            System.out.printf(
                    "run %2d  %10.0f ms  trace=%.6e  entries=%d%n",
                    run,
                    (System.nanoTime() - start) / 1e6,
                    trace(upper, parsed.cols),
                    upper.length);
        }
    }

    /** Prints the plan without submitting anything. */
    private static void gramExplain(Args args) {
        final TableEnvironment env =
                TableEnvironment.create(EnvironmentSettings.newInstance().inBatchMode().build());
        env.getConfig().getConfiguration().setString("table.exec.accelerator.enabled", "true");
        env.getConfig()
                .getConfiguration()
                .setString("table.exec.accelerator.approximate-projections", "true");
        env.getConfig()
                .getConfiguration()
                .setString("table.exec.accelerator.fuse-aggregate", "true");
        final String sqlType = "float".equals(args.type) ? "FLOAT" : "DOUBLE";
        final StringJoiner columns = new StringJoiner(",\n  ");
        for (int c = 0; c < args.cols; c++) {
            columns.add("c" + c + " " + sqlType + " NOT NULL");
        }
        env.executeSql(
                "CREATE TABLE Points (\n  "
                        + columns
                        + "\n) WITH (\n"
                        + "  'connector' = 'filesystem',\n"
                        + "  'path' = '"
                        + args.data
                        + "',\n"
                        + "  'format' = 'parquet'\n)");
        final StringJoiner sums = new StringJoiner(", ");
        for (int i = 0; i < args.cols; i++) {
            for (int j = i; j < args.cols; j++) {
                sums.add(String.format("SUM(f%d * f%d)", i, j));
            }
        }
        final StringJoiner features = new StringJoiner(", ");
        for (int c = 0; c < args.cols; c++) {
            features.add(String.format("SIN(c%d) AS f%d", c, c));
        }
        System.out.println(
                env.explainSql("SELECT " + sums + "\nFROM (SELECT " + features + " FROM Points)"));
    }

    private static double[] gram(Args args, int run) throws Exception {
        final TableEnvironment env =
                TableEnvironment.create(EnvironmentSettings.newInstance().inBatchMode().build());

        if (args.offload) {
            env.getConfig().getConfiguration().setString("table.exec.accelerator.enabled", "true");
            env.getConfig()
                    .getConfiguration()
                    .setString("table.exec.accelerator.approximate-projections", "true");
            // Decisive for WHICH device path runs, and not obvious.
            //
            // createScanSource is attempted only from BatchExecHashAggregate. With fusion on,
            // GpuOffloadProcessor rewrites that node into BatchExecAcceleratedGroup before
            // translation, and the fused node never attempts the substitution -- so
            // fuse-aggregate=true silently makes the source path unreachable. The operator
            // path needs fusion on; the source path needs it off.
            env.getConfig()
                    .getConfiguration()
                    .setString(
                            "table.exec.accelerator.fuse-aggregate",
                            String.valueOf(args.fuseAggregate));
        }
        env.getConfig()
                .getConfiguration()
                .setString(
                        "table.exec.resource.default-parallelism",
                        String.valueOf(args.parallelism));
        // The accelerator's own cap, which defaults to 1 and declines every subtask above it.
        // It has to be set HERE: a -D on bin/flink run reaches the cluster configuration and
        // not the TableEnvironment this example builds, so passing it there silently leaves the
        // default in place and every parallel arm runs on the CPU while still returning the
        // right answer.
        env.getConfig()
                .getConfiguration()
                .setString(
                        "table.exec.accelerator.max-parallelism",
                        String.valueOf(
                                args.maxParallelism > 0 ? args.maxParallelism : args.parallelism));
        env.getConfig().getConfiguration().setString("taskmanager.memory.managed.size", "2g");

        final String sqlType = "float".equals(args.type) ? "FLOAT" : "DOUBLE";
        final StringJoiner columns = new StringJoiner(",\n  ");
        for (int c = 0; c < args.cols; c++) {
            columns.add("c" + c + " " + sqlType + " NOT NULL");
        }
        env.executeSql(
                "CREATE TABLE Points (\n  "
                        + columns
                        + "\n) WITH (\n"
                        + "  'connector' = 'filesystem',\n"
                        + "  'path' = '"
                        + args.data
                        + "',\n"
                        + "  'format' = 'parquet'\n)");

        // Upper triangle only: A'A is symmetric, and asking for all d^2 entries would make the
        // CPU arm do twice the work for no extra information.
        final StringJoiner sums = new StringJoiner(", ");
        for (int i = 0; i < args.cols; i++) {
            for (int j = i; j < args.cols; j++) {
                sums.add(String.format("SUM(f%d * f%d)", i, j));
            }
        }
        final StringJoiner features = new StringJoiner(", ");
        for (int c = 0; c < args.cols; c++) {
            features.add(String.format("SIN(c%d) AS f%d", c, c));
        }
        final String sql = "SELECT " + sums + "\nFROM (SELECT " + features + " FROM Points)";

        if (System.getenv("EXPLAIN") != null) {
            final String plan = env.explainSql(sql);
            final int at = plan.indexOf("== GPU Offload ==");
            System.out.println(at >= 0 ? plan.substring(at) : plan);
        }

        final double[] upper;
        try (CloseableIterator<Row> it = env.executeSql(sql).collect()) {
            final Row row = it.next();
            upper = new double[row.getArity()];
            for (int i = 0; i < upper.length; i++) {
                upper[i] = ((Number) row.getField(i)).doubleValue();
            }
        }
        dump(upper, run);
        return upper;
    }

    private static double trace(double[] upper, int cols) {
        double sum = 0.0;
        int at = 0;
        for (int i = 0; i < cols; i++) {
            sum += upper[at];
            at += cols - i;
        }
        return sum;
    }

    /** The whole upper triangle, for a comparison against an independently computed reference. */
    private static void dump(double[] upper, int run) throws IOException {
        final String where = System.getenv("GRAM_DUMP");
        if (where == null || where.isEmpty()) {
            return;
        }
        final StringBuilder out = new StringBuilder(upper.length * 24);
        for (double v : upper) {
            out.append(Double.toString(v)).append('\n');
        }
        Files.write(Paths.get(where + ".run" + run), out.toString().getBytes("UTF-8"));
    }

    private static final class Args {
        private String data;
        private int cols = 32;
        private String type = "float";
        private int runs = 6;
        private int parallelism = 1;
        private boolean offload = true;
        private boolean explainOnly;
        private boolean fuseAggregate = true;
        private int maxParallelism = 0;

        static Args parse(String[] argv) {
            final Args a = new Args();
            for (int i = 0; i < argv.length; i++) {
                final String flag = argv[i];
                if ("--data".equals(flag)) {
                    a.data = argv[++i];
                } else if ("--cols".equals(flag)) {
                    a.cols = Integer.parseInt(argv[++i]);
                } else if ("--type".equals(flag)) {
                    a.type = argv[++i];
                } else if ("--runs".equals(flag)) {
                    a.runs = Integer.parseInt(argv[++i]);
                } else if ("--parallelism".equals(flag)) {
                    a.parallelism = Integer.parseInt(argv[++i]);
                } else if ("--max-parallelism".equals(flag)) {
                    a.maxParallelism = Integer.parseInt(argv[++i]);
                } else if ("--fuse-aggregate".equals(flag)) {
                    a.fuseAggregate = Boolean.parseBoolean(argv[++i]);
                } else if ("--explain-only".equals(flag)) {
                    a.explainOnly = true;
                } else if ("--offload".equals(flag)) {
                    a.offload = Boolean.parseBoolean(argv[++i]);
                } else {
                    throw new IllegalArgumentException("unknown argument " + flag);
                }
            }
            if (a.data == null) {
                throw new IllegalArgumentException("--data is required");
            }
            return a;
        }
    }

    private GramParquetSQLExample() {}
}
