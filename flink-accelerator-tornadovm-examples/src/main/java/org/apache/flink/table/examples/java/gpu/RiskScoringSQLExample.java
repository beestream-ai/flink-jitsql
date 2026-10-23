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

package org.apache.flink.table.examples.java.gpu;

import org.apache.flink.table.api.EnvironmentSettings;
import org.apache.flink.table.api.TableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.StringJoiner;

/**
 * Risk scoring: a custom expression computed per row, then ranked by a native relational operator.
 *
 * <p>The third application family. {@link FeatureGramBenchmark} and {@link
 * NearestNeighbourSQLExample} are both dense linear algebra, where cuBLAS is already a very good
 * answer and a generated kernel is competing against decades of tuning. This one is not: the
 * scoring expression is application-specific, no library has a form for it, and what a library
 * contributes is the <em>ranking</em> rather than the arithmetic.
 *
 * <h2>The computation, and why it is a real one</h2>
 *
 * <pre>{@code
 * risk = 0.4*LN(1 + amount) + 0.3*velocity^2 + 0.3*ABS(deviation)
 * }</pre>
 *
 * <p>A weighted combination of three terms with defined meanings: a log-compressed transaction
 * magnitude, so that an order of magnitude more money is a fixed increment of risk rather than a
 * proportional one; a squared rate term, so that bursts count superlinearly; and an absolute
 * deviation from a per-account baseline, which is symmetric because an unusually small transaction
 * is as suspicious as an unusually large one. The weights sum to one, so the score is on a scale
 * the three terms share.
 *
 * <p>This is the shape of nightly batch risk ranking, anomaly ordering and lead scoring. It is
 * deliberately <em>not</em> an arbitrary chain of transcendentals assembled to make the device look
 * busy — the point of the experiment is to measure a plausible expression, not a maximal one. The
 * {@code --terms} switch adds or removes terms of the same kind so that expression complexity can
 * be swept without changing what the query means.
 *
 * <h2>What makes it a hybrid region</h2>
 *
 * <p>{@code Calc} feeding {@code Sort} is rewritten by {@code GpuOffloadProcessor} into one node,
 * and the provider serves it with {@code GpuProjectedSortOperator}: the generated kernel writes its
 * columns into device buffers and {@code cudf::sorted_order} orders the score column where it lies,
 * in the same task graph. Unfused this is two operators, two task graphs, and two rounds of
 * building a {@code BinaryRowData} for every row — and §T12 attributed 616 ms of a 954 ms offloaded
 * sort to exactly that row building, against 15.7 ms for the ordering itself.
 *
 * <p>So this query is where the integration's host-side cost is most exposed, and it is included
 * for that reason rather than despite it.
 *
 * <h2>Ordinary SQL</h2>
 *
 * <p>No hint, no annotation, no accelerator type. The CPU and accelerated arms run the same
 * characters. The differences are session settings a cluster operator sets: {@code enabled}, and
 * {@code approximate-projections} because {@code LN} is a function a device may round differently
 * from {@code java.lang.Math}.
 *
 * <p>{@code NOT NULL} in the DDL is the one thing the query author writes, which is the project's
 * one sanctioned exception.
 *
 * <pre>{@code
 * RiskScoringSQLExample --generate --rows 4000000 --data DIR
 * RiskScoringSQLExample --rows 4000000 --data DIR --offload true --runs 6
 * RiskScoringSQLExample --rows 4000000 --data DIR --offload true --mode group --groups 1000
 * }</pre>
 */
public final class RiskScoringSQLExample {

    /** Ordered rows retained for the correctness comparison; the whole result is far too long. */
    private static final int HEAD = 1000;

    public static void main(String[] args) throws Exception {
        final Args parsed = Args.parse(args);

        if (parsed.generate) {
            write(Paths.get(parsed.data), parsed.rows, parsed.payload, parsed.groups, parsed.files);
            System.out.printf(
                    "wrote %,d rows to %s in %d file(s)%n", parsed.rows, parsed.data, parsed.files);
            return;
        }
        if (!parsed.explainOnly) {
            write(Paths.get(parsed.data), parsed.rows, parsed.payload, parsed.groups, parsed.files);
            if ("parquet".equals(parsed.format)) {
                toParquet(parsed);
                parsed.data = parsed.data + "-pq";
            }
        }

        System.out.printf(
                "data=%s  rows=%,d  terms=%d  payload=%d  mode=%s  groups=%d  offload=%s  runs=%d%n",
                parsed.data,
                parsed.rows,
                parsed.terms,
                parsed.payload,
                parsed.mode,
                parsed.groups,
                parsed.offload,
                parsed.runs);

        if (parsed.explainOnly) {
            // Planning only: no job is submitted, nothing touches the device, and the plan the
            // planner chose is printed. Safe to run when the machine is not in a state where a
            // timing would be valid, which is the point of having it.
            explain(parsed);
            return;
        }

        for (int run = 1; run <= parsed.runs; run++) {
            final long start = System.nanoTime();
            final long seen;
            if ("group-computed".equals(parsed.mode)) {
                seen = groupComputed(parsed, run);
            } else if ("group".equals(parsed.mode)) {
                seen = group(parsed, run);
            } else if ("sum".equals(parsed.mode)) {
                seen = sum(parsed, run);
            } else {
                seen = rank(parsed, run);
            }
            System.out.printf(
                    "run %2d  %10.0f ms  rows=%d%n", run, (System.nanoTime() - start) / 1e6, seen);
        }
    }

    /** Prints the plan without submitting anything. */
    private static void explain(Args args) {
        final TableEnvironment env = environment(args);
        declare(env, args);
        final StringJoiner passthrough = new StringJoiner(", ");
        for (int p = 0; p < args.payload; p++) {
            passthrough.add("p" + p);
        }
        final String sql =
                "group".equals(args.mode)
                        ? "SELECT bucket, SUM("
                                + score(args.terms)
                                + ") AS total FROM Transactions GROUP BY bucket"
                        : "SELECT id, "
                                + ("rank-int".equals(args.mode)
                                        ? intScore(args.terms)
                                        : score(args.terms))
                                + " AS risk"
                                + (args.payload > 0 ? ", " + passthrough : "")
                                + " FROM Transactions ORDER BY risk";
        System.out.println(sql);
        System.out.println(env.explainSql(sql));
    }

    /**
     * An integer scoring expression: the same three signals, quantised to score points.
     *
     * <p>Not a contrivance to get past a planner rule. Quantised scores are what real scoring
     * systems publish -- a credit score is an integer, and so is a risk band -- and the reason is
     * the same one that makes this shape offloadable: an integer score has one unambiguous
     * ordering, so two implementations that agree on the score agree on the ranking.
     *
     * <p>{@link #score} cannot be ordered on a device for two independent reasons, both planner
     * rules rather than device limits: a value a device may round differently is refused when it
     * reaches a consumer that decides what order rows come back in, and the cuDF sort binding takes
     * an {@code INT32} key column and nothing else.
     */
    private static String intScore(int terms) {
        // Over the integer signal columns, so the expression contains no cast. The generator
        // writes them because a risk table has them: a count of recent declines, a tenure band,
        // a merchant-category code. Casting the DOUBLE columns instead is refused by the kernel
        // generator ("unsupported cast to INTEGER"), which is a coverage limit worth recording
        // and not worth working around with a cast the data does not need.
        final StringJoiner s = new StringJoiner(" + ");
        if (terms >= 1) {
            s.add("declines * 40");
        }
        if (terms >= 2) {
            s.add("tenure * tenure * 2");
        }
        if (terms >= 3) {
            s.add("ABS(category - 50) * 3");
        }
        if (terms >= 4) {
            s.add("declines * tenure");
        }
        if (terms >= 5) {
            s.add("category * 7");
        }
        return s.toString();
    }

    /**
     * The DDL, and optionally the statistics the sort needs.
     *
     * <p>{@code GpuSortSupport} refuses a sort whose cardinality is unknown — "a sort holds its
     * whole input", so without an estimate the planner cannot size the staging and will not guess.
     * A filesystem table has no statistics until something computes them, which is why the fused
     * projection-and-sort was declined on a missing estimate rather than on cost. {@code ANALYZE
     * TABLE} supplies one.
     */
    private static void declare(TableEnvironment env, Args args) {
        env.executeSql(ddl(args));
        if (args.analyze) {
            try {
                env.executeSql("ANALYZE TABLE Transactions COMPUTE STATISTICS").await();
            } catch (Exception e) {
                throw new IllegalStateException("ANALYZE TABLE failed", e);
            }
        }
    }

    /** The scoring expression, as many terms as asked for, all of the same three kinds. */
    private static String score(int terms) {
        final StringJoiner s = new StringJoiner(" + ");
        if (terms >= 1) {
            s.add("CAST(0.4 AS DOUBLE) * LN(CAST(1.0 AS DOUBLE) + amount)");
        }
        if (terms >= 2) {
            s.add("CAST(0.3 AS DOUBLE) * velocity * velocity");
        }
        if (terms >= 3) {
            s.add("CAST(0.3 AS DOUBLE) * ABS(deviation)");
        }
        if (terms >= 4) {
            s.add("CAST(0.2 AS DOUBLE) * SQRT(amount)");
        }
        if (terms >= 5) {
            s.add("CAST(0.1 AS DOUBLE) * EXP(-ABS(deviation))");
        }
        return s.toString();
    }

    /** The ranking query: a custom score per row, ordered by it. */
    private static long rank(Args args, int run) throws Exception {
        final TableEnvironment env = environment(args);
        declare(env, args);

        final StringJoiner passthrough = new StringJoiner(", ");
        for (int p = 0; p < args.payload; p++) {
            passthrough.add("p" + p);
        }
        final String sql =
                "SELECT id, "
                        + ("rank-int".equals(args.mode) ? intScore(args.terms) : score(args.terms))
                        + " AS risk"
                        + (args.payload > 0 ? ", " + passthrough : "")
                        + " FROM Transactions ORDER BY risk";

        if (System.getenv("EXPLAIN") != null) {
            final String plan = env.explainSql(sql);
            final int at = plan.indexOf("== GPU Offload ==");
            System.out.println(at >= 0 ? plan.substring(at) : plan);
        }

        long seen = 0;
        double previous = Double.NEGATIVE_INFINITY;
        final StringBuilder head = new StringBuilder();
        try (CloseableIterator<Row> ordered = env.executeSql(sql).collect()) {
            while (ordered.hasNext()) {
                final Row row = ordered.next();
                final double risk = ((Number) row.getField(1)).doubleValue();
                // A sort reassociates nothing: anything out of order is a defect, not rounding.
                if (risk < previous) {
                    throw new IllegalStateException(
                            "row " + seen + " scores " + risk + " after " + previous);
                }
                if (seen < HEAD) {
                    head.append(row.getField(0)).append(',').append(risk).append('\n');
                }
                previous = risk;
                seen++;
            }
        }
        dump(head.toString(), run);
        return seen;
    }

    /**
     * The ungrouped companion: the same score, summed over everything.
     *
     * <p>The one aggregate shape the provider will serve as a SOURCE ({@code scanProjectSumSource}:
     * Parquet, no grouping, exactly one SUM). That matters for parallelism: a source substitution
     * does not go through {@code GpuOrCpuCalcOperatorFactory}, which is where the hardcoded {@code
     * maxParallelism = 1} lives, so this shape runs at any parallelism while the fused grouped
     * aggregate cannot.
     */
    private static long sum(Args args, int run) throws Exception {
        final TableEnvironment env = environment(args);
        declare(env, args);
        final String sql =
                "SELECT SUM("
                        + ("rank-int".equals(args.mode) ? intScore(args.terms) : score(args.terms))
                        + ") AS total FROM Transactions";
        if (System.getenv("EXPLAIN") != null) {
            final String plan = env.explainSql(sql);
            final int at = plan.indexOf("== GPU Offload ==");
            System.out.println(at >= 0 ? plan.substring(at) : plan);
        }
        final StringBuilder all = new StringBuilder();
        long seen = 0;
        try (CloseableIterator<Row> it = env.executeSql(sql).collect()) {
            while (it.hasNext()) {
                all.append("total,").append(it.next().getField(0)).append('\n');
                seen++;
            }
        }
        dump(all.toString(), run);
        return seen;
    }

    /**
     * The grouped aggregate over a COMPUTED key.
     *
     * <p>Exists to get past two things the plain grouped form runs into, neither of which is about
     * the arithmetic:
     *
     * <ul>
     *   <li>the kernel generator will not stage a field copied straight from input ("field 0 is
     *       copied from input field 0"), and {@code GROUP BY bucket} passes {@code bucket} through
     *       unchanged. A computed key is computed, so nothing is copied.
     *   <li>the region was declined at 2,064 rows a subtask. A key with many more distinct values
     *       raises the cardinality the sort is costed against.
     * </ul>
     *
     * <p>The key is still meaningful: a risk band derived from the integer signals, which is what a
     * scoring pipeline groups by in practice.
     */
    private static long groupComputed(Args args, int run) throws Exception {
        final TableEnvironment env = environment(args);
        declare(env, args);
        final String key = "MOD(declines * 7919 + tenure * 131 + category, " + args.groups + ")";
        final String sql =
                "SELECT "
                        + key
                        + " AS band, SUM("
                        + score(args.terms)
                        + ") AS total FROM Transactions GROUP BY "
                        + key;
        if (System.getenv("EXPLAIN") != null) {
            final String plan = env.explainSql(sql);
            final int at = plan.indexOf("== GPU Offload ==");
            System.out.println(at >= 0 ? plan.substring(at) : plan);
        }
        long seen = 0;
        final StringBuilder all = new StringBuilder();
        try (CloseableIterator<Row> it = env.executeSql(sql).collect()) {
            while (it.hasNext()) {
                final Row row = it.next();
                all.append(row.getField(0)).append(',').append(row.getField(1)).append('\n');
                seen++;
            }
        }
        dump(all.toString(), run);
        return seen;
    }

    /** The grouped-aggregate companion: the same score, summed per bucket. */
    private static long group(Args args, int run) throws Exception {
        final TableEnvironment env = environment(args);
        declare(env, args);

        final String sql =
                "SELECT bucket, SUM("
                        + score(args.terms)
                        + ") AS total FROM Transactions GROUP BY bucket";

        if (System.getenv("EXPLAIN") != null) {
            final String plan = env.explainSql(sql);
            final int at = plan.indexOf("== GPU Offload ==");
            System.out.println(at >= 0 ? plan.substring(at) : plan);
        }

        long seen = 0;
        final StringBuilder all = new StringBuilder();
        try (CloseableIterator<Row> it = env.executeSql(sql).collect()) {
            while (it.hasNext()) {
                final Row row = it.next();
                all.append(row.getField(0)).append(',').append(row.getField(1)).append('\n');
                seen++;
            }
        }
        dump(all.toString(), run);
        return seen;
    }

    private static TableEnvironment environment(Args args) {
        final TableEnvironment env =
                TableEnvironment.create(EnvironmentSettings.newInstance().inBatchMode().build());
        // The switch, and the whole of it. A cluster operator sets these; a query author never
        // sees them, and not one character of the SQL above depends on them.
        if (args.offload) {
            env.getConfig().getConfiguration().setString("table.exec.accelerator.enabled", "true");
            // LN and EXP are functions a device may round differently from java.lang.Math.
            env.getConfig()
                    .getConfiguration()
                    .setString("table.exec.accelerator.approximate-projections", "true");
            if ("group".equals(args.mode)) {
                env.getConfig()
                        .getConfiguration()
                        .setString("table.exec.accelerator.fuse-aggregate", "true");
            }
        }
        if (args.sortAgg) {
            // A sort-based aggregate instead of a hash one. Its LOCAL phase sorts without
            // redistributing, so the Sort sits directly on the Calc with no exchange between
            // them -- which is the shape GpuOffloadProcessor needs to fuse a projection into a
            // sort, and the shape a global ORDER BY never produces. An ordinary Flink option,
            // not a planner change.
            env.getConfig()
                    .getConfiguration()
                    .setString("table.exec.disabled-operators", "HashAgg");
        }
        env.getConfig()
                .getConfiguration()
                .setString(
                        "table.exec.resource.default-parallelism",
                        String.valueOf(args.parallelism));
        // The accelerator's own cap. It has to be set HERE, not with -D on bin/flink run, which
        // reaches the cluster configuration and not this TableEnvironment.
        env.getConfig()
                .getConfiguration()
                .setString(
                        "table.exec.accelerator.max-parallelism", String.valueOf(args.parallelism));
        env.getConfig().getConfiguration().setString("taskmanager.memory.managed.size", "2g");
        return env;
    }

    /** Converts the generated CSV into Parquet once, because the source path reads only Parquet. */
    private static void toParquet(Args args) throws Exception {
        final java.nio.file.Path pq = Paths.get(args.data + "-pq");
        if (Files.isDirectory(pq)) {
            return;
        }
        final TableEnvironment env =
                TableEnvironment.create(EnvironmentSettings.newInstance().inBatchMode().build());
        final Args csv = args.copyWithFormat("csv");
        env.executeSql(ddl(csv).replace("Transactions", "TxCsv"));
        final Args parquet = args.copyWithFormat("parquet");
        parquet.data = pq.toString();
        env.executeSql(ddl(parquet).replace("Transactions", "TxPq"));
        env.executeSql("INSERT INTO TxPq SELECT * FROM TxCsv").await();
    }

    private static String ddl(Args args) {
        final StringJoiner columns = new StringJoiner(",\n  ");
        columns.add("id BIGINT NOT NULL");
        columns.add("amount DOUBLE NOT NULL");
        columns.add("velocity DOUBLE NOT NULL");
        columns.add("deviation DOUBLE NOT NULL");
        columns.add("bucket INT NOT NULL");
        // Integer signals, which a risk table has: recent declines, tenure band, category code.
        columns.add("declines INT NOT NULL");
        columns.add("tenure INT NOT NULL");
        columns.add("category INT NOT NULL");
        for (int p = 0; p < args.payload; p++) {
            columns.add("p" + p + " DOUBLE NOT NULL");
        }
        return "CREATE TABLE Transactions (\n  "
                + columns
                + "\n) WITH (\n"
                + "  'connector' = 'filesystem',\n"
                + "  'path' = '"
                + args.data
                + "',\n"
                + "  'format' = '"
                + args.format
                + "'\n)";
    }

    /** Writes the ordered head, or the whole grouped result, where {@code RISK_DUMP} points. */
    private static void dump(String body, int run) throws IOException {
        final String where = System.getenv("RISK_DUMP");
        if (where == null || where.isEmpty()) {
            return;
        }
        Files.write(Paths.get(where + ".run" + run), body.getBytes("UTF-8"));
    }

    /**
     * Writes the input, once.
     *
     * <p>Plain Java rather than the {@code datagen} connector, for the reason {@link
     * CudfSortSQLExample} gives: {@code datagen} costs about 100 microseconds a row.
     *
     * <p>{@code amount} is strictly positive so {@code LN(1 + amount)} is defined everywhere, and
     * the sequence is a multiplicative hash rather than anything sorted, because ranking
     * already-ranked input demonstrates nothing.
     */
    private static void write(Path directory, long rows, int payload, int groups, int files)
            throws IOException {
        if (Files.isDirectory(directory)) {
            return;
        }
        Files.createDirectories(directory);
        // Several files, not one. A single-file input pins Flink's scan to one subtask whatever
        // the job's parallelism says, so a parallelism sweep over one file measures parallelism 1
        // three times -- which is exactly what the first run of X14 did.
        final java.util.List<BufferedWriter> writers = new java.util.ArrayList<>();
        for (int f = 0; f < Math.max(1, files); f++) {
            writers.add(Files.newBufferedWriter(directory.resolve("part-" + f + ".csv")));
        }
        try {
            final StringBuilder line = new StringBuilder(64 + payload * 20);
            for (long id = 0; id < rows; id++) {
                final long h = (id * 2654435761L) & 0x7fffffffL;
                final double amount = 1.0 + (h % 1_000_000L) / 100.0;
                final double velocity = ((h >> 7) % 10_000L) / 1000.0;
                final double deviation = (((h >> 13) % 20_000L) - 10_000L) / 1000.0;
                final int bucket = (int) (h % groups);
                final int declines = (int) ((h >> 3) % 12);
                final int tenure = (int) ((h >> 9) % 40);
                final int category = (int) ((h >> 17) % 100);
                line.setLength(0);
                line.append(id)
                        .append(',')
                        .append(amount)
                        .append(',')
                        .append(velocity)
                        .append(',')
                        .append(deviation)
                        .append(',')
                        .append(bucket)
                        .append(',')
                        .append(declines)
                        .append(',')
                        .append(tenure)
                        .append(',')
                        .append(category);
                for (int p = 0; p < payload; p++) {
                    line.append(',').append(((h >> (3 + p)) % 1000L) / 10.0);
                }
                final BufferedWriter out = writers.get((int) (id % writers.size()));
                out.write(line.toString());
                out.newLine();
            }
        } finally {
            for (BufferedWriter w : writers) {
                w.close();
            }
        }
    }

    private static final class Args {
        private String data;
        private long rows = 4_000_000L;
        private int terms = 3;
        private int payload = 0;
        private int groups = 1000;
        private int runs = 6;
        private int parallelism = 1;
        private String mode = "rank";
        private boolean offload;
        private boolean generate;
        private boolean explainOnly;
        private boolean sortAgg;
        private boolean analyze;
        private String format = "csv";
        private int files = 1;

        Args copyWithFormat(String f) {
            final Args a = new Args();
            a.data = data;
            a.rows = rows;
            a.terms = terms;
            a.payload = payload;
            a.groups = groups;
            a.runs = runs;
            a.parallelism = parallelism;
            a.mode = mode;
            a.offload = offload;
            a.format = f;
            a.files = files;
            a.sortAgg = sortAgg;
            a.analyze = analyze;
            return a;
        }

        static Args parse(String[] argv) {
            final Args a = new Args();
            for (int i = 0; i < argv.length; i++) {
                final String flag = argv[i];
                if ("--data".equals(flag)) {
                    a.data = argv[++i];
                } else if ("--rows".equals(flag)) {
                    a.rows = Long.parseLong(argv[++i]);
                } else if ("--terms".equals(flag)) {
                    a.terms = Integer.parseInt(argv[++i]);
                } else if ("--payload".equals(flag)) {
                    a.payload = Integer.parseInt(argv[++i]);
                } else if ("--groups".equals(flag)) {
                    a.groups = Integer.parseInt(argv[++i]);
                } else if ("--runs".equals(flag)) {
                    a.runs = Integer.parseInt(argv[++i]);
                } else if ("--parallelism".equals(flag)) {
                    a.parallelism = Integer.parseInt(argv[++i]);
                } else if ("--mode".equals(flag)) {
                    a.mode = argv[++i];
                } else if ("--offload".equals(flag)) {
                    a.offload = Boolean.parseBoolean(argv[++i]);
                } else if ("--files".equals(flag)) {
                    a.files = Integer.parseInt(argv[++i]);
                } else if ("--format".equals(flag)) {
                    a.format = argv[++i];
                } else if ("--analyze".equals(flag)) {
                    a.analyze = true;
                } else if ("--sort-agg".equals(flag)) {
                    a.sortAgg = true;
                } else if ("--explain-only".equals(flag)) {
                    a.explainOnly = true;
                } else if ("--generate".equals(flag)) {
                    a.generate = true;
                } else {
                    throw new IllegalArgumentException("unknown argument " + flag);
                }
            }
            if (a.data == null) {
                // The bucket column is written at generation time, so the group count is a
                // property of the data and not only of the query: it belongs in the path, or a
                // later run would silently read a file bucketed for a different group count.
                a.data =
                        "/home/mary/gpu-bench-data/risk-"
                                + a.rows
                                + "-"
                                + a.payload
                                + "-g"
                                + a.groups;
            }
            return a;
        }
    }

    private RiskScoringSQLExample() {}
}
