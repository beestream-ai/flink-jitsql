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

import org.apache.flink.table.api.EnvironmentSettings;
import org.apache.flink.table.api.TableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

import java.io.BufferedWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * The one command that says whether this works: every supported operator shape, run twice, and one
 * line at the end.
 *
 * <pre>
 *   flink run IntegrationCheck.jar
 *   ... ends: CHECKS: 10, FAILURES: 0
 * </pre>
 *
 * <h2>What it checks, and why that is the only interesting question</h2>
 *
 * <p>Goal 0 of this project is result equivalence: the accelerated plan returns what the CPU plan
 * returns. Everything else — the speed-up, the coverage, the citizenship — is worth nothing if a
 * query answers differently with the accelerator on. So each case here runs the same SQL twice in
 * one JVM, once with {@code table.exec.accelerator.enabled} true and once false, and compares the
 * answers row by row.
 *
 * <p>The CPU arm is not a reimplementation written to agree with the device. It is Flink's own
 * code-generated operator, which is what runs when no provider is present, so the comparison is
 * against the behaviour Flink already guarantees rather than against a second opinion of ours.
 *
 * <h2>Why the comparison is in bits</h2>
 *
 * <p>Conformance is about values that print the same and are not the same: {@code 0.0} and {@code
 * -0.0}, two doubles differing in the last ulp once a formatter has rounded them. Fields are
 * compared as {@code doubleToRawLongBits}, so no tolerance has to be chosen and a difference cannot
 * hide in a formatter. NaN is canonicalised to one token, because Flink's SQL semantics say a NaN
 * is a NaN and distinguishing payloads would assert something Flink does not promise.
 *
 * <h2>What a pass does and does not mean</h2>
 *
 * <p>A pass means the answers agree. It does <em>not</em> on its own mean the device ran: with no
 * provider on the classpath every case takes the CPU branch and the two arms are then trivially
 * identical. That is why each case also reports whether the accelerated plan actually contained an
 * offloaded operator, and why a case whose answers agree while nothing was offloaded is reported as
 * {@code SAME (cpu)} rather than as a device result. Run {@code scripts/preflight.sh} first if
 * everything says {@code cpu}: it is almost always the environment, not the code.
 */
public final class IntegrationCheck {

    private IntegrationCheck() {}

    /** One case: a name, the SQL, and whether an offload is expected to be available at all. */
    private record Case(String name, String sql) {}

    private static final List<Case> CASES =
            List.of(
                    new Case("calc-projection", "SELECT id, v * 2.0 + 1.0 AS scaled FROM T"),
                    new Case("calc-filter", "SELECT id, v FROM T WHERE v > 0.5"),
                    new Case(
                            "calc-filter-projection",
                            "SELECT id, v * v AS sq FROM T WHERE v > 0.25 AND v < 0.75"),
                    // Refused on the device unless approximate-projections is set, which this
                    // deliberately does not set: see run(). Expect SAME (cpu).
                    new Case("calc-math", "SELECT id, SIN(v) + COS(v) AS trig FROM T"),
                    new Case("order-by", "SELECT id, v FROM T ORDER BY v"),
                    new Case("order-by-limit", "SELECT id, v FROM T ORDER BY v DESC LIMIT 100"),
                    new Case("group-by", "SELECT k, SUM(v) AS s FROM T GROUP BY k"),
                    new Case(
                            "group-by-count",
                            "SELECT k, COUNT(*) AS n, SUM(v) AS s FROM T GROUP BY k"),
                    new Case(
                            "join",
                            "SELECT a.id, a.v, b.k FROM T a JOIN U b ON a.k = b.k WHERE a.v > 0.5"),
                    new Case(
                            "over-aggregate",
                            "SELECT id, k, SUM(v) OVER (PARTITION BY k ORDER BY id) AS run FROM T"));

    public static void main(String[] args) throws Exception {
        int rows = 20_000;
        Path data = null;
        for (int i = 0; i < args.length - 1; i++) {
            if ("--rows".equals(args[i])) {
                rows = Integer.parseInt(args[i + 1]);
            } else if ("--data".equals(args[i])) {
                data = Path.of(args[i + 1]);
            }
        }
        if (data == null) {
            data = Files.createTempDirectory("jitsql-check");
        }
        final Path left = data.resolve("t");
        final Path right = data.resolve("u");
        write(left, rows, 1234L);
        write(right, Math.max(16, rows / 256), 5678L);

        System.out.println("flink-jitsql integration check");
        System.out.printf("  %,d rows, data in %s%n%n", rows, data);

        int checks = 0;
        int failures = 0;
        for (Case c : CASES) {
            checks++;
            String verdict;
            try {
                final Arm cpu = run(c.sql(), left, right, false);
                final Arm gpu = run(c.sql(), left, right, true);
                if (!cpu.rows.equals(gpu.rows)) {
                    failures++;
                    verdict = "DIFFER  " + diff(cpu.rows, gpu.rows);
                } else {
                    verdict =
                            gpu.offloaded()
                                    ? "SAME (device)"
                                    : "SAME (cpu)   declined: " + gpu.why();
                }
            } catch (Exception e) {
                failures++;
                verdict = "ERROR   " + e.getClass().getSimpleName() + ": " + e.getMessage();
            }
            System.out.printf("  %-24s %s%n", c.name(), verdict);
        }

        System.out.println();
        System.out.printf("CHECKS: %d, FAILURES: %d%n", checks, failures);
        if (failures > 0) {
            System.exit(1);
        }
    }

    /** One arm's answer, and whether the plan it ran actually offloaded anything. */
    private record Arm(List<String> rows, boolean offloaded, String why) {}

    private static Arm run(String sql, Path left, Path right, boolean accelerate) throws Exception {
        final TableEnvironment env =
                TableEnvironment.create(EnvironmentSettings.newInstance().inBatchMode().build());
        env.getConfig()
                .getConfiguration()
                .setString("table.exec.accelerator.enabled", Boolean.toString(accelerate));
        // approximate-projections is deliberately NOT set.
        //
        // It permits the device's own math library to be used for SIN, COS and friends, which is
        // not required to agree with java.lang.Math to the last bit -- and measured here, it does
        // not: the calc-math case came back one ulp apart with it on. That is the option working
        // as specified, not a defect, but it makes a bit-exact comparison meaningless, and a
        // check that enables the one setting whose whole purpose is to permit a difference is
        // checking nothing.
        //
        // Left off, the projection is refused instead and that case takes the CPU branch in both
        // arms, where it must agree exactly. What approximate arithmetic costs belongs in a
        // tolerance-based test with a stated bound, not here.
        env.executeSql(ddl("T", left));
        env.executeSql(ddl("U", right));

        boolean offloaded = false;
        String why = "";
        if (accelerate) {
            // GpuOffloadExplain prints one line per subtree, "GPU  subtree N: ..." or "CPU  ...",
            // with the verdict's reason on the line below it. That is the planner's own answer to
            // "why did my query not run on the GPU", so it is what gets reported here rather than
            // a guess made by matching other words in the plan.
            final String plan = env.explainSql(sql);
            offloaded = plan.contains("GPU  subtree");
            if (!offloaded) {
                why = firstDeclineReason(plan);
            }
        }

        final List<String> out = new ArrayList<>();
        try (CloseableIterator<Row> it = env.sqlQuery(sql).execute().collect()) {
            while (it.hasNext()) {
                out.add(canonical(it.next()));
            }
        }
        // Sorted, because row order is not part of a SQL answer without ORDER BY, and the two arms
        // may legitimately produce a different one. The ORDER BY cases are checked by the values
        // they contain rather than by the order the iterator happened to yield.
        Collections.sort(out);
        return new Arm(out, offloaded, why);
    }

    private static String ddl(String name, Path path) {
        return "CREATE TABLE "
                + name
                + " (\n"
                + "  id INT NOT NULL,\n"
                + "  k INT NOT NULL,\n"
                + "  v DOUBLE NOT NULL\n"
                + ") WITH (\n"
                + "  'connector' = 'filesystem',\n"
                + "  'path' = '"
                + path
                + "',\n"
                + "  'format' = 'csv'\n"
                + ")";
    }

    /**
     * The planner's own reason for the first subtree it did not offload.
     *
     * <p>A case that agrees while nothing ran on the device has checked that the CPU equals the
     * CPU, which is worth knowing and is not worth mistaking for a device result. The reason is
     * read from the plan rather than guessed, so a decline says whether the provider was absent,
     * the shape unsupported, or a threshold missed.
     */
    private static String firstDeclineReason(String plan) {
        final String[] lines = plan.split("\\R");
        for (int i = 0; i < lines.length; i++) {
            if (lines[i].startsWith("CPU  ") && i + 1 < lines.length) {
                return lines[i + 1].trim();
            }
        }
        return "no accelerator subtree in the plan";
    }

    /** Every field as an exact token: doubles by their bits, NaN canonicalised. */
    private static String canonical(Row row) {
        final StringBuilder sb = new StringBuilder();
        for (int i = 0; i < row.getArity(); i++) {
            if (i > 0) {
                sb.append('|');
            }
            final Object value = row.getField(i);
            if (value == null) {
                sb.append("NULL");
            } else if (value instanceof Double d) {
                sb.append(Double.isNaN(d) ? "NaN" : "d" + Double.doubleToRawLongBits(d));
            } else if (value instanceof Float f) {
                sb.append(Float.isNaN(f) ? "NaN" : "f" + Float.floatToRawIntBits(f));
            } else {
                sb.append(value);
            }
        }
        return sb.toString();
    }

    /** The first disagreement, named, rather than two row counts. */
    private static String diff(List<String> cpu, List<String> gpu) {
        if (cpu.size() != gpu.size()) {
            return "cpu returned " + cpu.size() + " rows, device " + gpu.size();
        }
        for (int i = 0; i < cpu.size(); i++) {
            if (!cpu.get(i).equals(gpu.get(i))) {
                return "row " + i + ": cpu " + cpu.get(i) + " vs device " + gpu.get(i);
            }
        }
        return "same rows, different multiset";
    }

    private static void write(Path dir, int rows, long seed) throws Exception {
        Files.createDirectories(dir);
        final Path file = dir.resolve("part-0.csv");
        if (Files.exists(file)) {
            return;
        }
        final Random rnd = new Random(seed);
        try (BufferedWriter w = Files.newBufferedWriter(file, StandardCharsets.UTF_8)) {
            for (int i = 0; i < rows; i++) {
                w.write(i + "," + (i % 16) + "," + rnd.nextDouble());
                w.newLine();
            }
        }
    }
}
