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
package ai.beestream.flink.jitsql.catalogue;

import ai.beestream.flink.jitsql.codegen.GpuJoinSpec;
import ai.beestream.flink.jitsql.codegen.GpuOverAggregateSpec;
import ai.beestream.flink.jitsql.codegen.GpuSortSpec;

import org.apache.flink.table.accelerator.AccelAggCall;
import org.apache.flink.table.accelerator.AccelAggFunction;
import org.apache.flink.table.accelerator.AccelAggregate;
import org.apache.flink.table.accelerator.AccelFilter;
import org.apache.flink.table.accelerator.AccelJoin;
import org.apache.flink.table.accelerator.AccelNode;
import org.apache.flink.table.accelerator.AccelOverAggregate;
import org.apache.flink.table.accelerator.AccelProject;
import org.apache.flink.table.accelerator.AccelScan;
import org.apache.flink.table.accelerator.AccelSort;
import org.apache.flink.table.accelerator.AccelWorkProfile;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Which cuDF primitive expresses which IR node, and what to do when none does.
 *
 * <h2>The rule</h2>
 *
 * <blockquote>
 *
 * One IR node expressible as <em>one</em> cuDF primitive goes to cuDF. Anything else goes to the
 * kernel generator. What neither can express is refused, with a named reason.
 *
 * </blockquote>
 *
 * <p>It is a capability test, not a cost comparison, and with libraries out of scope it is total:
 * no node is expressible both ways, so nothing is left for a cost model to decide. That is why
 * nothing in this package mentions time, bandwidth or arithmetic intensity.
 *
 * <p>The word <em>one</em> is what keeps a projection out of cuDF. Per-column arithmetic does exist
 * in libcudf, but a projection would become a sequence of those with a materialised intermediate
 * between every pair, which is not direct expression -- it is an interpreter. The generator fuses
 * the whole tree into a single pass, so projections are its work by construction rather than by
 * measurement.
 *
 * <h2>What this replaces</h2>
 *
 * <p>The {@code instanceof} ladder in {@code TornadoVmAcceleratorProvider.acceptSubtree} and the
 * refusals scattered through {@code acceptAggregate}, {@code acceptSort}, {@code acceptJoin} and
 * {@code acceptOverAggregate}. Those refusals are preserved exactly -- the width and type tests
 * still live in {@code GpuSortSpec}, {@code GpuJoinSpec} and {@code GpuOverAggregateSpec}, and are
 * called from here rather than reimplemented.
 *
 * <h2>What this does not decide</h2>
 *
 * <ul>
 *   <li><b>Subtree admission.</b> The ceiling on operations per row, the drift-sensitive predicate
 *       refusal and the DECIMAL coercion refusal apply to a whole subtree and stay in the provider.
 *   <li><b>The library route.</b> {@code GpuGramSpec} and the cuBLAS engines are frozen and out of
 *       scope: the catalogue does not own them, and {@code acceptGram} remains the one hand-written
 *       branch in the provider. Freezing it leaves roughly 2x on the one workload where a wide
 *       reduction dimension makes a GEMM worth it; that is recorded rather than forgotten.
 *   <li><b>Whether the generator will succeed.</b> {@link Route.Kind#KERNEL} says the generator is
 *       the route to try, not that it will manage it. The provider turns a failed generation into a
 *       {@link RefusalCode#NO_KERNEL} refusal, because finding out costs a code generation and this
 *       class is consulted far more often than that.
 * </ul>
 */
public final class CudfCapabilityCatalogue {

    // ---------------------------------------------------------------------------------------
    // The primitives, as bound by uk.ac.manchester.tornado.cudf.Cudf in TornadoVM 8.1.0.
    //
    // Signatures are carried verbatim because these are native bindings: the argument order is not
    // guessable from the name, and a stage emitted against the wrong arity fails on the device
    // rather than in the compiler. Every one of them takes its row count as an ordinary int, which
    // is why every one of them is CAPTURED_AT_BUILD -- see RowCountBinding.
    // ---------------------------------------------------------------------------------------

    public static final CudfPrimitive SORTED_ORDER =
            CudfPrimitive.wired(
                    "sortedOrder",
                    "sortedOrder(int rows, IntArray keys, IntArray order)",
                    RowCountBinding.CAPTURED_AT_BUILD,
                    CudfPrimitive.Availability.CUDF);

    public static final CudfPrimitive GROUP_SUM =
            CudfPrimitive.wired(
                            "groupSum",
                            "groupSum(int rows, IntArray keys, DoubleArray values,"
                                    + " IntArray outKeys, DoubleArray outSums, IntArray outCount)",
                            RowCountBinding.CAPTURED_AT_BUILD,
                            CudfPrimitive.Availability.CUDF)
                    .writing(3);

    public static final CudfPrimitive RUNNING_SUM =
            CudfPrimitive.wired(
                    "runningSum",
                    "runningSum(int rows, DoubleArray values, DoubleArray out)",
                    RowCountBinding.CAPTURED_AT_BUILD,
                    CudfPrimitive.Availability.CUDF);

    public static final CudfPrimitive SELECTED_INDICES =
            CudfPrimitive.wired(
                            "selectedIndices",
                            "selectedIndices(int rows, int stride, ByteArray mask,"
                                    + " IntArray indices, IntArray count)",
                            RowCountBinding.CAPTURED_AT_BUILD,
                            CudfPrimitive.Availability.CUDF)
                    .writing(2);

    public static final CudfPrimitive INNER_JOIN =
            CudfPrimitive.wired(
                            "innerJoin",
                            "innerJoin(int buildRows, IntArray buildKeys, int probeRows,"
                                    + " IntArray probeKeys, int maxMatches, IntArray buildIdx,"
                                    + " IntArray probeIdx, IntArray count)",
                            RowCountBinding.CAPTURED_AT_BUILD,
                            CudfPrimitive.Availability.CUDF)
                    .writing(3);

    public static final CudfPrimitive READ_PARQUET =
            CudfPrimitive.wired(
                            "readParquet",
                            "readParquet(String path, int rowGroup, int columns, int rows,"
                                    + " int[] fields, long handle, IntArray meta, DoubleArray values)",
                            RowCountBinding.CAPTURED_AT_BUILD,
                            CudfPrimitive.Availability.PARQUET)
                    .writing(2);

    /**
     * An ungrouped reduction. Bound, and emitted today only by the device-region sources -- the SQL
     * aggregate path has no route that produces it, which is a gap this catalogue exists to show
     * rather than one it can close.
     */
    public static final CudfPrimitive REDUCE =
            CudfPrimitive.wired(
                    "reduce",
                    "reduce(int rows, int op, DoubleArray values, DoubleArray out)",
                    RowCountBinding.CAPTURED_AT_BUILD,
                    CudfPrimitive.Availability.CUDF);

    /**
     * Grouped aggregation over {@code CudfAggregation}, which is {@code SUM, MIN, MAX, MEAN,
     * COUNT}.
     *
     * <p>Worth reading twice, because it contradicts a refusal the provider has carried since M5.3:
     * the binding computes {@code MIN} and {@code MAX}, and the SQL path refuses them. The refusal
     * was correct when the only grouped entry point was {@code groupSum}; it is now a statement
     * about what this provider asks for rather than about what cuDF can do. {@code KMeansEngine}
     * already calls this primitive, so the gap is in the SQL aggregate route alone.
     */
    public static final CudfPrimitive GROUP_AGGREGATE =
            CudfPrimitive.bound(
                            "groupAggregate",
                            "groupAggregate(int rows, int groups, int aggregation, IntArray keys,"
                                    + " DoubleArray values, IntArray outKeys, DoubleArray outValues,"
                                    + " IntArray outCount)",
                            RowCountBinding.CAPTURED_AT_BUILD,
                            CudfPrimitive.Availability.CUDF,
                            "no AccelAggregate route emits it, so MIN and MAX stay refused in SQL")
                    .writing(3);

    private static final List<CudfPrimitive> PRIMITIVES =
            Collections.unmodifiableList(
                    new ArrayList<>(
                            List.of(
                                    SORTED_ORDER,
                                    GROUP_SUM,
                                    GROUP_AGGREGATE,
                                    RUNNING_SUM,
                                    REDUCE,
                                    SELECTED_INDICES,
                                    INNER_JOIN,
                                    READ_PARQUET)));

    // ---------------------------------------------------------------------------------------
    // The entries.
    // ---------------------------------------------------------------------------------------

    private static final Map<Class<? extends AccelNode>, CatalogueEntry> ENTRIES = buildEntries();

    private CudfCapabilityCatalogue() {}

    /** Every primitive the catalogue knows, wired or merely bound. For reporting and tests. */
    public static List<CudfPrimitive> primitives() {
        return PRIMITIVES;
    }

    /**
     * Routes one node.
     *
     * <p>Total by construction: a node kind with no entry is the generator's, because the rule's
     * second clause says so and not because of an oversight. Never returns null, never throws.
     */
    public static Route route(AccelNode node, AccelWorkProfile work, RoutingContext context) {
        if (node == null) {
            throw new IllegalArgumentException("a null node has no route");
        }
        if (!context.deviceAvailable()) {
            return RoutingDecisions.record(
                    node,
                    Route.refused(
                            node,
                            RefusalCode.DEVICE_UNAVAILABLE,
                            "TornadoVM is not usable in this JVM"));
        }
        CatalogueEntry entry = ENTRIES.get(node.getClass());
        if (entry == null) {
            // The rule's second clause. An unrecognised node kind is expression work until the
            // generator says otherwise, which is the conservative direction: the generator refuses
            // cleanly, whereas a cuDF stage emitted on a guess fails on the device.
            return RoutingDecisions.record(node, Route.kernel(node));
        }
        // Every decision passes through here, which is the only reason one log can claim to be
        // complete. A route recorded at each call site instead would be a log that is right until
        // somebody adds a call site.
        return RoutingDecisions.record(node, entry.route(node, work, context));
    }

    private static Map<Class<? extends AccelNode>, CatalogueEntry> buildEntries() {
        Map<Class<? extends AccelNode>, CatalogueEntry> entries = new LinkedHashMap<>();
        entries.put(AccelProject.class, new ProjectEntry());
        entries.put(AccelFilter.class, new FilterEntry());
        entries.put(AccelScan.class, new ScanEntry());
        entries.put(AccelSort.class, new SortEntry());
        entries.put(AccelAggregate.class, new AggregateEntry());
        entries.put(AccelJoin.class, new JoinEntry());
        entries.put(AccelOverAggregate.class, new OverAggregateEntry());
        return Collections.unmodifiableMap(entries);
    }

    private static Route cudfUnavailable(AccelNode node, String what) {
        return Route.refused(
                node,
                RefusalCode.CUDF_UNAVAILABLE,
                "the cuDF binding is not usable in this JVM, and " + what + " has no other route");
    }

    /** A projection is N primitives in cuDF and one pass in a kernel. The generator's, always. */
    private static final class ProjectEntry implements CatalogueEntry {
        @Override
        public Class<? extends AccelNode> nodeKind() {
            return AccelProject.class;
        }

        @Override
        public Route route(AccelNode node, AccelWorkProfile work, RoutingContext context) {
            return Route.kernel(node);
        }
    }

    /**
     * A filter is two stages wearing one node: the generator evaluates the predicate into a mask,
     * and {@code selectedIndices} turns that mask into a dense run of survivors. Naming both here
     * keeps the assembler from having to infer the second.
     */
    private static final class FilterEntry implements CatalogueEntry {
        @Override
        public Class<? extends AccelNode> nodeKind() {
            return AccelFilter.class;
        }

        @Override
        public Route route(AccelNode node, AccelWorkProfile work, RoutingContext context) {
            if (!context.available(CudfPrimitive.Availability.CUDF)) {
                // Still routable: a predicate kernel alone produces the mask, and the host can
                // compact. Slower, and correct, which is the right trade for a missing binding.
                return Route.kernel(node);
            }
            return Route.kernelThen(node, SELECTED_INDICES);
        }
    }

    private static final class ScanEntry implements CatalogueEntry {
        @Override
        public Class<? extends AccelNode> nodeKind() {
            return AccelScan.class;
        }

        @Override
        public Route route(AccelNode node, AccelWorkProfile work, RoutingContext context) {
            AccelScan scan = (AccelScan) node;
            if (!"parquet".equalsIgnoreCase(scan.format())) {
                return Route.refused(
                        node,
                        RefusalCode.NO_CUDF_PRIMITIVE,
                        "cuDF reads parquet on the device; this scan is " + scan.format());
            }
            if (!context.available(CudfPrimitive.Availability.PARQUET)) {
                return cudfUnavailable(node, "a device-side scan");
            }
            return Route.cudf(node, READ_PARQUET);
        }
    }

    private static final class SortEntry implements CatalogueEntry {
        @Override
        public Class<? extends AccelNode> nodeKind() {
            return AccelSort.class;
        }

        @Override
        public Route route(AccelNode node, AccelWorkProfile work, RoutingContext context) {
            if (!context.available(CudfPrimitive.Availability.CUDF)) {
                return cudfUnavailable(node, "an ordering");
            }
            AccelSort sort = (AccelSort) node;
            String refusal = GpuSortSpec.refuse(sort, work.estimatedRows(), context.maxSortBytes());
            if (refusal != null) {
                return Route.refused(node, RefusalCode.PRIMITIVE_SIGNATURE_MISMATCH, refusal);
            }
            return Route.cudf(node, SORTED_ORDER);
        }
    }

    private static final class JoinEntry implements CatalogueEntry {
        @Override
        public Class<? extends AccelNode> nodeKind() {
            return AccelJoin.class;
        }

        @Override
        public Route route(AccelNode node, AccelWorkProfile work, RoutingContext context) {
            if (!context.available(CudfPrimitive.Availability.CUDF)) {
                return cudfUnavailable(node, "a join");
            }
            String refusal = GpuJoinSpec.refuse((AccelJoin) node, context.maxJoinBytes());
            if (refusal != null) {
                return Route.refused(node, RefusalCode.PRIMITIVE_SIGNATURE_MISMATCH, refusal);
            }
            return Route.cudf(node, INNER_JOIN);
        }
    }

    private static final class OverAggregateEntry implements CatalogueEntry {
        @Override
        public Class<? extends AccelNode> nodeKind() {
            return AccelOverAggregate.class;
        }

        @Override
        public Route route(AccelNode node, AccelWorkProfile work, RoutingContext context) {
            if (!context.available(CudfPrimitive.Availability.CUDF)) {
                return cudfUnavailable(node, "a window aggregate");
            }
            String refusal = GpuOverAggregateSpec.refuse((AccelOverAggregate) node);
            if (refusal != null) {
                return Route.refused(node, RefusalCode.PRIMITIVE_SIGNATURE_MISMATCH, refusal);
            }
            return Route.cudf(node, RUNNING_SUM);
        }
    }

    /**
     * {@code groupSum} takes one key column and one value column, and that is the whole of what the
     * SQL aggregate route can express today.
     *
     * <p>Each refusal below is a shape the binding has no answer for, not a shape that would merely
     * be slow -- except the two marked {@link RefusalCode#PRIMITIVE_NOT_WIRED}, which are shapes
     * cuDF can compute and this provider does not ask it to.
     */
    private static final class AggregateEntry implements CatalogueEntry {
        @Override
        public Class<? extends AccelNode> nodeKind() {
            return AccelAggregate.class;
        }

        @Override
        public Route route(AccelNode node, AccelWorkProfile work, RoutingContext context) {
            if (!context.available(CudfPrimitive.Availability.CUDF)) {
                return cudfUnavailable(node, "an aggregate");
            }
            AccelAggregate agg = (AccelAggregate) node;
            if (agg.calls().size() != 1) {
                return Route.refused(
                        node,
                        RefusalCode.PRIMITIVE_SIGNATURE_MISMATCH,
                        agg.calls().size() + " aggregate calls, and the binding takes 1");
            }
            AccelAggCall call = agg.calls().get(0);
            if (agg.grouping().length == 0) {
                if (call.function() != AccelAggFunction.SUM
                        || call.inputField() == AccelAggCall.NO_INPUT_FIELD) {
                    return Route.refused(
                            node,
                            RefusalCode.UNSUPPORTED_AGGREGATION,
                            call + " is not a SUM over a column");
                }
                if (context.emitter() == RoutingContext.Emitter.REGION) {
                    return Route.cudf(node, REDUCE);
                }
                // Reachable from a region and from nowhere else. An operator decorates a plan
                // Flink already has, and there is no operator that emits a bare reduce over
                // buffers somebody else staged -- which is a gap in this provider rather than in
                // cuDF, and the refusal says which.
                return Route.refused(
                        node,
                        RefusalCode.PRIMITIVE_NOT_WIRED,
                        "Cudf.reduce expresses this ungrouped SUM, but only a device region"
                                + " emits it and this is an operator");
            }
            if (agg.grouping().length != 1) {
                return Route.refused(
                        node,
                        RefusalCode.PRIMITIVE_SIGNATURE_MISMATCH,
                        agg.grouping().length + " grouping columns, and the binding takes 1");
            }
            if (call.function() == AccelAggFunction.MAX) {
                return Route.refused(
                        node,
                        RefusalCode.PRIMITIVE_NOT_WIRED,
                        "Cudf.groupAggregate computes MAX, but "
                                + GROUP_AGGREGATE.notWiredBecause());
            }
            if (call.function() != AccelAggFunction.SUM
                    || call.inputField() == AccelAggCall.NO_INPUT_FIELD) {
                return Route.refused(
                        node,
                        RefusalCode.UNSUPPORTED_AGGREGATION,
                        call + " is not a SUM over a column");
            }
            return Route.cudf(node, GROUP_SUM);
        }
    }
}
