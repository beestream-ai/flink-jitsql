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
package ai.beestream.flink.jitsql.region;

import ai.beestream.flink.jitsql.catalogue.CudfCapabilityCatalogue;
import ai.beestream.flink.jitsql.catalogue.RefusalCode;
import ai.beestream.flink.jitsql.catalogue.Route;
import ai.beestream.flink.jitsql.catalogue.RoutingContext;

import org.apache.flink.table.accelerator.AccelAggCall;
import org.apache.flink.table.accelerator.AccelAggFunction;
import org.apache.flink.table.accelerator.AccelAggregate;
import org.apache.flink.table.accelerator.AccelFilter;
import org.apache.flink.table.accelerator.AccelFunction;
import org.apache.flink.table.accelerator.AccelInputRef;
import org.apache.flink.table.accelerator.AccelNode;
import org.apache.flink.table.accelerator.AccelProject;
import org.apache.flink.table.accelerator.AccelScan;
import org.apache.flink.table.accelerator.AccelSort;
import org.apache.flink.table.accelerator.AccelWorkProfile;
import org.apache.flink.table.types.logical.BooleanType;
import org.apache.flink.table.types.logical.DoubleType;
import org.apache.flink.table.types.logical.IntType;
import org.apache.flink.table.types.logical.RowType;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.stream.Collectors;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * A2 and A4, stated as tests: four known shapes with no code written for them, a fifth nobody has
 * ever written code for, and a subtree declined whole rather than half-offloaded.
 *
 * <p>These assemble a region; they do not run one. Everything structural about the assembler --
 * which stages, in what order, reading whose buffers -- is decided here and is the same with or
 * without a GPU. What a device would add is whether the stages compute the right numbers, which is
 * the device suite's job and not this one's.
 */
class RegionAssemblerTest {

    private static final int BATCH = 262_144;

    private static final RowType SCAN_ROW =
            RowType.of(new IntType(false), new DoubleType(false), new DoubleType(false));

    private static final RowType ONE_DOUBLE = RowType.of(new DoubleType(false));

    private static AccelWorkProfile work() {
        int[] opsPerRow = new int[AccelFunction.values().length];
        opsPerRow[AccelFunction.TIMES.ordinal()] = 6;
        return new AccelWorkProfile(opsPerRow, 24, 8, 10_000_000L);
    }

    private static RoutingContext context() {
        return RoutingContext.fullyAvailable(1L << 30);
    }

    private static AccelScan scan() {
        return new AccelScan(
                "parquet", List.of("/data/yellow.parquet"), new int[] {0, 1, 2}, SCAN_ROW);
    }

    private static AccelProject projectOne(AccelNode input) {
        return new AccelProject(
                List.of(new AccelInputRef(1, new DoubleType(false))), input, ONE_DOUBLE);
    }

    private static RegionPlan assemble(AccelNode subtree) {
        return RegionAssembler.assemble(subtree, work(), context(), BATCH);
    }

    private static List<String> primitives(RegionPlan plan) {
        return plan.stages().stream()
                .map(stage -> stage.route().primitive().map(p -> p.name()).orElse("kernel"))
                .collect(Collectors.toList());
    }

    // ------------------------------------------------------------------------------------
    // The four regions that exist as hand-written classes today.
    // ------------------------------------------------------------------------------------

    /** {@code DeviceParquetProjectSumSource}, assembled instead of written. */
    @Test
    void theProjectSumRegion() {
        AccelNode subtree =
                new AccelAggregate(
                        new int[] {},
                        List.of(new AccelAggCall(AccelAggFunction.SUM, 0, new DoubleType(false))),
                        projectOne(scan()),
                        ONE_DOUBLE);
        RegionPlan plan = assemble(subtree);
        assertThat(plan.accepted()).as("%s", plan).isTrue();
        assertThat(primitives(plan)).containsExactly("readParquet", "kernel", "reduce");
        assertThat(plan.cudfStages()).isEqualTo(2);
        assertThat(plan.kernelStages()).isEqualTo(1);
    }

    /** {@code DeviceParquetGroupSumSource}. */
    @Test
    void theGroupSumRegion() {
        AccelNode subtree =
                new AccelAggregate(
                        new int[] {0},
                        List.of(new AccelAggCall(AccelAggFunction.SUM, 0, new DoubleType(false))),
                        projectOne(scan()),
                        RowType.of(new IntType(false), new DoubleType(false)));
        RegionPlan plan = assemble(subtree);
        assertThat(plan.accepted()).as("%s", plan).isTrue();
        assertThat(primitives(plan)).containsExactly("readParquet", "kernel", "groupSum");
        // groupSum writes keys, sums and a group count, and the assembler knows that from the
        // catalogue rather than from knowing what a group-by is.
        assertThat(plan.stages().get(2).outputs()).hasSize(3);
    }

    /**
     * {@code DeviceParquetTopNSource}.
     *
     * <p>Ordered on an {@code INT} key because that is what the binding orders -- a {@code DOUBLE}
     * key is refused by {@code GpuSortSpec}, which the assembler reports as a declined subtree
     * rather than discovering on the device.
     */
    @Test
    void theTopNRegion() {
        AccelNode keyed =
                new AccelProject(
                        List.of(new AccelInputRef(0, new IntType(false))),
                        scan(),
                        RowType.of(new IntType(false)));
        AccelNode subtree =
                new AccelSort(0, true, true, 100L, keyed, RowType.of(new IntType(false)));
        RegionPlan plan = assemble(subtree);
        assertThat(plan.accepted()).as("%s", plan).isTrue();
        assertThat(primitives(plan)).containsExactly("readParquet", "kernel", "sortedOrder");
    }

    /** {@code DeviceParquetGrokSource}: a scan feeding a projection, and nothing after it. */
    @Test
    void theProjectionOnlyRegion() {
        RegionPlan plan = assemble(projectOne(scan()));
        assertThat(plan.accepted()).as("%s", plan).isTrue();
        assertThat(primitives(plan)).containsExactly("readParquet", "kernel");
    }

    // ------------------------------------------------------------------------------------
    // A2's actual acceptance criterion.
    // ------------------------------------------------------------------------------------

    /**
     * A shape nobody wrote a region for: a filtered grouped aggregate over Parquet.
     *
     * <p>No {@code DeviceParquetFilteredGroupSumSource} exists and none is needed. Five stages come
     * out, including the compaction that a filter needs and that no part of the assembler knows is
     * a filter's business -- the catalogue names it and the assembler emits it.
     */
    @Test
    void aShapeNobodyWroteARegionFor() {
        AccelNode filtered =
                new AccelFilter(
                        new AccelInputRef(0, new BooleanType(false)),
                        projectOne(scan()),
                        ONE_DOUBLE);
        AccelNode subtree =
                new AccelAggregate(
                        new int[] {0},
                        List.of(new AccelAggCall(AccelAggFunction.SUM, 0, new DoubleType(false))),
                        filtered,
                        RowType.of(new IntType(false), new DoubleType(false)));
        RegionPlan plan = assemble(subtree);
        assertThat(plan.accepted()).as("%s", plan).isTrue();
        assertThat(primitives(plan))
                .containsExactly("readParquet", "kernel", "kernel", "selectedIndices", "groupSum");
    }

    @Test
    void everyStageReadsTheStageBeforeIt() {
        AccelNode subtree =
                new AccelAggregate(
                        new int[] {0},
                        List.of(new AccelAggCall(AccelAggFunction.SUM, 0, new DoubleType(false))),
                        projectOne(scan()),
                        RowType.of(new IntType(false), new DoubleType(false)));
        RegionPlan plan = assemble(subtree);
        for (Stage stage : plan.stages()) {
            if (stage.index() == 0) {
                assertThat(stage.inputs()).isEmpty();
            } else {
                assertThat(stage.inputs()).as("stage %s reads nothing", stage.index()).isNotEmpty();
            }
            assertThat(stage.rows()).isEqualTo(BATCH);
        }
    }

    @Test
    void theResultIsTheLastStagesOutput() {
        RegionPlan plan = assemble(projectOne(scan()));
        assertThat(plan.result())
                .isEqualTo(plan.stages().get(plan.stages().size() - 1).outputs().get(0));
    }

    // ------------------------------------------------------------------------------------
    // A4: a subtree is admitted whole or not at all.
    // ------------------------------------------------------------------------------------

    /**
     * One unroutable stage declines the subtree, and the refusal says which stage.
     *
     * <p>Half a region is worse than none: the batch drains back to rows at the stage that has no
     * route, paying the staging cost the region exists to remove while keeping the compilation cost
     * it adds. Measured at 1.62x against 4.89x in the Spark integration, for exactly this.
     */
    @Test
    void aSubtreeWithOneUnroutableStageIsDeclinedWhole() {
        AccelNode subtree =
                new AccelAggregate(
                        new int[] {0, 1},
                        List.of(new AccelAggCall(AccelAggFunction.SUM, 0, new DoubleType(false))),
                        projectOne(scan()),
                        RowType.of(new IntType(false), new DoubleType(false)));
        RegionPlan plan = assemble(subtree);
        assertThat(plan.accepted()).isFalse();
        assertThat(plan.code()).get().isEqualTo(RefusalCode.PRIMITIVE_SIGNATURE_MISMATCH);
        assertThat(plan.because())
                .get()
                .asString()
                .contains("stage 2")
                .contains("AccelAggregate")
                .contains("2 grouping columns");
        assertThat(plan.refusedAt()).get().isInstanceOf(AccelAggregate.class);
    }

    @Test
    void aScanOfAnUnreadableFormatDeclinesAtTheFirstStage() {
        AccelScan csv =
                new AccelScan("csv", List.of("/data/yellow.csv"), new int[] {0, 1, 2}, SCAN_ROW);
        RegionPlan plan = assemble(projectOne(csv));
        assertThat(plan.accepted()).isFalse();
        assertThat(plan.because()).get().asString().contains("stage 0").contains("csv");
    }

    @Test
    void withoutADeviceNothingIsAssembled() {
        RegionPlan plan =
                RegionAssembler.assemble(
                        projectOne(scan()), work(), RoutingContext.noDevice(), BATCH);
        assertThat(plan.accepted()).isFalse();
        assertThat(plan.code()).get().isEqualTo(RefusalCode.DEVICE_UNAVAILABLE);
    }

    /**
     * An ungrouped SUM is cuDF's work in a region and nobody's in an operator.
     *
     * <p>The same node, the same binding, two answers -- because {@code accept} decorates a plan
     * Flink already has and there is no operator that emits a bare reduce. Asking on behalf of a
     * caller is what keeps the catalogue honest to both.
     */
    @Test
    void anUngroupedSumIsReachableFromARegionAndNotFromAnOperator() {
        AccelNode aggregate =
                new AccelAggregate(
                        new int[] {},
                        List.of(new AccelAggCall(AccelAggFunction.SUM, 0, new DoubleType(false))),
                        projectOne(scan()),
                        ONE_DOUBLE);
        Route asOperator = CudfCapabilityCatalogue.route(aggregate, work(), context());
        assertThat(asOperator.code()).get().isEqualTo(RefusalCode.PRIMITIVE_NOT_WIRED);

        Route asRegion = CudfCapabilityCatalogue.route(aggregate, work(), context().forRegion());
        assertThat(asRegion.kind()).isEqualTo(Route.Kind.CUDF);
        assertThat(asRegion.primitive()).get().isEqualTo(CudfCapabilityCatalogue.REDUCE);
    }

    @Test
    void aRegionCannotBePlannedForNoRows() {
        assertThat(
                        org.assertj.core.api.Assertions.catchThrowable(
                                () ->
                                        RegionAssembler.assemble(
                                                projectOne(scan()), work(), context(), 0)))
                .isInstanceOf(IllegalArgumentException.class);
    }
}
