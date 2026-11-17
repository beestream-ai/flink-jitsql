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

import org.apache.flink.table.accelerator.AccelAggCall;
import org.apache.flink.table.accelerator.AccelAggFunction;
import org.apache.flink.table.accelerator.AccelAggregate;
import org.apache.flink.table.accelerator.AccelFilter;
import org.apache.flink.table.accelerator.AccelFunction;
import org.apache.flink.table.accelerator.AccelInput;
import org.apache.flink.table.accelerator.AccelInputRef;
import org.apache.flink.table.accelerator.AccelJoin;
import org.apache.flink.table.accelerator.AccelNode;
import org.apache.flink.table.accelerator.AccelOverAggregate;
import org.apache.flink.table.accelerator.AccelProject;
import org.apache.flink.table.accelerator.AccelScan;
import org.apache.flink.table.accelerator.AccelSort;
import org.apache.flink.table.accelerator.AccelWorkProfile;
import org.apache.flink.table.types.logical.BigIntType;
import org.apache.flink.table.types.logical.BooleanType;
import org.apache.flink.table.types.logical.DoubleType;
import org.apache.flink.table.types.logical.IntType;
import org.apache.flink.table.types.logical.RowType;

import org.junit.jupiter.api.Test;

import java.io.File;
import java.lang.reflect.Modifier;
import java.net.URI;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Enumeration;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;
import java.util.jar.JarEntry;
import java.util.jar.JarFile;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * The catalogue's one structural promise: every node gets a definite answer.
 *
 * <p>A routing table that can return nothing is a routing table that silently drops a node, and a
 * dropped node is a query that quietly ran on the CPU with nobody able to say why. That is the
 * failure these tests exist to make impossible, so most of them are about totality rather than
 * about any particular primitive.
 */
class CudfCapabilityCatalogueTest {

    private static final long ONE_GIB = 1L << 30;

    private static final RowType KEYED =
            RowType.of(new IntType(false), new DoubleType(false), new BigIntType(false));

    private static final AccelWorkProfile WORK = work();

    /**
     * {@code AccelWorkProfile} counts operations per row in one slot per {@link AccelFunction}, so
     * the array is as long as that enum rather than as long as the expression. Built here rather
     * than written out, so that a function added upstream does not break every test in this file.
     */
    private static AccelWorkProfile work() {
        int[] opsPerRow = new int[AccelFunction.values().length];
        opsPerRow[AccelFunction.PLUS.ordinal()] = 4;
        return new AccelWorkProfile(opsPerRow, 24, 24, 1_000_000L);
    }

    private static RoutingContext full() {
        return RoutingContext.fullyAvailable(ONE_GIB);
    }

    /** One instance of every node kind the catalogue can meet. */
    private static List<AccelNode> everyNodeKind() {
        AccelInput input = new AccelInput(KEYED);
        List<AccelNode> nodes = new ArrayList<>();
        nodes.add(input);
        nodes.add(
                new AccelProject(
                        List.of(new AccelInputRef(0, new IntType(false))),
                        input,
                        RowType.of(new IntType(false))));
        nodes.add(new AccelFilter(new AccelInputRef(0, new BooleanType(false)), input, KEYED));
        nodes.add(new AccelScan("parquet", List.of("/tmp/x.parquet"), new int[] {0, 1, 2}, KEYED));
        nodes.add(new AccelSort(0, true, true, input, KEYED));
        nodes.add(
                new AccelAggregate(
                        new int[] {0},
                        List.of(new AccelAggCall(AccelAggFunction.SUM, 1, new DoubleType(false))),
                        input,
                        RowType.of(new IntType(false), new DoubleType(false))));
        nodes.add(new AccelJoin(0, 0, true, 1_000L, input, input, KEYED));
        nodes.add(new AccelOverAggregate(1, input, KEYED));
        return nodes;
    }

    @Test
    void everyNodeKindGetsADefiniteRoute() {
        List<RoutingContext> contexts =
                List.of(full(), RoutingContext.kernelOnly(ONE_GIB), RoutingContext.noDevice());
        for (AccelNode node : everyNodeKind()) {
            for (RoutingContext context : contexts) {
                Route route = CudfCapabilityCatalogue.route(node, WORK, context);
                assertThat(route)
                        .as("no route at all for %s", node.getClass().getSimpleName())
                        .isNotNull();
                assertThat(route.kind()).isNotNull();
                if (route.kind() == Route.Kind.CUDF) {
                    assertThat(route.primitive()).isPresent();
                } else if (route.kind() == Route.Kind.REFUSED) {
                    assertThat(route.code())
                            .as("a refusal without a code for %s", node)
                            .isPresent();
                    assertThat(route.because())
                            .as("a refusal without a sentence for %s", node)
                            .isPresent()
                            .get()
                            .asString()
                            .isNotBlank();
                }
            }
        }
    }

    /**
     * A new IR node kind must not route by accident.
     *
     * <p>Totality over the kinds we know of is worth little if Flink can add a ninth and have it
     * fall through the default arm unnoticed. This pins the set: adding a node kind upstream fails
     * here and forces someone to decide what it is, which is the whole point.
     */
    @Test
    void noNodeKindIsSilentlyAdded() throws Exception {
        Set<String> expected =
                new TreeSet<>(
                        Arrays.asList(
                                "AccelAggregate",
                                "AccelFilter",
                                "AccelInput",
                                "AccelJoin",
                                "AccelOverAggregate",
                                "AccelProject",
                                "AccelScan",
                                "AccelSort"));
        assertThat(nodeKindsOnTheClasspath()).isEqualTo(expected);
    }

    private static Set<String> nodeKindsOnTheClasspath() throws Exception {
        URI source = AccelNode.class.getProtectionDomain().getCodeSource().getLocation().toURI();
        String packagePath = "org/apache/flink/table/accelerator";
        Set<String> simpleNames = new LinkedHashSet<>();
        File location = new File(source);
        if (location.isDirectory()) {
            File[] files = new File(location, packagePath).listFiles();
            assertThat(files).as("no accelerator package under %s", location).isNotNull();
            for (File file : files) {
                collect(file.getName(), simpleNames);
            }
        } else {
            // flink-table-common resolves as a jar in an ordinary build, and as a directory only
            // when someone has the Flink reactor open beside this one. Both happen here.
            try (JarFile jar = new JarFile(location)) {
                for (Enumeration<JarEntry> entries = jar.entries(); entries.hasMoreElements(); ) {
                    String name = entries.nextElement().getName();
                    if (name.startsWith(packagePath + "/")) {
                        collect(name.substring(packagePath.length() + 1), simpleNames);
                    }
                }
            }
        }
        Set<String> kinds = new TreeSet<>();
        for (String simple : simpleNames) {
            Class<?> type = Class.forName("org.apache.flink.table.accelerator." + simple);
            if (AccelNode.class.isAssignableFrom(type)
                    && !type.isInterface()
                    && !Modifier.isAbstract(type.getModifiers())) {
                kinds.add(simple);
            }
        }
        return kinds;
    }

    private static void collect(String fileName, Set<String> into) {
        if (fileName.endsWith(".class") && !fileName.contains("$") && !fileName.contains("/")) {
            into.add(fileName.substring(0, fileName.length() - ".class".length()));
        }
    }

    @Test
    void aProjectionIsTheGeneratorsWork() {
        AccelNode project =
                new AccelProject(
                        List.of(new AccelInputRef(0, new IntType(false))),
                        new AccelInput(KEYED),
                        RowType.of(new IntType(false)));
        Route route = CudfCapabilityCatalogue.route(project, WORK, full());
        assertThat(route.kind()).isEqualTo(Route.Kind.KERNEL);
        assertThat(route.compaction()).isEmpty();
    }

    @Test
    void aFilterIsAKernelAndThenACompaction() {
        AccelNode filter =
                new AccelFilter(
                        new AccelInputRef(0, new BooleanType(false)), new AccelInput(KEYED), KEYED);
        Route route = CudfCapabilityCatalogue.route(filter, WORK, full());
        assertThat(route.kind()).isEqualTo(Route.Kind.KERNEL);
        assertThat(route.compaction()).get().isEqualTo(CudfCapabilityCatalogue.SELECTED_INDICES);
    }

    @Test
    void aParquetScanIsCudfsWork() {
        AccelNode scan =
                new AccelScan("parquet", List.of("/tmp/x.parquet"), new int[] {0, 1, 2}, KEYED);
        Route route = CudfCapabilityCatalogue.route(scan, WORK, full());
        assertThat(route.kind()).isEqualTo(Route.Kind.CUDF);
        assertThat(route.primitive()).get().isEqualTo(CudfCapabilityCatalogue.READ_PARQUET);
    }

    @Test
    void aScanOfAnotherFormatHasNoPrimitive() {
        AccelNode scan = new AccelScan("csv", List.of("/tmp/x.csv"), new int[] {0, 1, 2}, KEYED);
        Route route = CudfCapabilityCatalogue.route(scan, WORK, full());
        assertThat(route.code()).get().isEqualTo(RefusalCode.NO_CUDF_PRIMITIVE);
        assertThat(route.because()).get().asString().contains("csv");
    }

    @Test
    void withoutADeviceEveryRouteRefusesWithAReason() {
        for (AccelNode node : everyNodeKind()) {
            Route route = CudfCapabilityCatalogue.route(node, WORK, RoutingContext.noDevice());
            assertThat(route.kind()).isEqualTo(Route.Kind.REFUSED);
            assertThat(route.code()).get().isEqualTo(RefusalCode.DEVICE_UNAVAILABLE);
        }
    }

    @Test
    void withoutCudfAProjectionStillRoutesToTheGenerator() {
        AccelNode project =
                new AccelProject(
                        List.of(new AccelInputRef(0, new IntType(false))),
                        new AccelInput(KEYED),
                        RowType.of(new IntType(false)));
        Route route =
                CudfCapabilityCatalogue.route(project, WORK, RoutingContext.kernelOnly(ONE_GIB));
        assertThat(route.kind()).isEqualTo(Route.Kind.KERNEL);
    }

    @Test
    void twoGroupingColumnsDoNotFitTheBinding() {
        AccelNode agg =
                new AccelAggregate(
                        new int[] {0, 1},
                        List.of(new AccelAggCall(AccelAggFunction.SUM, 2, new DoubleType(false))),
                        new AccelInput(KEYED),
                        KEYED);
        Route route = CudfCapabilityCatalogue.route(agg, WORK, full());
        assertThat(route.code()).get().isEqualTo(RefusalCode.PRIMITIVE_SIGNATURE_MISMATCH);
        assertThat(route.because()).get().asString().contains("2 grouping columns");
    }

    @Test
    void oneKeyAndOneSumIsCudfsWork() {
        AccelNode agg =
                new AccelAggregate(
                        new int[] {0},
                        List.of(new AccelAggCall(AccelAggFunction.SUM, 1, new DoubleType(false))),
                        new AccelInput(KEYED),
                        RowType.of(new IntType(false), new DoubleType(false)));
        Route route = CudfCapabilityCatalogue.route(agg, WORK, full());
        assertThat(route.kind()).isEqualTo(Route.Kind.CUDF);
        assertThat(route.primitive()).get().isEqualTo(CudfCapabilityCatalogue.GROUP_SUM);
    }

    /**
     * MAX is refused because nothing emits it, not because cuDF cannot compute it.
     *
     * <p>The distinction is the catalogue earning its keep: {@code Cudf.groupAggregate} takes a
     * {@code CudfAggregation} of SUM, MIN, MAX, MEAN or COUNT, and the SQL path has carried a
     * blanket refusal of MIN and MAX since the only grouped entry point was {@code groupSum}. The
     * refusal stands until something emits the primitive, but it now says which of the two it is.
     */
    @Test
    void maxIsRefusedAsUnwiredRatherThanAsImpossible() {
        AccelNode agg =
                new AccelAggregate(
                        new int[] {0},
                        List.of(new AccelAggCall(AccelAggFunction.MAX, 1, new DoubleType(false))),
                        new AccelInput(KEYED),
                        RowType.of(new IntType(false), new DoubleType(false)));
        Route route = CudfCapabilityCatalogue.route(agg, WORK, full());
        assertThat(route.kind()).isEqualTo(Route.Kind.REFUSED);
        assertThat(route.code()).get().isEqualTo(RefusalCode.PRIMITIVE_NOT_WIRED);
        assertThat(route.because()).get().asString().contains("groupAggregate");
    }

    @Test
    void anUngroupedSumNamesTheReduceItDoesNotYetEmit() {
        AccelNode agg =
                new AccelAggregate(
                        new int[] {},
                        List.of(new AccelAggCall(AccelAggFunction.SUM, 1, new DoubleType(false))),
                        new AccelInput(KEYED),
                        RowType.of(new DoubleType(false)));
        Route route = CudfCapabilityCatalogue.route(agg, WORK, full());
        assertThat(route.code()).get().isEqualTo(RefusalCode.PRIMITIVE_NOT_WIRED);
        assertThat(route.because()).get().asString().contains("reduce");
    }

    /**
     * Every bound primitive fixes its row count when the graph is built.
     *
     * <p>Pinned as a test because it is the assumption the assembler and the plan cache are both
     * built on: one graph per distinct row count, and the row count in the cache key. If a future
     * binding takes its count at execution instead, this fails and the two of them get revisited
     * rather than quietly going wrong on a short final batch.
     */
    @Test
    void everyPrimitiveCapturesItsRowCountWhenTheGraphIsBuilt() {
        assertThat(CudfCapabilityCatalogue.primitives())
                .isNotEmpty()
                .allSatisfy(
                        p ->
                                assertThat(p.rowCountBinding())
                                        .as("%s", p.name())
                                        .isEqualTo(RowCountBinding.CAPTURED_AT_BUILD));
    }

    @Test
    void everyPrimitiveCarriesItsSignature() {
        assertThat(CudfCapabilityCatalogue.primitives())
                .allSatisfy(
                        p -> {
                            assertThat(p.signature()).contains(p.name() + "(");
                            if (!p.wired()) {
                                assertThat(p.notWiredBecause()).isNotBlank();
                            }
                        });
    }
}
