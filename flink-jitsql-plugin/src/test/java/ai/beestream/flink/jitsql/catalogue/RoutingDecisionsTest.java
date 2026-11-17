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
import org.apache.flink.table.accelerator.AccelFunction;
import org.apache.flink.table.accelerator.AccelInput;
import org.apache.flink.table.accelerator.AccelInputRef;
import org.apache.flink.table.accelerator.AccelNode;
import org.apache.flink.table.accelerator.AccelProject;
import org.apache.flink.table.accelerator.AccelWorkProfile;
import org.apache.flink.table.types.logical.DoubleType;
import org.apache.flink.table.types.logical.IntType;
import org.apache.flink.table.types.logical.RowType;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.concurrent.atomic.AtomicReference;

import static org.assertj.core.api.Assertions.assertThat;

/** What was tried, what won, and why each loser lost. */
class RoutingDecisionsTest {

    private static final RowType ROW =
            RowType.of(new IntType(false), new DoubleType(false), new DoubleType(false));

    private static AccelWorkProfile work() {
        return new AccelWorkProfile(new int[AccelFunction.values().length], 24, 8, 1_000L);
    }

    private static AccelNode project() {
        return new AccelProject(
                List.of(new AccelInputRef(1, new DoubleType(false))),
                new AccelInput(ROW),
                RowType.of(new DoubleType(false)));
    }

    private static AccelNode unroutableAggregate() {
        return new AccelAggregate(
                new int[] {0, 1},
                List.of(new AccelAggCall(AccelAggFunction.SUM, 2, new DoubleType(false))),
                new AccelInput(ROW),
                ROW);
    }

    @BeforeEach
    @AfterEach
    void clear() {
        RoutingDecisions.clear();
    }

    @Test
    void everyRoutingDecisionIsKeptInOrder() {
        RoutingContext context = RoutingContext.fullyAvailable(1L << 30);
        CudfCapabilityCatalogue.route(project(), work(), context);
        CudfCapabilityCatalogue.route(unroutableAggregate(), work(), context);
        assertThat(RoutingDecisions.decisions())
                .extracting(RoutingDecisions.Decision::node)
                .containsExactly("AccelProject", "AccelAggregate");
    }

    @Test
    void theRefusalsAreSeparableFromTheAcceptances() {
        RoutingContext context = RoutingContext.fullyAvailable(1L << 30);
        CudfCapabilityCatalogue.route(project(), work(), context);
        CudfCapabilityCatalogue.route(unroutableAggregate(), work(), context);
        assertThat(RoutingDecisions.refusals()).hasSize(1);
        assertThat(RoutingDecisions.refusals().get(0).route().code())
                .get()
                .isEqualTo(RefusalCode.PRIMITIVE_SIGNATURE_MISMATCH);
    }

    @Test
    void aRefusalNamesTheConstructThatCausedIt() {
        CudfCapabilityCatalogue.route(
                unroutableAggregate(), work(), RoutingContext.fullyAvailable(1L << 30));
        assertThat(RoutingDecisions.format())
                .contains("AccelAggregate")
                .contains("PRIMITIVE_SIGNATURE_MISMATCH")
                .contains("2 grouping columns");
    }

    @Test
    void anEmptyLogSaysSoRatherThanBeingBlank() {
        assertThat(RoutingDecisions.format()).contains("nothing was offered");
    }

    /** A long job routes a great many nodes, and none of this is worth a leak. */
    @Test
    void theLogIsBounded() {
        RoutingContext context = RoutingContext.fullyAvailable(1L << 30);
        for (int i = 0; i < RoutingDecisions.KEEP * 3; i++) {
            CudfCapabilityCatalogue.route(project(), work(), context);
        }
        assertThat(RoutingDecisions.decisions()).hasSize(RoutingDecisions.KEEP);
    }

    /** Two subtasks route in one JVM; mixing their decisions makes neither readable. */
    @Test
    void oneThreadsDecisionsDoNotAppearInAnothers() throws Exception {
        RoutingContext context = RoutingContext.fullyAvailable(1L << 30);
        CudfCapabilityCatalogue.route(project(), work(), context);
        AtomicReference<Integer> seenElsewhere = new AtomicReference<>();
        Thread other =
                new Thread(
                        () -> {
                            seenElsewhere.set(RoutingDecisions.decisions().size());
                            RoutingDecisions.clear();
                        });
        other.start();
        other.join();
        assertThat(seenElsewhere.get()).isZero();
        assertThat(RoutingDecisions.decisions()).hasSize(1);
    }
}
