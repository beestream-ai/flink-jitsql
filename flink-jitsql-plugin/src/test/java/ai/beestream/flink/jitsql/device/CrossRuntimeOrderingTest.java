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
package ai.beestream.flink.jitsql.device;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.concurrent.atomic.AtomicInteger;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/** Ordering a kernel against whoever filled its buffers, and charging for it. */
class CrossRuntimeOrderingTest {

    @BeforeEach
    void clear() {
        CrossRuntimeOrdering.reset();
    }

    @Test
    void theInGraphBarrierWaitsForNothing() {
        long elapsed = CrossRuntimeOrdering.await(CrossRuntimeOrdering.IN_GRAPH);
        assertThat(elapsed).isGreaterThanOrEqualTo(0);
        assertThat(CrossRuntimeOrdering.waits()).isEqualTo(1);
    }

    @Test
    void aProducersBarrierIsActuallyWaitedFor() {
        AtomicInteger awaited = new AtomicInteger();
        CrossRuntimeOrdering.await(awaited::incrementAndGet);
        assertThat(awaited).hasValue(1);
    }

    /**
     * The cost is counted, which is the whole of B3's acceptance criterion.
     *
     * <p>A whole-device wait is the coarse way to order two runtimes and it is not free. Left out
     * of a breakdown, its time is attributed to the kernel, and the kernel is then reported as
     * slower than it is -- which is how an integration ends up optimising the wrong thing.
     */
    @Test
    void everyWaitIsCountedAndTimed() {
        CrossRuntimeOrdering.await(() -> sleepALittle());
        CrossRuntimeOrdering.await(() -> sleepALittle());
        assertThat(CrossRuntimeOrdering.waits()).isEqualTo(2);
        assertThat(CrossRuntimeOrdering.nanos()).isGreaterThan(0);
        assertThat(CrossRuntimeOrdering.line()).contains("2 waits").contains("ms");
    }

    @Test
    void aProfileWithNoWaitsSaysWhyRatherThanBeingBlank() {
        assertThat(CrossRuntimeOrdering.line()).contains("no waits").contains("same task graph");
    }

    /**
     * A missing barrier is refused rather than treated as "nothing to wait for".
     *
     * <p>The two look identical at the call site and are opposites: one says the producer is a task
     * of the same graph, the other says nobody considered the question. Only the first is safe, and
     * only the first is spellable.
     */
    @Test
    void aMissingBarrierIsNotTheSameAsNoWait() {
        assertThatThrownBy(() -> CrossRuntimeOrdering.await(null))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("IN_GRAPH");
    }

    /**
     * Nothing to bind means nothing to wait for.
     *
     * <p>The lookup comes before the barrier on purpose: a wait that orders a read which is never
     * going to happen is pure cost, and on the coarse path that cost is a whole-device
     * synchronisation.
     */
    @Test
    void aBindThatCannotHappenDoesNotPayForOrdering() {
        AtomicInteger awaited = new AtomicInteger();
        try (StageBuffers buffers = new StageBuffers()) {
            buffers.owned(BufferKey.output(0), ElementKind.DOUBLE, 4);
            // Owned, not a view, so there is nothing to bind and the barrier is never reached.
            assertThatThrownBy(
                            () ->
                                    buffers.bind(
                                            null,
                                            BufferKey.output(0),
                                            DeviceBinding.describe(
                                                    "cuDF", 0x7f00L, 32, 4, ElementKind.DOUBLE),
                                            true,
                                            awaited::incrementAndGet))
                    .isInstanceOf(IllegalStateException.class);
        }
        assertThat(awaited).hasValue(0);
    }

    private static void sleepALittle() {
        long until = System.nanoTime() + 200_000L;
        while (System.nanoTime() < until) {
            Thread.onSpinWait();
        }
    }
}
