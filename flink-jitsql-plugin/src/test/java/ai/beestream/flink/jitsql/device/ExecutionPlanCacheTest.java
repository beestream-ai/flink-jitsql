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

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * What the cache keeps, what it separates and what it closes.
 *
 * <p>The plans here carry no {@code TornadoExecutionPlan}, because building one needs a device and
 * none of what is under test does. What is under test is the bookkeeping -- which lookups build and
 * which do not, which keys are the same key, and whether eviction releases what it drops -- and all
 * of that is exactly as wrong or right without a GPU attached.
 */
class ExecutionPlanCacheTest {

    @AfterEach
    void tidy() {
        ExecutionPlanCache.releaseThread();
    }

    private static CachedPlan plan(PlanShape shape) {
        return new CachedPlan(null, new StageBuffers(), shape);
    }

    @Test
    void aSecondQueryOfTheSameShapeBuildsNothing() {
        AtomicInteger builds = new AtomicInteger();
        PlanShape shape = PlanShape.ofRows(262_144);
        CachedPlan first =
                ExecutionPlanCache.get(
                        "kernel-a",
                        shape,
                        () -> {
                            builds.incrementAndGet();
                            return plan(shape);
                        });
        CachedPlan second =
                ExecutionPlanCache.get(
                        "kernel-a",
                        shape,
                        () -> {
                            builds.incrementAndGet();
                            return plan(shape);
                        });
        assertThat(builds).hasValue(1);
        assertThat(second).isSameAs(first);
    }

    /**
     * The short final batch is a different plan, and that is not an inefficiency to remove.
     *
     * <p>Every cuDF primitive fixes its row count when the graph is built, so a plan compiled for a
     * full batch cannot process a smaller one. A cache that ignored the row count would hand it
     * back anyway and the query would quietly reduce over the wrong number of rows.
     */
    @Test
    void adifferentRowCountIsADifferentPlan() {
        AtomicInteger builds = new AtomicInteger();
        for (int rows : new int[] {262_144, 1_013, 262_144, 1_013}) {
            PlanShape shape = PlanShape.ofRows(rows);
            ExecutionPlanCache.get(
                    "kernel-a",
                    shape,
                    () -> {
                        builds.incrementAndGet();
                        return plan(shape);
                    });
        }
        assertThat(builds).hasValue(2);
    }

    @Test
    void adifferentDeviceIsADifferentPlan() {
        AtomicInteger builds = new AtomicInteger();
        for (int ordinal : new int[] {0, 1, 0, 1}) {
            PlanShape shape = new PlanShape(1024, 0L, ordinal);
            ExecutionPlanCache.get(
                    "kernel-a",
                    shape,
                    () -> {
                        builds.incrementAndGet();
                        return plan(shape);
                    });
        }
        assertThat(builds).hasValue(2);
    }

    @Test
    void adifferentKernelIsADifferentPlan() {
        AtomicInteger builds = new AtomicInteger();
        PlanShape shape = PlanShape.ofRows(1024);
        for (String kernel : new String[] {"a", "b", "a", "b"}) {
            ExecutionPlanCache.get(
                    kernel,
                    shape,
                    () -> {
                        builds.incrementAndGet();
                        return plan(shape);
                    });
        }
        assertThat(builds).hasValue(2);
    }

    /**
     * Two subtasks in one JVM get two plans, and this is the test that matters most.
     *
     * <p>A plan holds bound device views and a view binds to one buffer at a time. If this ever
     * returned one plan to both threads, the failure would not be an exception -- it would be one
     * subtask launching a kernel over the other subtask's buffer, intermittently, under load.
     */
    @Test
    void twoThreadsDoNotShareAPlan() throws Exception {
        AtomicInteger builds = new AtomicInteger();
        List<CachedPlan> seen = new ArrayList<>();
        PlanShape shape = PlanShape.ofRows(4096);
        Runnable task =
                () -> {
                    CachedPlan got =
                            ExecutionPlanCache.get(
                                    "kernel-a",
                                    shape,
                                    () -> {
                                        builds.incrementAndGet();
                                        return plan(shape);
                                    });
                    synchronized (seen) {
                        seen.add(got);
                    }
                    ExecutionPlanCache.releaseThread();
                };
        Thread one = new Thread(task, "subtask-1");
        Thread two = new Thread(task, "subtask-2");
        one.start();
        two.start();
        one.join();
        two.join();
        assertThat(builds).hasValue(2);
        assertThat(seen).hasSize(2);
        assertThat(seen.get(0)).isNotSameAs(seen.get(1));
    }

    @Test
    void evictionClosesTheplanItDrops() {
        List<CachedPlan> built = new ArrayList<>();
        for (int i = 0; i <= ExecutionPlanCache.MAX_PLANS_PER_KERNEL; i++) {
            PlanShape shape = PlanShape.ofRows(i);
            built.add(
                    ExecutionPlanCache.get(
                            "kernel-a",
                            shape,
                            () -> {
                                CachedPlan fresh = plan(shape);
                                return fresh;
                            }));
        }
        // One more than the bound, so the least recently used one is gone -- and released, not
        // merely forgotten. A plan owns a task graph, compiled code and device memory.
        assertThat(built.get(0).closed()).isTrue();
        assertThat(built.get(built.size() - 1).closed()).isFalse();
    }

    @Test
    void evictingAKernelClosesEveryPlanItHeld() {
        List<CachedPlan> firstKernel = new ArrayList<>();
        for (int i = 0; i < 3; i++) {
            PlanShape shape = PlanShape.ofRows(i);
            firstKernel.add(ExecutionPlanCache.get("kernel-0", shape, () -> plan(shape)));
        }
        for (int k = 1; k <= ExecutionPlanCache.MAX_KERNELS_PER_THREAD; k++) {
            PlanShape shape = PlanShape.ofRows(0);
            ExecutionPlanCache.get("kernel-" + k, shape, () -> plan(shape));
        }
        assertThat(firstKernel).allSatisfy(p -> assertThat(p.closed()).isTrue());
    }

    @Test
    void releasingAThreadClosesEverythingItHeld() {
        List<CachedPlan> built = new ArrayList<>();
        for (int i = 0; i < 4; i++) {
            PlanShape shape = PlanShape.ofRows(i);
            built.add(ExecutionPlanCache.get("kernel-a", shape, () -> plan(shape)));
        }
        ExecutionPlanCache.releaseThread();
        assertThat(built).allSatisfy(p -> assertThat(p.closed()).isTrue());
    }

    @Test
    void closingTwiceIsHarmless() {
        CachedPlan plan = plan(PlanShape.ofRows(8));
        plan.close();
        plan.close();
        assertThat(plan.closed()).isTrue();
    }
}
