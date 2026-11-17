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

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Supplier;

/**
 * Keeps the work that must not happen per batch from happening per batch.
 *
 * <h2>Why a second cache, when kernels are already cached</h2>
 *
 * <p>{@code GeneratedKernel} caches the generated Java <em>class</em> for the whole JVM, keyed on
 * its source and reference-counted. That is the right scope for it: the bytecode is immutable and
 * defining it twice would make TornadoVM compile two identical kernels under different names.
 *
 * <p>It is not the expensive half. What costs is turning a task graph into device code, and that
 * has never been cached beyond the lifetime of one operator. The numbers are not marginal: driver
 * compilation alone is 107 ms median and 1186 ms at worst, plan reuse across files was measured at
 * 34x, and at 960 operations per row -- inside the shipped correctness ceiling -- compilation is
 * 16.2 s of a 17.8 s job. Compile time is between quadratic and cubic in expression size, so this
 * gets worse exactly where the offload is otherwise most worth having.
 *
 * <h2>Why per thread</h2>
 *
 * <p>Because a plan holds bound device views, and a view can be bound to one buffer at a time.
 * Several subtasks run in one TaskManager JVM; sharing a plan between them would be two threads
 * overwriting each other's bindings between the bind and the launch. That is a race that produces
 * wrong answers rather than an error, so the saving is not a saving.
 *
 * <p>This is also the one thing worth copying exactly rather than approximately from the Spark
 * integration, where the same structure is arrived at for the same reason.
 *
 * <h2>Bounds</h2>
 *
 * <p>Both levels are bounded and eviction closes what it drops. A plan owns a task graph, compiled
 * native code and device memory, so an executor thread meeting an unusual number of distinct
 * kernels or batch sizes would otherwise accumulate device resources for the life of the JVM. A
 * query normally has one batch size and one smaller remainder, so the per-kernel map stays tiny.
 */
public final class ExecutionPlanCache {

    private static final Logger LOG = LoggerFactory.getLogger(ExecutionPlanCache.class);

    /** Row counts one thread keeps plans for, per kernel, before evicting the least recent. */
    static final int MAX_PLANS_PER_KERNEL = 8;

    /** Distinct kernels one thread keeps plans for before evicting the least recent. */
    static final int MAX_KERNELS_PER_THREAD = 32;

    private static final AtomicLong HITS = new AtomicLong();
    private static final AtomicLong MISSES = new AtomicLong();
    private static final AtomicLong EVICTIONS = new AtomicLong();

    private static final ThreadLocal<Map<String, Map<PlanShape, CachedPlan>>> PLANS =
            ThreadLocal.withInitial(ExecutionPlanCache::newKernelMap);

    private ExecutionPlanCache() {}

    /**
     * A plan for this kernel at this shape, built by {@code build} only if there is not one
     * already.
     *
     * @param kernelKey what makes one kernel a different kernel -- the generated source, or a
     *     digest of it, and never a name. Two kernels sharing a key compute each other's
     *     arithmetic, which is the one failure mode here that is silent.
     */
    public static CachedPlan get(String kernelKey, PlanShape shape, Supplier<CachedPlan> build) {
        Map<String, Map<PlanShape, CachedPlan>> kernels = PLANS.get();
        Map<PlanShape, CachedPlan> byShape =
                kernels.computeIfAbsent(kernelKey, key -> newShapeMap());
        CachedPlan hit = byShape.get(shape);
        if (hit != null) {
            HITS.incrementAndGet();
            return hit;
        }
        MISSES.incrementAndGet();
        CachedPlan fresh = build.get();
        if (fresh == null) {
            throw new IllegalStateException("no plan was built for " + kernelKey + " at " + shape);
        }
        byShape.put(shape, fresh);
        return fresh;
    }

    /** Drops and closes everything this thread holds. Call when a subtask finishes. */
    public static void releaseThread() {
        Map<String, Map<PlanShape, CachedPlan>> kernels = PLANS.get();
        for (Map<PlanShape, CachedPlan> byShape : kernels.values()) {
            byShape.values().forEach(CachedPlan::close);
        }
        kernels.clear();
        PLANS.remove();
    }

    /** Hits, misses and evictions since the JVM started. For tests and metrics, not for logic. */
    public static long hits() {
        return HITS.get();
    }

    public static long misses() {
        return MISSES.get();
    }

    public static long evictions() {
        return EVICTIONS.get();
    }

    private static Map<String, Map<PlanShape, CachedPlan>> newKernelMap() {
        return new LinkedHashMap<>(8, 0.75f, true) {
            @Override
            protected boolean removeEldestEntry(
                    Map.Entry<String, Map<PlanShape, CachedPlan>> eldest) {
                if (size() > MAX_KERNELS_PER_THREAD) {
                    eldest.getValue().values().forEach(CachedPlan::close);
                    EVICTIONS.addAndGet(eldest.getValue().size());
                    LOG.debug("evicting every plan for kernel {}", eldest.getKey());
                    return true;
                }
                return false;
            }
        };
    }

    private static Map<PlanShape, CachedPlan> newShapeMap() {
        return new LinkedHashMap<>(4, 0.75f, true) {
            @Override
            protected boolean removeEldestEntry(Map.Entry<PlanShape, CachedPlan> eldest) {
                if (size() > MAX_PLANS_PER_KERNEL) {
                    eldest.getValue().close();
                    EVICTIONS.incrementAndGet();
                    LOG.debug("evicting the plan for {}", eldest.getKey());
                    return true;
                }
                return false;
            }
        };
    }
}
