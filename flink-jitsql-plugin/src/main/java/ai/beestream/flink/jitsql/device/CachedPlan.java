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
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;

/**
 * One compiled execution plan and the buffers its graph was wired to.
 *
 * <p>The two travel together because they were built together: the plan's kernels take the
 * registry's arrays as arguments, so a plan handed a different registry's buffers is a plan
 * computing over memory nobody filled.
 */
public final class CachedPlan implements AutoCloseable {

    private static final Logger LOG = LoggerFactory.getLogger(CachedPlan.class);

    private final TornadoExecutionPlan plan;
    private final StageBuffers buffers;
    private final PlanShape shape;
    private boolean closed;

    public CachedPlan(TornadoExecutionPlan plan, StageBuffers buffers, PlanShape shape) {
        this.plan = plan;
        this.buffers = buffers;
        this.shape = shape;
    }

    public TornadoExecutionPlan plan() {
        return plan;
    }

    public StageBuffers buffers() {
        return buffers;
    }

    public PlanShape shape() {
        return shape;
    }

    /** Whether this plan's device resources have already been released. */
    public boolean closed() {
        return closed;
    }

    /**
     * Releases the device resources this plan holds.
     *
     * <p>Tolerates a second call and never throws: eviction runs inside a map operation, and a
     * close that threw there would leave the cache holding a plan it believes it has dropped.
     */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        try {
            buffers.unbindAll(plan);
        } catch (RuntimeException e) {
            LOG.debug("could not unbind the views of the plan for {}", shape, e);
        }
        try {
            buffers.close();
        } catch (RuntimeException e) {
            LOG.debug("could not release the buffers of the plan for {}", shape, e);
        }
        try {
            plan.close();
        } catch (Exception e) {
            LOG.debug("could not close the execution plan for {}", shape, e);
        }
    }
}
