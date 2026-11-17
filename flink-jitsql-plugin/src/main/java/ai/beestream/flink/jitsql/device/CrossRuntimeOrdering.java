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

import java.util.concurrent.atomic.AtomicLong;

/**
 * Orders a kernel against whoever filled the buffers it reads, and charges for doing so.
 *
 * <h2>What the problem is, and what it is not here</h2>
 *
 * <p>Two runtimes writing the same device memory on different streams have to be ordered, or a
 * kernel reads bytes its producer has not finished writing. That is a real hazard and it is why an
 * integration where RAPIDS and TornadoVM issue independently has to wait for the whole device
 * between them.
 *
 * <p>Inside a region this provider does not have that hazard, and the reason is structural rather
 * than lucky: cuDF is reached through TornadoVM's {@code libraryTask}, so a cuDF stage and a
 * generated kernel are two tasks of <em>one</em> task graph and TornadoVM sequences them. The
 * barrier for that case is therefore {@link #IN_GRAPH}, which does nothing -- and does nothing
 * <em>deliberately</em>, which is a different thing from nothing being there.
 *
 * <p>The hazard returns the moment a buffer arrives from outside that graph, which is exactly what
 * {@link DeviceBinding} describes. Then somebody has to establish the order, and the only party who
 * can is the one that owns the allocation and knows when its writes landed. So this is a seam
 * rather than a call: a producer supplies a {@link DeviceBarrier}, and the kernel path waits on it.
 *
 * <h2>Why it is measured</h2>
 *
 * <p>Because the cheap version of this is to assume it is free, and it is not. A whole-device wait
 * is the coarse way to order two runtimes and it costs what it costs; a breakdown that leaves it
 * out attributes that time to the kernel. Every wait here is counted and timed, so it can appear as
 * its own line.
 */
public final class CrossRuntimeOrdering {

    /** What a producer supplies so a kernel can wait for its writes. */
    @FunctionalInterface
    public interface DeviceBarrier {
        /** Returns once this producer's writes to the device are visible to another runtime. */
        void await();
    }

    /**
     * The barrier for a buffer produced inside the same task graph: nothing to wait for.
     *
     * <p>Named rather than left implicit so that a stage which uses it has said so, and a stage
     * which should not be using it is visible as such.
     */
    public static final DeviceBarrier IN_GRAPH = () -> {};

    private static final AtomicLong WAITS = new AtomicLong();
    private static final AtomicLong NANOS = new AtomicLong();

    private CrossRuntimeOrdering() {}

    /**
     * Waits for {@code barrier}, counting and timing the wait.
     *
     * @return how long the wait took, in nanoseconds
     */
    public static long await(DeviceBarrier barrier) {
        if (barrier == null) {
            throw new IllegalArgumentException(
                    "a binding from outside the task graph needs a barrier; pass IN_GRAPH only"
                            + " when the producer really is a task of the same graph");
        }
        long started = System.nanoTime();
        barrier.await();
        long elapsed = System.nanoTime() - started;
        WAITS.incrementAndGet();
        NANOS.addAndGet(elapsed);
        return elapsed;
    }

    /** How many waits have happened. */
    public static long waits() {
        return WAITS.get();
    }

    /** How long they took in total, in nanoseconds. */
    public static long nanos() {
        return NANOS.get();
    }

    /** One line for a profile, so the cost is reported rather than assumed. */
    public static String line() {
        long waits = WAITS.get();
        if (waits == 0) {
            return "cross-runtime ordering: no waits (every buffer came from the same task graph)";
        }
        return String.format(
                "cross-runtime ordering: %d waits, %.3f ms total, %.3f ms each",
                waits, NANOS.get() / 1e6, NANOS.get() / 1e6 / waits);
    }

    /** Resets the counters. For a benchmark that reports per run rather than per JVM. */
    public static void reset() {
        WAITS.set(0);
        NANOS.set(0);
    }
}
