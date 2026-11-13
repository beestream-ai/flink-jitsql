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
package ai.beestream.flink.jitsql.operator;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import uk.ac.manchester.tornado.api.TornadoBackend;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.common.TornadoDevice;
import uk.ac.manchester.tornado.api.enums.TornadoVMBackendType;
import uk.ac.manchester.tornado.api.runtime.TornadoRuntimeProvider;

/**
 * Which GPU a subtask's kernels run on.
 *
 * <h2>The problem this exists for</h2>
 *
 * <p>Left alone, TornadoVM runs every execution plan on its default device: device 0 of its first
 * backend. A TaskManager with four slots and four cards therefore puts all four subtasks on card 0
 * and leaves three idle, and nothing says so — the query is correct, merely slower than the
 * hardware allows, which is the hardest kind of waste to notice.
 *
 * <p>This maps a subtask to a device and binds it for the duration of that subtask's thread.
 *
 * <h2>Why an unavailable ordinal is refused rather than defaulted</h2>
 *
 * <p>Falling back to the default device when the requested one is absent would be the friendly
 * thing to do and is the wrong thing to do. A device pointer is only meaningful on the device that
 * produced it, so a plan that was prepared for one card and then silently run on another does not
 * fail cleanly: it produces a wrong answer, or a fault a long way from its cause. The same
 * reasoning as {@code spark-jitudf}'s {@code DeviceAffinity}, which refuses for the same reason.
 *
 * <p>Here there is no second library choosing the device — TornadoVM allocates every buffer this
 * provider uses — so the mismatch cannot arise from disagreeing with cuDF the way it can in Spark.
 * What can arise is a configured ordinal that the backend does not have, and that is refused.
 *
 * <h2>Scope, and what it is not</h2>
 *
 * <p>This is a <em>deployment</em> concern: it binds the cluster operator's hardware to the slots
 * the cluster operator configured. Nothing about it is visible to the query author, who still
 * writes ordinary SQL — so it does not touch the transparency invariant.
 *
 * <p>On a single-GPU machine it changes nothing: every subtask resolves to device 0, which is where
 * they all ran anyway. The value is on a host with more than one card, and that case is <b>not
 * measured here</b> — this machine has one. It is correctness and headroom, not a speed-up with a
 * number behind it.
 */
public final class DeviceAffinity {

    private static final Logger LOG = LoggerFactory.getLogger(DeviceAffinity.class);

    /**
     * The device this subtask's plans run on.
     *
     * <p>Per thread, because Flink runs each subtask on its own task thread and two subtasks in one
     * TaskManager may legitimately be bound to different cards. A static field would make the last
     * {@code open()} win for everyone.
     */
    private static final ThreadLocal<TornadoDevice> BOUND = new ThreadLocal<>();

    private DeviceAffinity() {}

    /**
     * Binds this thread to the device for {@code subtaskIndex}, if a CUDA backend is present.
     *
     * <p>Round robin over the backend's devices. Not a scheduler: it does not know what the other
     * subtasks are doing and makes no attempt to balance by load. It spreads subtasks over the
     * cards that exist, which is the whole of the claim.
     *
     * <p>Absence of a CUDA backend is not an error. It means this TaskManager has no device for
     * this provider, the plans fall back to TornadoVM's own default, and whatever ran before this
     * existed runs now.
     */
    public static void bindSubtask(int subtaskIndex) {
        try {
            var runtime = TornadoRuntimeProvider.getTornadoRuntime();
            for (int i = 0; i < runtime.getNumBackends(); i++) {
                TornadoBackend backend = runtime.getBackend(i);
                if (backend.getBackendType() != TornadoVMBackendType.CUDA) {
                    continue;
                }
                int count = backend.getNumDevices();
                if (count <= 0) {
                    LOG.debug("the CUDA backend reports no devices; leaving the default in place");
                    return;
                }
                int ordinal = Math.floorMod(subtaskIndex, count);
                BOUND.set(backend.getDevice(ordinal));
                LOG.info("subtask {} bound to CUDA device {} of {}", subtaskIndex, ordinal, count);
                return;
            }
            LOG.debug("no CUDA backend on this TaskManager; leaving the default device in place");
        } catch (RuntimeException | LinkageError e) {
            // Asking TornadoVM for its devices can fail on a machine where it is not usable at
            // all. That is the same condition as having no device, and the provider already
            // declines in that case, so it must not become a different and louder failure here.
            LOG.debug(
                    "could not resolve a device for subtask {}; using the default",
                    subtaskIndex,
                    e);
        }
    }

    /**
     * The device for an explicitly named CUDA ordinal, or a refusal.
     *
     * @throws IllegalStateException if there is no CUDA backend, or it does not have that ordinal
     */
    public static TornadoDevice forCudaOrdinal(int ordinal) {
        var runtime = TornadoRuntimeProvider.getTornadoRuntime();
        for (int i = 0; i < runtime.getNumBackends(); i++) {
            TornadoBackend backend = runtime.getBackend(i);
            if (backend.getBackendType() != TornadoVMBackendType.CUDA) {
                continue;
            }
            int count = backend.getNumDevices();
            if (ordinal < 0 || ordinal >= count) {
                throw new IllegalStateException(
                        "CUDA device "
                                + ordinal
                                + " was asked for, but this TornadoVM's CUDA backend has "
                                + count
                                + " device(s); a plan prepared for one device cannot be run on"
                                + " another");
            }
            return backend.getDevice(ordinal);
        }
        throw new IllegalStateException(
                "TornadoVM has no CUDA backend on this classpath, so no CUDA device can be bound");
    }

    /** Applies this thread's binding to {@code plan}, or returns it untouched when unbound. */
    public static TornadoExecutionPlan apply(TornadoExecutionPlan plan) {
        TornadoDevice device = BOUND.get();
        return device == null ? plan : plan.withDevice(device);
    }

    /** Whether this thread is bound, for tests and for logging. */
    public static boolean isBound() {
        return BOUND.get() != null;
    }

    /** Releases the binding. Called from an operator's {@code close()}; task threads are pooled. */
    public static void clear() {
        BOUND.remove();
    }
}
