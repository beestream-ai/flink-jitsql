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

import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.DoubleArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.api.types.arrays.LongArray;
import uk.ac.manchester.tornado.api.types.arrays.TornadoNativeArray;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Every buffer an assembled region uses, wired by key instead of by field name.
 *
 * <p>This is the piece the six hand-written regions do not have and cannot have. Each of them
 * declares its intermediates as fields and wires one stage to the next by writing the field name in
 * both places, which is why a seventh shape needs a seventh class. A stage here asks for a buffer
 * by what it is for, and the stage after it asks for the same key; neither names the other and
 * neither is written for a particular query.
 *
 * <h2>Owned buffers and views</h2>
 *
 * <p>Two kinds live here, and the difference is who allocated.
 *
 * <ul>
 *   <li>An <b>owned</b> buffer is an ordinary TornadoVM array. Intermediates are these, because
 *       TornadoVM allocates and frees them with the plan and nothing else has a claim on them.
 *   <li>A <b>view</b> points at memory somebody else owns and is rebound per batch. Anything
 *       arriving from outside is one of these.
 * </ul>
 *
 * <p>Chaining one stage's output into the next works for both, but by different routes: an owned
 * buffer is simply handed over as the same object, while a view can report where it points and so
 * can be passed on by address. TornadoVM does not publish the device address of an array it
 * manages, so a buffer that has to be handed to a <em>separate</em> plan has to have entered as a
 * view. Within one task graph, which is what a region is, owned buffers chain freely.
 *
 * <h2>Lifetime</h2>
 *
 * <p>Closing releases the views' bindings. It does not free owned buffers, because TornadoVM owns
 * those and freeing them here would be a double free rather than a tidy-up.
 */
public final class StageBuffers implements AutoCloseable {

    private final Map<BufferKey, TornadoNativeArray> owned = new LinkedHashMap<>();
    private final Map<BufferKey, BoundView> views = new LinkedHashMap<>();

    /**
     * An intermediate this region allocates and TornadoVM frees.
     *
     * <p>Idempotent on purpose: an assembler that walks a subtree twice -- once to size the graph
     * and once to build it -- must get the same buffer both times, or the second walk silently
     * builds a graph wired to arrays the first walk's stages do not hold.
     */
    public TornadoNativeArray owned(BufferKey key, ElementKind kind, int elements) {
        TornadoNativeArray existing = owned.get(key);
        if (existing != null) {
            ElementKind existingKind = ElementKind.of(existing);
            if (existingKind != kind) {
                throw new IllegalStateException(
                        key + " was already allocated as " + existingKind + ", not " + kind);
            }
            if (existing.getSize() < elements) {
                throw new IllegalStateException(
                        key
                                + " was already allocated for "
                                + existing.getSize()
                                + " elements, fewer than the "
                                + elements
                                + " now asked for");
            }
            return existing;
        }
        if (views.containsKey(key)) {
            throw new IllegalStateException(key + " is already a view and cannot also be owned");
        }
        TornadoNativeArray array = allocate(kind, elements);
        owned.put(key, array);
        return array;
    }

    /** A kernel argument pointing at memory somebody else owns, rebound per batch. */
    public BoundView view(BufferKey key, ElementKind kind, long capacityElements) {
        BoundView existing = views.get(key);
        if (existing != null) {
            if (existing.kind() != kind) {
                throw new IllegalStateException(
                        key + " is already a view of " + existing.kind() + ", not " + kind);
            }
            return existing;
        }
        if (owned.containsKey(key)) {
            throw new IllegalStateException(key + " is already owned and cannot also be a view");
        }
        BoundView view = BoundView.open(kind, capacityElements);
        views.put(key, view);
        return view;
    }

    /**
     * The argument a stage passes to the task graph for {@code key}, whichever kind it is.
     *
     * <p>This is the call that makes the assembler generic: a stage wires its input by key and
     * never learns whether the buffer behind it was allocated here or handed in from outside.
     */
    public TornadoNativeArray argument(BufferKey key) {
        TornadoNativeArray array = owned.get(key);
        if (array != null) {
            return array;
        }
        BoundView view = views.get(key);
        if (view != null) {
            return view.argument();
        }
        throw new IllegalStateException(
                "nothing is registered for " + key + "; registered: " + registered());
    }

    /** Points a view at a foreign allocation. Call once per batch, before executing the plan. */
    public void bind(
            TornadoExecutionPlan plan, BufferKey key, DeviceBinding binding, boolean readOnly) {
        bind(plan, key, binding, readOnly, CrossRuntimeOrdering.IN_GRAPH);
    }

    /**
     * Points a view at a foreign allocation, after waiting for its producer.
     *
     * <p>The barrier is a parameter rather than something this works out, because only the owner of
     * the allocation knows when its writes landed. Passing {@link CrossRuntimeOrdering#IN_GRAPH} is
     * the claim that the producer is a task of the same graph and TornadoVM has already ordered it
     * -- a claim worth making explicitly, since the failure when it is wrong is a kernel reading
     * half-written memory rather than an error.
     */
    public void bind(
            TornadoExecutionPlan plan,
            BufferKey key,
            DeviceBinding binding,
            boolean readOnly,
            CrossRuntimeOrdering.DeviceBarrier barrier) {
        BoundView view = views.get(key);
        if (view == null) {
            throw new IllegalStateException(
                    key + " is not a view, so there is nothing to bind it to");
        }
        CrossRuntimeOrdering.await(barrier);
        view.bind(plan, binding, readOnly);
    }

    /**
     * Describes a view's current buffer so a later stage can read it in place.
     *
     * <p>No copy and no second allocation: the next stage binds its own view to the same address.
     */
    public DeviceBinding handOn(BufferKey key) {
        BoundView view = views.get(key);
        if (view == null) {
            throw new IllegalStateException(
                    key
                            + " is owned by TornadoVM, which does not publish a device address."
                            + " A buffer that has to reach a separate plan must enter as a view.");
        }
        return DeviceBinding.ofBoundView(key.toString(), view);
    }

    /** Every key registered, in the order the assembler asked for them. For errors and tests. */
    public List<BufferKey> registered() {
        List<BufferKey> keys = new ArrayList<>(owned.keySet());
        keys.addAll(views.keySet());
        return keys;
    }

    public void unbindAll(TornadoExecutionPlan plan) {
        for (BoundView view : views.values()) {
            view.unbind(plan);
        }
    }

    @Override
    public void close() {
        // Only the bindings are ours to drop. The owned arrays belong to TornadoVM and go with the
        // plan; releasing them here would be a double free, which is a crash rather than a leak.
        views.clear();
        owned.clear();
    }

    private static TornadoNativeArray allocate(ElementKind kind, int elements) {
        if (elements < 0) {
            throw new IllegalArgumentException("cannot allocate " + elements + " elements");
        }
        switch (kind) {
            case BYTE:
                return new ByteArray(elements);
            case INT:
                return new IntArray(elements);
            case LONG:
                return new LongArray(elements);
            case FLOAT:
                return new FloatArray(elements);
            case DOUBLE:
                return new DoubleArray(elements);
            default:
                throw new IllegalStateException("no allocator for " + kind);
        }
    }
}
