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
import uk.ac.manchester.tornado.api.memory.CUDADeviceViewBinding;
import uk.ac.manchester.tornado.api.types.arrays.TornadoNativeArray;

/**
 * A kernel argument that points at memory somebody else owns.
 *
 * <p>The view is created once, with the plan, and rebound before each execution. That order is the
 * whole economy of the thing: the view carries no length, so rebinding it does not change anything
 * the kernel was compiled against, and a plan compiled for one batch serves the next one at a
 * different address for nothing.
 *
 * <p>A view binds to one buffer at a time, which is why a plan holding views belongs to one thread.
 * Two subtasks sharing a plan would be two subtasks overwriting each other's bindings between the
 * bind and the launch -- a race that produces wrong answers rather than an error, and the reason
 * the execution-plan cache is per thread rather than shared.
 */
public final class BoundView {

    /** TornadoVM's "no particular device index", as the binding API spells it. */
    private static final int ANY_DEVICE = -1;

    private final TornadoNativeArray view;
    private final ElementKind kind;
    private final long capacityElements;

    private BoundView(TornadoNativeArray view, ElementKind kind, long capacityElements) {
        this.view = view;
        this.kind = kind;
        this.capacityElements = capacityElements;
    }

    /** Opens a view with room for {@code capacityElements}, bound to nothing yet. */
    public static BoundView open(ElementKind kind, long capacityElements) {
        if (capacityElements < 0) {
            throw new IllegalArgumentException(
                    "a view cannot hold " + capacityElements + " elements");
        }
        return new BoundView(kind.newView(capacityElements), kind, capacityElements);
    }

    /** The array to hand the task graph as a kernel argument. */
    public TornadoNativeArray argument() {
        return view;
    }

    public ElementKind kind() {
        return kind;
    }

    public long capacityElements() {
        return capacityElements;
    }

    /**
     * Points this view at {@code binding}.
     *
     * @param readOnly true for an input the kernel only reads, false for one it writes
     */
    public BoundView bind(TornadoExecutionPlan plan, DeviceBinding binding, boolean readOnly) {
        binding.requireKind(kind);
        if (binding.rows() > capacityElements) {
            throw new IllegalStateException(
                    "the buffer from "
                            + binding.owner()
                            + " holds "
                            + binding.rows()
                            + " elements, more than the "
                            + capacityElements
                            + " this view was opened for");
        }
        plan.bindCUDADeviceBuffer(
                view,
                binding.address(),
                binding.byteLength(),
                binding.rows(),
                ANY_DEVICE,
                readOnly);
        return this;
    }

    /** Releases the binding, leaving the view reusable. Safe to call when nothing is bound. */
    public void unbind(TornadoExecutionPlan plan) {
        if (isBound()) {
            plan.unbindCUDADeviceBuffer(view);
        }
    }

    public boolean isBound() {
        CUDADeviceViewBinding binding = binding();
        return binding != null && binding.isBound();
    }

    /** Where this view currently points. Only meaningful while {@link #isBound()}. */
    public long devicePointer() {
        return requireBinding().devicePointer();
    }

    public long deviceByteLength() {
        return requireBinding().deviceByteLength();
    }

    /** How many elements the current binding covers. */
    public long rows() {
        return requireBinding().logicalLength();
    }

    private CUDADeviceViewBinding binding() {
        return view.isCUDADeviceView() ? view.cudaDeviceViewBinding() : null;
    }

    private CUDADeviceViewBinding requireBinding() {
        CUDADeviceViewBinding binding = binding();
        if (binding == null || !binding.isBound()) {
            throw new IllegalStateException("this view is not bound to anything");
        }
        return binding;
    }

    @Override
    public String toString() {
        return "view[" + capacityElements + " x " + kind + (isBound() ? ", bound]" : ", free]");
    }
}
