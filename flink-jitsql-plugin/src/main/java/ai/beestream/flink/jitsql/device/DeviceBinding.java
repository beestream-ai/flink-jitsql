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

/**
 * One device allocation this code did not make, described well enough to bind a kernel to it.
 *
 * <p>Nothing is copied. An allocation of fixed-width elements already has the layout a kernel
 * reads, so what crosses is an address and a length and the kernel reads the owner's bytes where
 * the owner put them. The allocation belongs to whoever made it: nothing here frees one, and the
 * {@link #address} is only valid for as long as that owner keeps it alive.
 *
 * <h2>Ordering, and why there is no synchronisation call here</h2>
 *
 * <p>Within a region there is nothing to order. This provider reaches cuDF through TornadoVM's
 * {@code libraryTask}, so a cuDF stage and a generated kernel are two tasks of one task graph and
 * TornadoVM sequences them. That is a real difference from an integration where cuDF and TornadoVM
 * are separate runtimes issuing on separate streams, which has to wait for the whole device between
 * the two and pay for it.
 *
 * <p>What that leaves is a contract rather than a call, and it belongs to whoever hands a binding
 * over: <b>the owner's writes to this allocation must already be visible to the device when {@code
 * describe} is called.</b> Nothing here can check it -- an address carries no completion state --
 * so a producer outside the task graph has to establish the order itself, and a producer inside one
 * gets it from TornadoVM for free.
 *
 * <h2>Why validate at bind time at all</h2>
 *
 * <p>Because the alternative is a wrong answer rather than an error. A view bound to a buffer of
 * the wrong element type reinterprets its bytes and the kernel computes something plausible from
 * them; a view bound to a buffer shorter than the rows need reads past the end, which on a device
 * is unmapped memory on a good day and another allocation on a bad one. Every check below turns one
 * of those into a message naming the actual mismatch.
 */
public final class DeviceBinding {

    private final long address;
    private final long byteLength;
    private final long rows;
    private final ElementKind kind;
    private final String owner;

    private DeviceBinding(
            long address, long byteLength, long rows, ElementKind kind, String owner) {
        this.address = address;
        this.byteLength = byteLength;
        this.rows = rows;
        this.kind = kind;
        this.owner = owner;
    }

    /**
     * Describes a foreign allocation, refusing anything a kernel could misread.
     *
     * @param owner what holds the allocation, named only so a refusal can say so
     */
    public static DeviceBinding describe(
            String owner, long address, long byteLength, long rows, ElementKind kind) {
        if (kind == null) {
            throw new IllegalArgumentException(
                    "the buffer from " + owner + " has no element kind to bind against");
        }
        if (address == 0L) {
            // The Spark adapter meets this as a cuDF LIST whose every row is empty, and answers it
            // with an owned sentinel allocation so the kernel still gets a pointer it provably
            // never reads through. Nothing here allocates, so the honest answer is to refuse and
            // let the caller decide whether a sentinel is warranted.
            throw new IllegalArgumentException(
                    "the buffer from " + owner + " has no address, so there is nothing to bind");
        }
        if (rows < 0) {
            throw new IllegalArgumentException(
                    "the buffer from " + owner + " was described as holding " + rows + " rows");
        }
        long needed = rows * kind.byteWidth();
        if (byteLength < needed) {
            throw new IllegalArgumentException(
                    "the buffer from "
                            + owner
                            + " is "
                            + byteLength
                            + " bytes, less than the "
                            + needed
                            + " bytes "
                            + rows
                            + " rows of "
                            + kind
                            + " need");
        }
        return new DeviceBinding(address, byteLength, rows, kind, owner);
    }

    /**
     * Describes the allocation behind a view this code already bound.
     *
     * <p>This is how one stage's output becomes the next stage's input without a copy and without
     * either stage knowing about the other: a bound view can say where it points, so the buffer
     * outlives the plan that wrote it. It only works for views -- TornadoVM does not publish the
     * device address of an array it manages itself, so an intermediate that has to be chained has
     * to enter as a view in the first place.
     */
    public static DeviceBinding ofBoundView(String owner, BoundView view) {
        if (!view.isBound()) {
            throw new IllegalStateException(
                    "the view from " + owner + " is not bound, so it has no address to pass on");
        }
        return new DeviceBinding(
                view.devicePointer(), view.deviceByteLength(), view.rows(), view.kind(), owner);
    }

    public long address() {
        return address;
    }

    public long byteLength() {
        return byteLength;
    }

    public long rows() {
        return rows;
    }

    public ElementKind kind() {
        return kind;
    }

    public String owner() {
        return owner;
    }

    /** Refuses a binding whose element type is not the one a kernel was compiled for. */
    public DeviceBinding requireKind(ElementKind expected) {
        if (kind != expected) {
            throw new IllegalStateException(
                    "the buffer from "
                            + owner
                            + " holds "
                            + kind
                            + " but the kernel was compiled for "
                            + expected);
        }
        return this;
    }

    @Override
    public String toString() {
        return owner
                + "["
                + rows
                + " x "
                + kind
                + " @ 0x"
                + Long.toHexString(address)
                + ", "
                + byteLength
                + " bytes]";
    }
}
