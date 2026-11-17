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

import java.util.Optional;

/**
 * An {@code ARRAY} column as cuDF holds it: offsets, and one flattened run of elements.
 *
 * <p>Row {@code r}'s elements are the child's entries between {@code offsets[r]} and {@code
 * offsets[r + 1]}. Rows are free to differ in length and nothing is padded, which is why this is
 * two buffers rather than a rectangle -- and why a kernel reading one has to be given the offsets
 * rather than a stride.
 *
 * <p>Nothing is gathered and nothing is copied. Both buffers are the owner's, exactly as they
 * already are.
 *
 * <h2>Why a sliced column is refused</h2>
 *
 * <p>A slice of a LIST keeps the parent's offsets, so its first offset is wherever the slice starts
 * rather than zero. A kernel indexing the child with those offsets reads the wrong rows -- and
 * reads them successfully, which is the problem. The check is a length test because a slice is the
 * only ordinary way for the offsets buffer to be shorter than {@code rows + 1} entries.
 */
public final class ListBinding {

    private final DeviceBinding offsets;
    private final DeviceBinding values;
    private final long rows;

    private ListBinding(DeviceBinding offsets, DeviceBinding values, long rows) {
        this.offsets = offsets;
        this.values = values;
        this.rows = rows;
    }

    /**
     * Describes a LIST column's two buffers.
     *
     * @param valuesAddress the child payload's address, or 0 when every row is empty
     * @param childElements how many elements the child holds in total, across all rows
     */
    public static ListBinding describe(
            String owner,
            long offsetsAddress,
            long offsetsByteLength,
            long valuesAddress,
            long valuesByteLength,
            long childElements,
            long rows,
            ElementKind elementKind) {
        long offsetsNeeded = (rows + 1) * ElementKind.INT.byteWidth();
        if (offsetsByteLength < offsetsNeeded) {
            throw new IllegalArgumentException(
                    "the offsets buffer from "
                            + owner
                            + " is "
                            + offsetsByteLength
                            + " bytes, less than the "
                            + offsetsNeeded
                            + " bytes "
                            + rows
                            + " rows need. A sliced LIST column is not supported: its offsets"
                            + " would not be indices into the child from zero.");
        }
        DeviceBinding offsets =
                DeviceBinding.describe(
                        owner + ".offsets",
                        offsetsAddress,
                        offsetsByteLength,
                        rows + 1,
                        ElementKind.INT);
        if (childElements == 0 || valuesAddress == 0L) {
            // Every row is empty. cuDF need not allocate a child payload at all for that, and it
            // is an ordinary batch rather than a broken one: a length-controlled loop over such a
            // row runs zero iterations, so no element is ever read. The caller binds a sentinel so
            // the kernel still receives a pointer it provably never dereferences.
            return new ListBinding(offsets, null, rows);
        }
        DeviceBinding values =
                DeviceBinding.describe(
                        owner + ".values",
                        valuesAddress,
                        valuesByteLength,
                        childElements,
                        elementKind);
        return new ListBinding(offsets, values, rows);
    }

    public DeviceBinding offsets() {
        return offsets;
    }

    /** Empty when every row of the column is empty and cuDF allocated no child payload. */
    public Optional<DeviceBinding> values() {
        return Optional.ofNullable(values);
    }

    public long rows() {
        return rows;
    }

    /** How many elements the child holds across every row. Zero when there is no payload. */
    public long childElements() {
        return values == null ? 0L : values.rows();
    }

    public boolean isEmpty() {
        return values == null;
    }

    @Override
    public String toString() {
        return "list[" + rows + " rows, " + childElements() + " elements]";
    }
}
