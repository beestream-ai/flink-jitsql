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
package ai.beestream.flink.jitsql.gather;

import ai.beestream.flink.jitsql.device.ElementKind;

import org.apache.flink.table.data.ArrayData;
import org.apache.flink.table.data.GenericArrayData;
import org.apache.flink.table.data.RowData;
import org.apache.flink.table.data.binary.BinaryArrayData;

import uk.ac.manchester.tornado.api.types.arrays.DoubleArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.api.types.arrays.LongArray;
import uk.ac.manchester.tornado.api.types.arrays.TornadoNativeArray;

import java.lang.foreign.MemorySegment;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Stages an {@code ARRAY} column into the layout cuDF already uses for a {@code LIST}.
 *
 * <p>Offsets and one flattened run of elements, so row {@code r} occupies the child between {@code
 * offsets[r]} and {@code offsets[r + 1]}. Rows are free to differ in length and nothing is padded:
 * a column of a thousand rows averaging seven elements stages seven thousand values, not a thousand
 * times the longest row.
 *
 * <p>Padding would be the obvious alternative and is the wrong one twice over. It wastes whatever
 * the ragged tail costs -- on a feature column with a few long rows that can be most of the buffer
 * -- and it makes the staged column a different shape from the one a device-side read produces, so
 * the same kernel could not serve both paths.
 *
 * <p>This is the host path. A column that cuDF read on the device never comes through here at all;
 * it is already in this layout, which is the point of choosing this layout.
 */
public final class ArrayColumnGather {

    // Diagnostic, off unless -Dflink.accelerator.tornadovm.gatherTrace=true. Which staging
    // path the rows of a real query actually take is not something the unit tests can answer:
    // they construct the ArrayData themselves, and what a join hands this class is the whole
    // question. Counted rather than logged per row, because per row is the thing being measured.
    private static final boolean TRACE =
            Boolean.getBoolean("flink.accelerator.tornadovm.gatherTrace");

    private static final AtomicLong IN_PLACE = new AtomicLong();
    private static final AtomicLong VIA_HEAP = new AtomicLong();
    private static final AtomicLong ELEMENTWISE = new AtomicLong();
    private static final AtomicLong ELEMENTS = new AtomicLong();
    private static final Set<String> ARRIVED_AS = ConcurrentHashMap.newKeySet();

    static {
        if (TRACE) {
            Runtime.getRuntime()
                    .addShutdownHook(
                            new Thread(
                                    () ->
                                            System.err.printf(
                                                    "%nGATHER TRACE  in-place=%d via-heap=%d"
                                                            + " element-wise=%d  elements=%d"
                                                            + "  arrived as %s%n",
                                                    IN_PLACE.get(),
                                                    VIA_HEAP.get(),
                                                    ELEMENTWISE.get(),
                                                    ELEMENTS.get(),
                                                    ARRIVED_AS)));
        }
    }

    private final int field;
    private final ElementKind elementKind;
    private final IntArray offsets;
    private final TornadoNativeArray values;
    private int rows;
    private int written;

    /**
     * @param maxRows how many rows a batch carries
     * @param capacityElements how many elements the child can hold across every row of a batch
     */
    public ArrayColumnGather(
            int field, ElementKind elementKind, int maxRows, int capacityElements) {
        if (maxRows < 0 || capacityElements < 0) {
            throw new IllegalArgumentException(
                    "a batch cannot hold "
                            + maxRows
                            + " rows of "
                            + capacityElements
                            + " elements");
        }
        this.field = field;
        this.elementKind = elementKind;
        // An IntArray rather than an int[], because this buffer is a kernel argument: the
        // offsets go to the device unchanged, and copying them into one on every batch would be
        // the kind of per-row host work the whole path exists to remove.
        this.offsets = new IntArray(maxRows + 1);
        this.values = allocate(elementKind, capacityElements);
    }

    /** Stages one row's array. Rows must arrive in order, as the offsets are cumulative. */
    public void accept(RowData row, int position) {
        if (position != rows) {
            throw new IllegalStateException(
                    "rows must be staged in order: expected " + rows + ", got " + position);
        }
        if (row.isNullAt(field)) {
            // Admitted only for NOT NULL columns, so this is a planner bug rather than data.
            throw new IllegalStateException(
                    "row "
                            + position
                            + " has no array in field "
                            + field
                            + ", and this column was admitted as NOT NULL");
        }
        ArrayData array = row.getArray(field);
        int length = array.size();
        if (written + length > capacityElements()) {
            throw new IllegalStateException(
                    "row "
                            + position
                            + " needs "
                            + length
                            + " more elements than the "
                            + (capacityElements() - written)
                            + " left in a child sized for "
                            + capacityElements());
        }
        offsets.set(position, written);
        if (TRACE) {
            ARRIVED_AS.add(array.getClass().getSimpleName());
            ELEMENTS.addAndGet(length);
        }
        if (!copyInBulk(array, length, written)) {
            if (TRACE) {
                ELEMENTWISE.incrementAndGet();
            }
            for (int i = 0; i < length; i++) {
                if (array.isNullAt(i)) {
                    throw new IllegalStateException(
                            "element "
                                    + i
                                    + " of row "
                                    + position
                                    + " is null, and no kernel here"
                                    + " reads a validity mask");
                }
                copy(array, i, written + i);
            }
        }
        written += length;
        rows = position + 1;
        offsets.set(rows, written);
    }

    /** Resets for the next batch without reallocating anything. */
    public void reset() {
        rows = 0;
        written = 0;
        offsets.set(0, 0);
    }

    /** The offsets buffer a kernel reads: {@code rows + 1} cumulative entries. */
    public IntArray offsetsBuffer() {
        return offsets;
    }

    /** The offsets of the rows staged so far, copied out. For tests and diagnostics. */
    public int[] offsets() {
        int[] exact = new int[rows + 1];
        for (int i = 0; i <= rows; i++) {
            exact[i] = offsets.get(i);
        }
        return exact;
    }

    public TornadoNativeArray values() {
        return values;
    }

    public int rows() {
        return rows;
    }

    /** How many elements the staged rows hold between them. */
    public int elements() {
        return written;
    }

    public int capacityElements() {
        return values.getSize();
    }

    public ElementKind elementKind() {
        return elementKind;
    }

    /**
     * Stages one row's elements as a single copy rather than one virtual call each.
     *
     * <p>Both sides are already contiguous runs of the same primitive in the same byte order: a
     * {@link BinaryArrayData} holds its elements back to back after a fixed header, and a TornadoVM
     * array's {@link TornadoNativeArray#getSegment()} begins at element zero. So the copy is a
     * {@code memcpy} and nothing is converted on the way.
     *
     * <p>What this replaces was three virtual calls and two bounds checks a value -- about 3,400 of
     * them for one 1,128-element feature vector -- measured at roughly 5.3 ns an element, which was
     * four fifths of everything an offloaded projection spent. The device was never the slow part;
     * reconstructing its operand one {@code double} at a time was.
     *
     * @return false when this array has no contiguous run to copy, leaving the caller to take the
     *     element-at-a-time path and report which element was at fault
     */
    private boolean copyInBulk(ArrayData array, int length, int to) {
        if (length == 0) {
            return true;
        }
        MemorySegment source = sourceSegment(array, length);
        if (source == null) {
            return false;
        }
        int elementBytes = values.getElementSize();
        MemorySegment.copy(
                source,
                0L,
                values.getSegment(),
                (long) to * elementBytes,
                (long) length * elementBytes);
        return true;
    }

    /** This row's elements as one contiguous run, or null when there is no such run. */
    private MemorySegment sourceSegment(ArrayData array, int length) {
        if (array instanceof BinaryArrayData) {
            BinaryArrayData binary = (BinaryArrayData) array;
            if (binary.anyNull()) {
                // Refused either way, but the element path names the offending index.
                return null;
            }
            MemorySegment inPlace = withoutCopying(binary, length);
            return inPlace != null ? inPlace : viaHeapArray(array);
        }
        if (array instanceof GenericArrayData && ((GenericArrayData) array).isPrimitiveArray()) {
            // A primitive-backed GenericArrayData hands back its own array, so this copies once.
            return viaHeapArray(array);
        }
        return null;
    }

    /**
     * The binary row's own bytes, wrapped where they already lie.
     *
     * <p>Null whenever that is not possible: elements that straddle two segments, or a segment over
     * memory Flink declines to hand out. Both fall back to a copy rather than to the element path,
     * because a copy is still one call instead of a thousand.
     */
    private MemorySegment withoutCopying(BinaryArrayData binary, int length) {
        org.apache.flink.core.memory.MemorySegment[] segments = binary.getSegments();
        if (segments.length != 1) {
            return null;
        }
        int from = binary.getOffset() + BinaryArrayData.calculateHeaderInBytes(length);
        int bytes = length * values.getElementSize();
        if (from < 0 || bytes < 0 || from + bytes > segments[0].size()) {
            return null;
        }
        try {
            MemorySegment wrapped = MemorySegment.ofBuffer(segments[0].wrap(from, bytes));
            if (TRACE) {
                IN_PLACE.incrementAndGet();
            }
            return wrapped;
        } catch (UnsupportedOperationException notWrappable) {
            return null;
        }
    }

    /** A heap array of the right primitive, which every {@link ArrayData} can produce at once. */
    private MemorySegment viaHeapArray(ArrayData array) {
        if (TRACE) {
            VIA_HEAP.incrementAndGet();
        }
        switch (elementKind) {
            case INT:
                return MemorySegment.ofArray(array.toIntArray());
            case LONG:
                return MemorySegment.ofArray(array.toLongArray());
            case FLOAT:
                return MemorySegment.ofArray(array.toFloatArray());
            case DOUBLE:
                return MemorySegment.ofArray(array.toDoubleArray());
            default:
                return null;
        }
    }

    private void copy(ArrayData array, int from, int to) {
        switch (elementKind) {
            case INT:
                ((IntArray) values).set(to, array.getInt(from));
                break;
            case LONG:
                ((LongArray) values).set(to, array.getLong(from));
                break;
            case FLOAT:
                ((FloatArray) values).set(to, array.getFloat(from));
                break;
            case DOUBLE:
                ((DoubleArray) values).set(to, array.getDouble(from));
                break;
            default:
                throw new IllegalStateException("no array element reader for " + elementKind);
        }
    }

    private static TornadoNativeArray allocate(ElementKind kind, int elements) {
        switch (kind) {
            case INT:
                return new IntArray(elements);
            case LONG:
                return new LongArray(elements);
            case FLOAT:
                return new FloatArray(elements);
            case DOUBLE:
                return new DoubleArray(elements);
            default:
                throw new IllegalArgumentException(
                        kind + " is not an array element type a kernel reads");
        }
    }
}
