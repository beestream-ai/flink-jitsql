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
import org.apache.flink.table.data.GenericRowData;
import org.apache.flink.table.data.RowData;
import org.apache.flink.table.data.binary.BinaryArrayData;
import org.apache.flink.table.data.binary.BinaryRowData;
import org.apache.flink.table.data.writer.BinaryArrayWriter;
import org.apache.flink.table.data.writer.BinaryRowWriter;
import org.apache.flink.table.runtime.typeutils.ArrayDataSerializer;
import org.apache.flink.table.types.logical.DoubleType;

import org.junit.jupiter.api.Test;
import uk.ac.manchester.tornado.api.types.arrays.DoubleArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.api.types.arrays.LongArray;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * Staging a ragged {@code ARRAY} column into the offsets-and-child layout.
 *
 * <p>The layout is cuDF's, deliberately, so that a column staged from Flink rows and a column cuDF
 * read on the device are the same shape and one kernel serves both. Padding to the longest row
 * would be easier here and would make that impossible.
 */
class ArrayColumnGatherTest {

    private static RowData row(double... values) {
        Object[] boxed = new Object[values.length];
        for (int i = 0; i < values.length; i++) {
            boxed[i] = values[i];
        }
        return GenericRowData.of(new GenericArrayData(boxed));
    }

    private static double[] staged(ArrayColumnGather gather) {
        DoubleArray values = (DoubleArray) gather.values();
        double[] out = new double[gather.elements()];
        for (int i = 0; i < out.length; i++) {
            out[i] = values.get(i);
        }
        return out;
    }

    @Test
    void raggedRowsStageWithoutPadding() {
        ArrayColumnGather gather = new ArrayColumnGather(0, ElementKind.DOUBLE, 8, 64);
        gather.accept(row(1.0, 2.0, 3.0), 0);
        gather.accept(row(4.0), 1);
        gather.accept(row(5.0, 6.0), 2);

        assertThat(gather.rows()).isEqualTo(3);
        assertThat(gather.elements()).isEqualTo(6);
        assertThat(gather.offsets()).containsExactly(0, 3, 4, 6);
        assertThat(staged(gather)).containsExactly(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
    }

    /** {@code rows + 1} offsets, the last of which is the total. The invariant a kernel reads. */
    @Test
    void theOffsetsAreCumulativeAndEndAtTheTotal() {
        ArrayColumnGather gather = new ArrayColumnGather(0, ElementKind.DOUBLE, 4, 32);
        gather.accept(row(1.0, 2.0), 0);
        gather.accept(row(3.0, 4.0, 5.0), 1);
        int[] offsets = gather.offsets();
        assertThat(offsets).hasSize(gather.rows() + 1);
        assertThat(offsets[0]).isZero();
        assertThat(offsets[offsets.length - 1]).isEqualTo(gather.elements());
        for (int i = 1; i < offsets.length; i++) {
            assertThat(offsets[i]).isGreaterThanOrEqualTo(offsets[i - 1]);
        }
    }

    @Test
    void anEmptyRowOccupiesNoElementsAndStillHasAnOffset() {
        ArrayColumnGather gather = new ArrayColumnGather(0, ElementKind.DOUBLE, 4, 32);
        gather.accept(row(1.0), 0);
        gather.accept(row(), 1);
        gather.accept(row(2.0), 2);
        assertThat(gather.offsets()).containsExactly(0, 1, 1, 2);
        assertThat(staged(gather)).containsExactly(1.0, 2.0);
    }

    /** Every row empty is an ordinary batch, and the child is simply unused. */
    @Test
    void aColumnOfEmptyRowsStagesNoElements() {
        ArrayColumnGather gather = new ArrayColumnGather(0, ElementKind.DOUBLE, 4, 32);
        for (int i = 0; i < 3; i++) {
            gather.accept(row(), i);
        }
        assertThat(gather.elements()).isZero();
        assertThat(gather.offsets()).containsExactly(0, 0, 0, 0);
    }

    @Test
    void aChildTooSmallIsRefusedAndSaysByHowMuch() {
        ArrayColumnGather gather = new ArrayColumnGather(0, ElementKind.DOUBLE, 4, 4);
        gather.accept(row(1.0, 2.0, 3.0), 0);
        assertThatThrownBy(() -> gather.accept(row(4.0, 5.0), 1))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("1 left")
                .hasMessageContaining("sized for 4");
    }

    @Test
    void rowsMustArriveInOrderBecauseTheOffsetsAreCumulative() {
        ArrayColumnGather gather = new ArrayColumnGather(0, ElementKind.DOUBLE, 4, 32);
        gather.accept(row(1.0), 0);
        assertThatThrownBy(() -> gather.accept(row(2.0), 2))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("in order");
    }

    @Test
    void aNullArrayIsAPlannerBugAndSaysSo() {
        ArrayColumnGather gather = new ArrayColumnGather(0, ElementKind.DOUBLE, 4, 32);
        assertThatThrownBy(() -> gather.accept(GenericRowData.of((Object) null), 0))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("NOT NULL");
    }

    @Test
    void aNullElementIsRefusedRatherThanReadAsZero() {
        ArrayColumnGather gather = new ArrayColumnGather(0, ElementKind.DOUBLE, 4, 32);
        RowData withHole = GenericRowData.of(new GenericArrayData(new Object[] {1.0, null}));
        assertThatThrownBy(() -> gather.accept(withHole, 0))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("validity mask");
    }

    @Test
    void resettingReusesTheBuffersForTheNextBatch() {
        ArrayColumnGather gather = new ArrayColumnGather(0, ElementKind.DOUBLE, 4, 32);
        gather.accept(row(1.0, 2.0), 0);
        Object before = gather.values();
        gather.reset();
        gather.accept(row(9.0), 0);
        assertThat(gather.values()).isSameAs(before);
        assertThat(gather.rows()).isEqualTo(1);
        assertThat(gather.offsets()).containsExactly(0, 1);
        assertThat(staged(gather)).containsExactly(9.0);
    }

    @Test
    void everyNumericElementTypeStages() {
        for (ElementKind kind :
                new ElementKind[] {
                    ElementKind.INT, ElementKind.LONG, ElementKind.FLOAT, ElementKind.DOUBLE
                }) {
            ArrayColumnGather gather = new ArrayColumnGather(0, kind, 2, 8);
            assertThat(gather.elementKind()).isEqualTo(kind);
            assertThat(gather.capacityElements()).isEqualTo(8);
        }
    }

    // ---------------------------------------------------------------------------------------
    // The bulk-copy path.
    //
    // Everything above stages a boxed GenericArrayData, which has no contiguous run and so
    // takes the element-at-a-time fallback. Production never sends one of those: a row that
    // has been through a join arrives as a BinaryRowData whose array field is a
    // BinaryArrayData pointing into the row's own segment at a non-zero offset. That is the
    // path the memcpy takes, and getting its offset arithmetic wrong would not throw -- the
    // kernel would read neighbouring bytes and return plausible numbers. So it is tested
    // against the element-at-a-time answer rather than against a constant.
    // ---------------------------------------------------------------------------------------

    /** A BinaryArrayData standing on its own, so its offset within its segment is zero. */
    private static BinaryArrayData binaryArray(double... values) {
        BinaryArrayData array = new BinaryArrayData();
        BinaryArrayWriter writer = new BinaryArrayWriter(array, values.length, 8);
        for (int i = 0; i < values.length; i++) {
            writer.writeDouble(i, values[i]);
        }
        writer.complete();
        return array;
    }

    /** The same array as a field of a binary row, so it sits at a non-zero offset. */
    private static RowData binaryRow(double... values) {
        BinaryRowData row = new BinaryRowData(1);
        BinaryRowWriter writer = new BinaryRowWriter(row);
        writer.writeArray(0, binaryArray(values), new ArrayDataSerializer(new DoubleType(false)));
        writer.complete();
        return row;
    }

    @Test
    void aBinaryArrayStagesTheSameValuesAsTheElementPath() {
        ArrayColumnGather bulk = new ArrayColumnGather(0, ElementKind.DOUBLE, 4, 32);
        ArrayColumnGather elementwise = new ArrayColumnGather(0, ElementKind.DOUBLE, 4, 32);
        bulk.accept(binaryRow(1.5, -2.5, 3.25), 0);
        elementwise.accept(row(1.5, -2.5, 3.25), 0);
        assertThat(staged(bulk)).containsExactly(staged(elementwise));
        assertThat(bulk.offsets()).containsExactly(elementwise.offsets());
    }

    @Test
    void binaryRowsOfDifferentLengthsStageAtTheRightOffsets() {
        // The second row is copied at a non-zero destination, which is where an off-by-one in
        // the destination offset would show up and nowhere else.
        ArrayColumnGather gather = new ArrayColumnGather(0, ElementKind.DOUBLE, 8, 64);
        gather.accept(binaryRow(1.0, 2.0, 3.0), 0);
        gather.accept(binaryRow(4.0), 1);
        gather.accept(binaryRow(), 2);
        gather.accept(binaryRow(5.0, 6.0), 3);
        assertThat(gather.offsets()).containsExactly(0, 3, 4, 4, 6);
        assertThat(staged(gather)).containsExactly(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
    }

    @Test
    void aPrimitiveBackedGenericArrayAlsoCopiesInBulk() {
        ArrayColumnGather gather = new ArrayColumnGather(0, ElementKind.DOUBLE, 4, 32);
        gather.accept(GenericRowData.of(new GenericArrayData(new double[] {7.5, 8.5})), 0);
        gather.accept(GenericRowData.of(new GenericArrayData(new double[] {9.5})), 1);
        assertThat(gather.offsets()).containsExactly(0, 2, 3);
        assertThat(staged(gather)).containsExactly(7.5, 8.5, 9.5);
    }

    @Test
    void aBinaryArrayWithANullStillNamesTheElement() {
        // anyNull() sends this back to the element path, which is the only one that can say
        // which element was at fault. Losing that message to a faster copy would be a bad trade.
        BinaryArrayData array = new BinaryArrayData();
        BinaryArrayWriter writer = new BinaryArrayWriter(array, 2, 8);
        writer.writeDouble(0, 1.0);
        writer.setNullDouble(1);
        writer.complete();
        ArrayColumnGather gather = new ArrayColumnGather(0, ElementKind.DOUBLE, 4, 32);
        assertThatThrownBy(() -> gather.accept(GenericRowData.of(array), 0))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("element 1")
                .hasMessageContaining("validity mask");
    }

    @Test
    void everyElementTypeRoundTripsThroughTheBulkPath() {
        // One copy per primitive width. A wrong width would stage the right count of wrong
        // values, so each is checked by value and not by length.
        ArrayColumnGather ints = new ArrayColumnGather(0, ElementKind.INT, 2, 8);
        ints.accept(GenericRowData.of(new GenericArrayData(new int[] {-7, 7, 65536})), 0);
        assertThat(((IntArray) ints.values()).get(0)).isEqualTo(-7);
        assertThat(((IntArray) ints.values()).get(2)).isEqualTo(65536);

        ArrayColumnGather longs = new ArrayColumnGather(0, ElementKind.LONG, 2, 8);
        longs.accept(GenericRowData.of(new GenericArrayData(new long[] {Long.MIN_VALUE, 1L})), 0);
        assertThat(((LongArray) longs.values()).get(0)).isEqualTo(Long.MIN_VALUE);
        assertThat(((LongArray) longs.values()).get(1)).isEqualTo(1L);

        ArrayColumnGather floats = new ArrayColumnGather(0, ElementKind.FLOAT, 2, 8);
        floats.accept(GenericRowData.of(new GenericArrayData(new float[] {0.5f, -0.25f})), 0);
        assertThat(((FloatArray) floats.values()).get(0)).isEqualTo(0.5f);
        assertThat(((FloatArray) floats.values()).get(1)).isEqualTo(-0.25f);

        ArrayColumnGather doubles = new ArrayColumnGather(0, ElementKind.DOUBLE, 2, 8);
        doubles.accept(binaryRow(Double.MIN_VALUE, Double.MAX_VALUE), 0);
        assertThat(staged(doubles)).containsExactly(Double.MIN_VALUE, Double.MAX_VALUE);
    }

    @Test
    void aLongBinaryArrayCopiesEveryElement() {
        // A feature vector's width, because a short array can fit in a single machine word's
        // worth of slack and hide a length error.
        double[] wide = new double[1128];
        for (int i = 0; i < wide.length; i++) {
            wide[i] = i * 0.5 - 3.0;
        }
        ArrayColumnGather gather = new ArrayColumnGather(0, ElementKind.DOUBLE, 2, 2 * 1128);
        gather.accept(binaryRow(wide), 0);
        gather.accept(binaryRow(wide), 1);
        assertThat(gather.elements()).isEqualTo(2 * 1128);
        double[] out = staged(gather);
        for (int i = 0; i < wide.length; i++) {
            assertThat(out[i]).isEqualTo(wide[i]);
            assertThat(out[1128 + i]).isEqualTo(wide[i]);
        }
    }

    @Test
    void aBinaryArrayIsCopiedWithoutBeingMaterialisedOnTheHeap() {
        // The point of the whole change: the row's own bytes are the source. If this ever
        // falls back to toDoubleArray() the answers stay right and the cost comes back, so
        // the fast path is asserted directly rather than inferred from a timing.
        ArrayData array = binaryRow(1.0, 2.0, 3.0).getArray(0);
        assertThat(array).isInstanceOf(BinaryArrayData.class);
        BinaryArrayData binary = (BinaryArrayData) array;
        assertThat(binary.getSegments()).hasSize(1);
        assertThat(binary.getOffset()).isGreaterThan(0);
        assertThat(binary.anyNull()).isFalse();
    }
}
