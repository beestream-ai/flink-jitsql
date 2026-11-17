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

import org.apache.flink.table.types.logical.ArrayType;
import org.apache.flink.table.types.logical.BigIntType;
import org.apache.flink.table.types.logical.DecimalType;
import org.apache.flink.table.types.logical.DoubleType;
import org.apache.flink.table.types.logical.IntType;
import org.apache.flink.table.types.logical.VarCharType;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/** {@code ARRAY} columns as cuDF {@code LIST}s, and the shapes that are refused. */
class ListBindingTest {

    private static final long OFFSETS = 0x7f0000001000L;
    private static final long VALUES = 0x7f0000002000L;

    private static ListBinding list(long rows, long childElements) {
        return ListBinding.describe(
                "cuDF",
                OFFSETS,
                (rows + 1) * Integer.BYTES,
                childElements == 0 ? 0L : VALUES,
                childElements * Double.BYTES,
                childElements,
                rows,
                ElementKind.DOUBLE);
    }

    @Test
    void describesAListOfRaggedRows() {
        // 1000 rows holding 7324 elements between them: no padding, no rectangle, no gather.
        ListBinding binding = list(1_000, 7_324);
        assertThat(binding.rows()).isEqualTo(1_000);
        assertThat(binding.childElements()).isEqualTo(7_324);
        assertThat(binding.offsets().rows()).isEqualTo(1_001);
        assertThat(binding.values()).isPresent();
        assertThat(binding.isEmpty()).isFalse();
    }

    @Test
    void theOffsetsBufferHoldsOneMoreEntryThanThereAreRows() {
        assertThat(list(16, 100).offsets().rows()).isEqualTo(17);
    }

    /**
     * A column whose every row is empty is ordinary, not broken.
     *
     * <p>cuDF need not allocate a child at all for it. A loop over such a row runs zero iterations,
     * so no element is ever read and there is nothing to point at -- which the caller answers with
     * a sentinel rather than this refusing.
     */
    @Test
    void aColumnOfEmptyRowsHasNoChildPayload() {
        ListBinding binding = list(1_000, 0);
        assertThat(binding.isEmpty()).isTrue();
        assertThat(binding.values()).isEmpty();
        assertThat(binding.childElements()).isZero();
        assertThat(binding.offsets()).isNotNull();
    }

    /**
     * A sliced LIST is refused, and this is the case that would otherwise be silent.
     *
     * <p>A slice keeps the parent's offsets, so its first offset is where the slice begins rather
     * than zero. A kernel indexing the child with them reads other rows' elements and returns
     * confidently wrong numbers.
     */
    @Test
    void aSlicedListIsRefusedWithTheReason() {
        assertThatThrownBy(
                        () ->
                                ListBinding.describe(
                                        "cuDF",
                                        OFFSETS,
                                        16 * Integer.BYTES,
                                        VALUES,
                                        100 * Double.BYTES,
                                        100,
                                        1_000,
                                        ElementKind.DOUBLE))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("sliced LIST")
                .hasMessageContaining("from zero");
    }

    @Test
    void aChildShorterThanItsElementCountIsRefused() {
        assertThatThrownBy(
                        () ->
                                ListBinding.describe(
                                        "cuDF",
                                        OFFSETS,
                                        1_001 * Integer.BYTES,
                                        VALUES,
                                        10 * Double.BYTES,
                                        100,
                                        1_000,
                                        ElementKind.DOUBLE))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("values");
    }

    // ----------------------------------------------------------------------------------------
    // Which column types have a device representation at all.
    // ----------------------------------------------------------------------------------------

    @Test
    void fixedWidthNumericColumnsAreBindable() {
        assertThat(ColumnKinds.scalar(new IntType(false))).get().isEqualTo(ElementKind.INT);
        assertThat(ColumnKinds.scalar(new BigIntType(false))).get().isEqualTo(ElementKind.LONG);
        assertThat(ColumnKinds.scalar(new DoubleType(false))).get().isEqualTo(ElementKind.DOUBLE);
    }

    @Test
    void anArrayOfThemIsBindableAsAList() {
        ArrayType arrayOfDouble = new ArrayType(false, new DoubleType(false));
        assertThat(ColumnKinds.listElement(arrayOfDouble)).get().isEqualTo(ElementKind.DOUBLE);
        assertThat(ColumnKinds.isBindable(arrayOfDouble)).isTrue();
        assertThat(ColumnKinds.scalar(arrayOfDouble)).isEmpty();
    }

    @Test
    void anArrayOfSomethingElseIsNot() {
        ArrayType arrayOfString = new ArrayType(false, new VarCharType(false, 32));
        assertThat(ColumnKinds.listElement(arrayOfString)).isEmpty();
        assertThat(ColumnKinds.isBindable(arrayOfString)).isFalse();
        assertThat(ColumnKinds.refuse("features", arrayOfString)).contains("fixed-width numeric");
    }

    @Test
    void aNullableColumnIsRefusedAndToldHowToFixIt() {
        assertThat(ColumnKinds.isBindable(new DoubleType(true))).isFalse();
        assertThat(ColumnKinds.refuse("fare", new DoubleType(true)))
                .contains("validity mask")
                .contains("NOT NULL");
    }

    @Test
    void aNullableArrayIsRefusedTooRatherThanHalfAdmitted() {
        assertThat(ColumnKinds.isBindable(new ArrayType(true, new DoubleType(false)))).isFalse();
    }

    @Test
    void typesWithNoRepresentationNameThemselves() {
        assertThat(ColumnKinds.refuse("amount", new DecimalType(false, 10, 2)))
                .contains("DECIMAL")
                .contains("no device representation");
    }
}
