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

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * Bind-time validation, which is the half of binding that can be checked without a GPU.
 *
 * <p>Each case here is a way for a kernel to read bytes that are not what it thinks they are. None
 * of them would fail loudly on a device: the wrong element type reinterprets the bits and computes
 * something plausible, and a short buffer reads past its end into whatever is next. Refusing with a
 * message that names the actual mismatch is the only point at which either is visible.
 */
class DeviceBindingTest {

    private static final long SOME_ADDRESS = 0x7f0000000000L;

    @Test
    void describesAWellFormedBuffer() {
        DeviceBinding binding =
                DeviceBinding.describe("cuDF", SOME_ADDRESS, 8_000, 1_000, ElementKind.DOUBLE);
        assertThat(binding.address()).isEqualTo(SOME_ADDRESS);
        assertThat(binding.rows()).isEqualTo(1_000);
        assertThat(binding.kind()).isEqualTo(ElementKind.DOUBLE);
        assertThat(binding.toString()).contains("cuDF").contains("1000");
    }

    @Test
    void aBufferLargerThanTheRowsNeedIsFine() {
        // An owner is free to hand over a capacity rather than an exact fit, and usually does.
        assertThat(DeviceBinding.describe("cuDF", SOME_ADDRESS, 1 << 20, 1_000, ElementKind.DOUBLE))
                .isNotNull();
    }

    @Test
    void aShortBufferIsRefusedAndTheMessageNamesBothSizes() {
        assertThatThrownBy(
                        () ->
                                DeviceBinding.describe(
                                        "cuDF", SOME_ADDRESS, 7_992, 1_000, ElementKind.DOUBLE))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("7992")
                .hasMessageContaining("8000")
                .hasMessageContaining("cuDF");
    }

    @Test
    void aMismatchedElementTypeIsRefusedAndTheMessageNamesBoth() {
        DeviceBinding asDouble =
                DeviceBinding.describe("cuDF", SOME_ADDRESS, 8_000, 1_000, ElementKind.DOUBLE);
        assertThatThrownBy(() -> asDouble.requireKind(ElementKind.INT))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("DOUBLE")
                .hasMessageContaining("INT");
    }

    @Test
    void theRightElementTypePassesThrough() {
        DeviceBinding asDouble =
                DeviceBinding.describe("cuDF", SOME_ADDRESS, 8_000, 1_000, ElementKind.DOUBLE);
        assertThat(asDouble.requireKind(ElementKind.DOUBLE)).isSameAs(asDouble);
    }

    @Test
    void aBufferWithNoAddressIsRefusedRatherThanSubstituted() {
        assertThatThrownBy(() -> DeviceBinding.describe("cuDF", 0L, 0L, 0L, ElementKind.DOUBLE))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("no address");
    }

    @Test
    void aNegativeRowCountIsRefused() {
        assertThatThrownBy(
                        () ->
                                DeviceBinding.describe(
                                        "cuDF", SOME_ADDRESS, 8_000, -1, ElementKind.DOUBLE))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("-1");
    }

    @Test
    void anUnknownElementKindIsRefused() {
        assertThatThrownBy(() -> DeviceBinding.describe("cuDF", SOME_ADDRESS, 8_000, 1_000, null))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("element kind");
    }

    @Test
    void everyKindKnowsItsWidth() {
        assertThat(ElementKind.BYTE.byteWidth()).isEqualTo(1);
        assertThat(ElementKind.INT.byteWidth()).isEqualTo(4);
        assertThat(ElementKind.LONG.byteWidth()).isEqualTo(8);
        assertThat(ElementKind.FLOAT.byteWidth()).isEqualTo(4);
        assertThat(ElementKind.DOUBLE.byteWidth()).isEqualTo(8);
    }

    /**
     * A view carries no length, and that is a property worth pinning rather than assuming.
     *
     * <p>If a view ever started reporting its capacity as a kernel-visible length, rebinding it
     * between batches would change what the kernel was compiled against and the plan cache would
     * start missing on every short final batch -- silently, as a slowdown rather than a failure.
     */
    @Test
    void anUnboundViewHasNothingToReport() {
        BoundView view = BoundView.open(ElementKind.DOUBLE, 1_024);
        assertThat(view.isBound()).isFalse();
        assertThat(view.capacityElements()).isEqualTo(1_024);
        assertThat(view.kind()).isEqualTo(ElementKind.DOUBLE);
        assertThat(ElementKind.of(view.argument())).isEqualTo(ElementKind.DOUBLE);
        assertThatThrownBy(view::devicePointer).isInstanceOf(IllegalStateException.class);
    }

    @Test
    void aViewOpensForEveryKind() {
        for (ElementKind kind : ElementKind.values()) {
            BoundView view = BoundView.open(kind, 16);
            assertThat(ElementKind.of(view.argument())).as("%s", kind).isEqualTo(kind);
        }
    }

    @Test
    void aBoundViewPassesItsBufferOnRatherThanCopyingIt() {
        // The chaining rule: only a view can say where it points, so an intermediate that has to
        // reach a second stage has to enter as a view. Unbound, there is nothing to pass on, and
        // saying so is better than handing the next stage a zero.
        BoundView view = BoundView.open(ElementKind.DOUBLE, 16);
        assertThatThrownBy(() -> DeviceBinding.ofBoundView("stage 1", view))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("not bound");
    }
}
