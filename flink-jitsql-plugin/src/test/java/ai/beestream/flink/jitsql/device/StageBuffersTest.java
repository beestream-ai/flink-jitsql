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
import uk.ac.manchester.tornado.api.types.arrays.TornadoNativeArray;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/** Wiring a region by key, with no stage naming another. */
class StageBuffersTest {

    private static final int ROWS = 1 << 16;

    /**
     * The device region's shape -- cuDF reads, a kernel computes, cuDF reduces -- wired entirely
     * from stage indices.
     *
     * <p>This is A3's acceptance criterion as a test. Compare it with {@code
     * DeviceParquetProjectSumSource}, which wires the same three stages through fields called
     * {@code packedIn}, {@code outputs} and {@code sum}: those names are why that class serves one
     * query shape and this loop serves any.
     */
    @Test
    void athreeStageRegionWiresItselfFromStageNumbersAlone() {
        try (StageBuffers buffers = new StageBuffers()) {
            // Stage 0: the read. Its outputs are whatever the primitive writes.
            TornadoNativeArray readMeta = buffers.owned(BufferKey.output(0, 0), ElementKind.INT, 8);
            TornadoNativeArray readValues =
                    buffers.owned(BufferKey.output(0, 1), ElementKind.DOUBLE, ROWS);

            // Stage 1: the kernel. It asks for stage 0's output by key, not by name.
            assertThat(buffers.argument(BufferKey.output(0, 1))).isSameAs(readValues);
            TornadoNativeArray computed =
                    buffers.owned(BufferKey.output(1), ElementKind.DOUBLE, ROWS);

            // Stage 2: the reduce. Same again, one stage further on.
            assertThat(buffers.argument(BufferKey.output(1))).isSameAs(computed);
            TornadoNativeArray sum = buffers.owned(BufferKey.output(2), ElementKind.DOUBLE, 1);

            assertThat(buffers.argument(BufferKey.output(0, 0))).isSameAs(readMeta);
            assertThat(buffers.argument(BufferKey.output(2))).isSameAs(sum);
            assertThat(buffers.registered()).hasSize(4);
        }
    }

    /** A longer pipeline needs no more code, which is the whole point. */
    @Test
    void anyNumberOfStagesWiresTheSameWay() {
        try (StageBuffers buffers = new StageBuffers()) {
            int stages = 12;
            for (int stage = 0; stage < stages; stage++) {
                if (stage > 0) {
                    assertThat(buffers.argument(BufferKey.output(stage - 1))).isNotNull();
                }
                buffers.owned(BufferKey.output(stage), ElementKind.DOUBLE, ROWS);
            }
            assertThat(buffers.registered()).hasSize(stages);
        }
    }

    @Test
    void askingTwiceGivesTheSameBuffer() {
        try (StageBuffers buffers = new StageBuffers()) {
            TornadoNativeArray first = buffers.owned(BufferKey.output(0), ElementKind.INT, 64);
            TornadoNativeArray second = buffers.owned(BufferKey.output(0), ElementKind.INT, 64);
            assertThat(second).isSameAs(first);
        }
    }

    @Test
    void askingAgainForADifferentTypeIsRefused() {
        try (StageBuffers buffers = new StageBuffers()) {
            buffers.owned(BufferKey.output(0), ElementKind.INT, 64);
            assertThatThrownBy(() -> buffers.owned(BufferKey.output(0), ElementKind.DOUBLE, 64))
                    .isInstanceOf(IllegalStateException.class)
                    .hasMessageContaining("INT")
                    .hasMessageContaining("DOUBLE");
        }
    }

    @Test
    void askingAgainForMoreRoomIsRefusedRatherThanSilentlyTooSmall() {
        try (StageBuffers buffers = new StageBuffers()) {
            buffers.owned(BufferKey.output(0), ElementKind.INT, 64);
            assertThatThrownBy(() -> buffers.owned(BufferKey.output(0), ElementKind.INT, 65))
                    .isInstanceOf(IllegalStateException.class)
                    .hasMessageContaining("65");
        }
    }

    @Test
    void anUnregisteredKeyNamesWhatIsRegistered() {
        try (StageBuffers buffers = new StageBuffers()) {
            buffers.owned(BufferKey.output(0), ElementKind.INT, 8);
            assertThatThrownBy(() -> buffers.argument(BufferKey.output(3)))
                    .isInstanceOf(IllegalStateException.class)
                    .hasMessageContaining("stage3.out")
                    .hasMessageContaining("stage0.out");
        }
    }

    @Test
    void aKeyIsEitherOwnedOrAViewAndNeverBoth() {
        try (StageBuffers buffers = new StageBuffers()) {
            buffers.owned(BufferKey.output(0), ElementKind.DOUBLE, 8);
            assertThatThrownBy(() -> buffers.view(BufferKey.output(0), ElementKind.DOUBLE, 8))
                    .isInstanceOf(IllegalStateException.class)
                    .hasMessageContaining("already owned");
        }
        try (StageBuffers buffers = new StageBuffers()) {
            buffers.view(BufferKey.output(0), ElementKind.DOUBLE, 8);
            assertThatThrownBy(() -> buffers.owned(BufferKey.output(0), ElementKind.DOUBLE, 8))
                    .isInstanceOf(IllegalStateException.class)
                    .hasMessageContaining("already a view");
        }
    }

    @Test
    void aViewIsHandedToAStageTheSameWayAnOwnedBufferIs() {
        try (StageBuffers buffers = new StageBuffers()) {
            BoundView view = buffers.view(BufferKey.output(0), ElementKind.DOUBLE, ROWS);
            // The stage after it cannot tell the difference, which is what makes it generic.
            assertThat(buffers.argument(BufferKey.output(0))).isSameAs(view.argument());
        }
    }

    /**
     * Handing a buffer to a <em>separate</em> plan only works for views, and the refusal says why.
     *
     * <p>Worth a test because the limitation is TornadoVM's rather than ours and is easy to forget:
     * a managed array has no published device address, so an intermediate that has to outlive its
     * task graph has to have entered as a view in the first place. Within one graph -- which is
     * what a region is -- owned buffers chain freely, which is why this is a constraint on the
     * assembler's boundaries rather than on its internals.
     */
    @Test
    void anOwnedBufferCannotBeHandedToAnotherPlan() {
        try (StageBuffers buffers = new StageBuffers()) {
            buffers.owned(BufferKey.output(0), ElementKind.DOUBLE, 8);
            assertThatThrownBy(() -> buffers.handOn(BufferKey.output(0)))
                    .isInstanceOf(IllegalStateException.class)
                    .hasMessageContaining("does not publish a device address");
        }
    }

    @Test
    void scratchBuffersAreDistinctFromOutputs() {
        try (StageBuffers buffers = new StageBuffers()) {
            TornadoNativeArray mask =
                    buffers.owned(BufferKey.scratch(1, "mask"), ElementKind.BYTE, ROWS);
            TornadoNativeArray out = buffers.owned(BufferKey.output(1), ElementKind.DOUBLE, ROWS);
            assertThat(mask).isNotSameAs(out);
            assertThat(buffers.registered())
                    .extracting(Object::toString)
                    .containsExactlyInAnyOrderElementsOf(List.of("stage1.mask", "stage1.out"));
        }
    }
}
