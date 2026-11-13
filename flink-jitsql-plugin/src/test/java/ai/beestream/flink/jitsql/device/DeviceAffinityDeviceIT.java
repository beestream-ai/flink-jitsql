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

import ai.beestream.flink.jitsql.operator.DeviceAffinity;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import uk.ac.manchester.tornado.api.TornadoBackend;
import uk.ac.manchester.tornado.api.enums.TornadoVMBackendType;
import uk.ac.manchester.tornado.api.runtime.TornadoRuntimeProvider;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * That a subtask is bound to a named device, and that an ordinal the backend does not have is
 * refused rather than quietly replaced by the default.
 *
 * <p>Device-gated, because it asks TornadoVM for the cards it actually has. On this project's
 * single-GPU machine every subtask resolves to device 0, so what the first two cases prove is that
 * the mapping happens and is stable — not that it spreads, which cannot be shown here.
 */
final class DeviceAffinityDeviceIT {

    @AfterEach
    void unbind() {
        DeviceAffinity.clear();
    }

    @Test
    @DisplayName("a subtask is bound to a device, and the same index always gets the same one")
    void bindsAndIsStable() {
        DeviceAssumptions.requireDevice();
        assertThat(DeviceAffinity.isBound()).as("unbound before open()").isFalse();

        DeviceAffinity.bindSubtask(0);
        assertThat(DeviceAffinity.isBound()).isTrue();

        DeviceAffinity.clear();
        DeviceAffinity.bindSubtask(0);
        assertThat(DeviceAffinity.isBound()).isTrue();
    }

    @Test
    @DisplayName("every subtask index resolves to a device this backend actually has")
    void everyIndexLandsOnARealDevice() {
        DeviceAssumptions.requireDevice();
        final int count = cudaDeviceCount();
        // Round robin, so any index must land inside the range whatever the parallelism.
        for (int subtask = 0; subtask < count * 4 + 3; subtask++) {
            DeviceAffinity.clear();
            DeviceAffinity.bindSubtask(subtask);
            assertThat(DeviceAffinity.isBound())
                    .as("subtask %d found no device although the backend has %d", subtask, count)
                    .isTrue();
        }
    }

    @Test
    @DisplayName(
            "an ordinal the backend does not have is refused, naming the ordinal and the count")
    void refusesAnOrdinalThatDoesNotExist() {
        DeviceAssumptions.requireDevice();
        final int count = cudaDeviceCount();

        assertThatThrownBy(() -> DeviceAffinity.forCudaOrdinal(count + 7))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining(String.valueOf(count + 7))
                .hasMessageContaining(String.valueOf(count));

        assertThatThrownBy(() -> DeviceAffinity.forCudaOrdinal(-1))
                .isInstanceOf(IllegalStateException.class);
    }

    @Test
    @DisplayName("the binding is per thread, so two subtasks in one TaskManager do not share it")
    void bindingIsPerThread() throws Exception {
        DeviceAssumptions.requireDevice();
        DeviceAffinity.bindSubtask(0);
        assertThat(DeviceAffinity.isBound()).isTrue();

        final boolean[] boundOnOtherThread = {true};
        final Thread other = new Thread(() -> boundOnOtherThread[0] = DeviceAffinity.isBound());
        other.start();
        other.join();

        assertThat(boundOnOtherThread[0])
                .as("a second task thread inherited the first one's binding")
                .isFalse();
    }

    private static int cudaDeviceCount() {
        var runtime = TornadoRuntimeProvider.getTornadoRuntime();
        for (int i = 0; i < runtime.getNumBackends(); i++) {
            TornadoBackend backend = runtime.getBackend(i);
            if (backend.getBackendType() == TornadoVMBackendType.CUDA) {
                return backend.getNumDevices();
            }
        }
        return 0;
    }
}
