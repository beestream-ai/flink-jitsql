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
package ai.beestream.flink.jitsql.bench;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;

/**
 * A device allocation TornadoVM did not make, for verifying that a kernel can read one.
 *
 * <p>This exists because the thing {@code bindCUDADeviceBuffer} is for cannot be demonstrated with
 * a TornadoVM array: binding a view to memory the same runtime already manages proves nothing about
 * the case that matters, which is a buffer filled by somebody else entirely. So this calls {@code
 * cudaMalloc} and {@code cudaMemcpy} directly, through the foreign function API, and hands back a
 * raw device pointer.
 *
 * <p>It is a test fixture standing in for whatever would really own such a buffer -- a cuDF column
 * held through the RAPIDS Java API, another library, another process's IPC handle. What the
 * provider needs from any of them is the same: an address, a length, and the promise that the
 * writes have landed.
 */
public final class ForeignDeviceBuffer implements AutoCloseable {

    private static final int CUDA_MEMCPY_HOST_TO_DEVICE = 1;
    private static final int CUDA_MEMCPY_DEVICE_TO_HOST = 2;

    private final Arena arena;
    private final MethodHandle cudaMalloc;
    private final MethodHandle cudaFree;
    private final MethodHandle cudaMemcpy;
    private final long address;
    private final long byteLength;

    private ForeignDeviceBuffer(
            Arena arena,
            MethodHandle cudaMalloc,
            MethodHandle cudaFree,
            MethodHandle cudaMemcpy,
            long address,
            long byteLength) {
        this.arena = arena;
        this.cudaMalloc = cudaMalloc;
        this.cudaFree = cudaFree;
        this.cudaMemcpy = cudaMemcpy;
        this.address = address;
        this.byteLength = byteLength;
    }

    /** Allocates device memory and copies {@code values} into it. */
    public static ForeignDeviceBuffer of(double[] values, String libcudart) throws Throwable {
        Arena arena = Arena.ofShared();
        Linker linker = Linker.nativeLinker();
        SymbolLookup cudart = SymbolLookup.libraryLookup(libcudart, arena);

        MethodHandle malloc =
                linker.downcallHandle(
                        cudart.find("cudaMalloc").orElseThrow(),
                        FunctionDescriptor.of(
                                ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG));
        MethodHandle free =
                linker.downcallHandle(
                        cudart.find("cudaFree").orElseThrow(),
                        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
        MethodHandle memcpy =
                linker.downcallHandle(
                        cudart.find("cudaMemcpy").orElseThrow(),
                        FunctionDescriptor.of(
                                ValueLayout.JAVA_INT,
                                ValueLayout.ADDRESS,
                                ValueLayout.ADDRESS,
                                ValueLayout.JAVA_LONG,
                                ValueLayout.JAVA_INT));

        long bytes = (long) values.length * Double.BYTES;
        MemorySegment pointer = arena.allocate(ValueLayout.ADDRESS.byteSize(), Long.BYTES);
        check((int) malloc.invokeExact(pointer, bytes), "cudaMalloc");
        long address = pointer.get(ValueLayout.ADDRESS, 0).address();

        // allocate(JAVA_DOUBLE, n) binds to the "allocate one double holding n" overload, which
        // gives an eight-byte segment and an out-of-bounds copy. Size in bytes, explicitly.
        MemorySegment host = arena.allocate(bytes, Double.BYTES);
        MemorySegment.copy(values, 0, host, ValueLayout.JAVA_DOUBLE, 0, values.length);
        check(
                (int)
                        memcpy.invokeExact(
                                MemorySegment.ofAddress(address).reinterpret(bytes),
                                host,
                                bytes,
                                CUDA_MEMCPY_HOST_TO_DEVICE),
                "cudaMemcpy host to device");

        return new ForeignDeviceBuffer(arena, malloc, free, memcpy, address, bytes);
    }

    /** Reads the allocation back, so a test can show the kernel wrote where it was told to. */
    public double[] read(int elements) throws Throwable {
        long bytes = (long) elements * Double.BYTES;
        MemorySegment host = arena.allocate(bytes, Double.BYTES);
        check(
                (int)
                        cudaMemcpy.invokeExact(
                                host,
                                MemorySegment.ofAddress(address).reinterpret(bytes),
                                bytes,
                                CUDA_MEMCPY_DEVICE_TO_HOST),
                "cudaMemcpy device to host");
        double[] out = new double[elements];
        MemorySegment.copy(host, ValueLayout.JAVA_DOUBLE, 0, out, 0, elements);
        return out;
    }

    public long address() {
        return address;
    }

    public long byteLength() {
        return byteLength;
    }

    @Override
    public void close() {
        try {
            cudaFree.invokeExact(MemorySegment.ofAddress(address));
        } catch (Throwable ignored) {
            // Closing a fixture; a failure here would mask whatever the test was reporting.
        }
        arena.close();
    }

    private static void check(int status, String what) {
        if (status != 0) {
            throw new IllegalStateException(what + " failed with CUDA status " + status);
        }
    }
}
