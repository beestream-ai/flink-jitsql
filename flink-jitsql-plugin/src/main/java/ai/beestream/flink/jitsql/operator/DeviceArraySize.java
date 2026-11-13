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
package ai.beestream.flink.jitsql.operator;

/**
 * The element count of a two-dimensional device array, multiplied in {@code long} and refused if it
 * will not fit an {@code int}.
 *
 * <p>Every engine here sizes its buffers as a product of two configured numbers — rows by features,
 * references by dimensions, queries by references — and every one of them multiplied in {@code int}
 * and handed the result straight to a {@code FloatArray} constructor. The products are not small:
 * these are the benchmarks, and their sizes come from the command line.
 *
 * <p>What makes this worth a refusal rather than a comment is that the usual outcome is not a
 * crash. {@code ReferenceJoinEngine} with a 65,536-row reference table and a 65,536-row batch —
 * both plausible powers of two, neither remarkable — multiplies to exactly 2<sup>32</sup>, which
 * truncates to <b>zero</b>. {@code new FloatArray(0)} is legal, the kernel's {@code @Parallel}
 * bound is {@code 0}, no iteration runs, nothing fails, and the benchmark prints a time for a
 * contraction it never performed. A number that fast is the sort of result one wants to believe.
 *
 * <p>Negative products throw out of the array constructor, which at least says something; this
 * exists for the half that does not.
 */
final class DeviceArraySize {

    private DeviceArraySize() {}

    /**
     * {@code a * b} as an {@code int}.
     *
     * @param what the buffer being sized, named in the message
     * @throws IllegalArgumentException if the product does not fit an {@code int}
     */
    static int elements(String what, int a, int b) {
        final long product = (long) a * b;
        if (product > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(
                    what
                            + " needs "
                            + product
                            + " elements ("
                            + a
                            + " x "
                            + b
                            + "), past the "
                            + Integer.MAX_VALUE
                            + " a device array can index. Lower one of the two; multiplied in"
                            + " int this wraps, and at exactly 2^32 it wraps to zero, which"
                            + " allocates cleanly and runs no iterations at all.");
        }
        return (int) product;
    }
}
