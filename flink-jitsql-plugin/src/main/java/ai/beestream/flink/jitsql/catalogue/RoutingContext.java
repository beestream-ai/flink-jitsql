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
package ai.beestream.flink.jitsql.catalogue;

import java.util.EnumSet;
import java.util.Set;

/**
 * What is true of this JVM, supplied to the catalogue rather than discovered by it.
 *
 * <p>The catalogue calls nothing native. Every {@code Cudf.isXxxAvailable()} probe loads a shared
 * library on first use, so a catalogue that called them could not be unit-tested anywhere without a
 * GPU -- and the one test that matters most, that routing is total for every node kind, has nothing
 * to do with a device being present. Resolving availability here keeps that test honest and keeps
 * the probes in the provider, which is where the rest of them already live.
 */
public final class RoutingContext {

    /**
     * Which of the provider's two halves is asking, because they can emit different things.
     *
     * <p>Not a detail of implementation. The SPI has always had two seams: {@code accept} decorates
     * a plan Flink already built, and a scan source replaces the front of one. A primitive can be
     * reachable from one and not the other -- {@code Cudf.reduce} is emitted by every device region
     * and by no operator -- so "can cuDF express this" has to be asked on behalf of somebody.
     * Baking the answer into the primitive instead would make the catalogue lie to one caller or
     * the other.
     */
    public enum Emitter {
        /** {@code accept}, decorating an operator Flink already has a CPU implementation for. */
        OPERATOR,
        /** The assembler, building a whole task graph from a subtree. */
        REGION
    }

    private final boolean deviceAvailable;
    private final boolean cudfAvailable;
    private final Set<CudfPrimitive.Availability> available;
    private final long maxSortBytes;
    private final long maxJoinBytes;
    private final Emitter emitter;

    private RoutingContext(
            boolean deviceAvailable,
            boolean cudfAvailable,
            Set<CudfPrimitive.Availability> available,
            long maxSortBytes,
            long maxJoinBytes,
            Emitter emitter) {
        this.emitter = emitter;
        this.deviceAvailable = deviceAvailable;
        this.cudfAvailable = cudfAvailable;
        this.available = available;
        this.maxSortBytes = maxSortBytes;
        this.maxJoinBytes = maxJoinBytes;
    }

    public static RoutingContext of(
            boolean deviceAvailable,
            boolean cudfAvailable,
            Set<CudfPrimitive.Availability> available,
            long maxSortBytes,
            long maxJoinBytes) {
        return new RoutingContext(
                deviceAvailable,
                cudfAvailable,
                available.isEmpty()
                        ? EnumSet.noneOf(CudfPrimitive.Availability.class)
                        : EnumSet.copyOf(available),
                maxSortBytes,
                maxJoinBytes,
                Emitter.OPERATOR);
    }

    /** Every cuDF entry point present, for a host where the binding is fully installed. */
    public static RoutingContext fullyAvailable(long maxStagingBytes) {
        return new RoutingContext(
                true,
                true,
                EnumSet.allOf(CudfPrimitive.Availability.class),
                maxStagingBytes,
                maxStagingBytes,
                Emitter.OPERATOR);
    }

    /** No device at all: every route refuses, which is an ordinary answer and not a failure. */
    public static RoutingContext noDevice() {
        return new RoutingContext(
                false,
                false,
                EnumSet.noneOf(CudfPrimitive.Availability.class),
                0L,
                0L,
                Emitter.OPERATOR);
    }

    /** A device, but no usable cuDF binding: the kernel generator is the only route. */
    public static RoutingContext kernelOnly(long maxStagingBytes) {
        return new RoutingContext(
                true,
                false,
                EnumSet.noneOf(CudfPrimitive.Availability.class),
                maxStagingBytes,
                maxStagingBytes,
                Emitter.OPERATOR);
    }

    /** The same context, asked on behalf of the assembler rather than of {@code accept}. */
    public RoutingContext forRegion() {
        return new RoutingContext(
                deviceAvailable,
                cudfAvailable,
                available,
                maxSortBytes,
                maxJoinBytes,
                Emitter.REGION);
    }

    public Emitter emitter() {
        return emitter;
    }

    public boolean deviceAvailable() {
        return deviceAvailable;
    }

    public boolean cudfAvailable() {
        return cudfAvailable;
    }

    public boolean available(CudfPrimitive.Availability probe) {
        return cudfAvailable && available.contains(probe);
    }

    /**
     * The width budget {@code GpuSortSpec} refuses against.
     *
     * <p>Two budgets rather than one because the provider has always had two: a sort stages a
     * permutation beside its payload and a join stages two inputs, so the constants differ and
     * collapsing them here would quietly change which rows are refused.
     */
    public long maxSortBytes() {
        return maxSortBytes;
    }

    /** The width budget {@code GpuJoinSpec} refuses against. */
    public long maxJoinBytes() {
        return maxJoinBytes;
    }
}
