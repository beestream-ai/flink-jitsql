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

/**
 * One cuDF entry point this provider can emit a stage for.
 *
 * <p>The catalogue is a statement about <em>capability</em>, so a primitive is described by what it
 * can express and what has to be true for it to run -- never by what it would cost. Cost does not
 * appear anywhere in this package, and that is the point of it: with libraries out of scope there
 * is no node that both cuDF and the kernel generator can serve, so nothing is left for a cost
 * comparison to decide.
 *
 * <p>{@link #signature} is carried for the same reason a schema is: these are native bindings whose
 * argument order is not guessable from the name, and a stage emitted against the wrong arity fails
 * at run time rather than at compile time.
 */
public final class CudfPrimitive {

    /**
     * What gates a primitive at run time. Resolved by {@link RoutingContext}, never called here.
     */
    public enum Availability {
        /** Available whenever the cuDF binding loaded at all. */
        CUDF,
        /** {@code Cudf.isParquetAvailable()}. */
        PARQUET,
        /** {@code Cudf.isColumnsAvailable()}. */
        COLUMNS,
        /** {@code Cudf.isSortKeysAvailable()}. */
        SORT_KEYS,
        /** {@code Cudf.isSortedOrderLongsAvailable()}. */
        SORTED_ORDER_LONGS,
        /** {@code Cudf.isStringColumnsAvailable()}. */
        STRING_COLUMNS,
        /** {@code Cudf.isWriterAvailable()}. */
        WRITER
    }

    private final String name;
    private final String signature;
    private final RowCountBinding rowCountBinding;
    private final Availability availability;
    private final boolean wired;
    private final String notWiredBecause;
    private int outputArity = 1;

    private CudfPrimitive(
            String name,
            String signature,
            RowCountBinding rowCountBinding,
            Availability availability,
            boolean wired,
            String notWiredBecause) {
        this.name = name;
        this.signature = signature;
        this.rowCountBinding = rowCountBinding;
        this.availability = availability;
        this.wired = wired;
        this.notWiredBecause = notWiredBecause;
    }

    /** A primitive an operator can emit a stage for today. */
    public static CudfPrimitive wired(
            String name,
            String signature,
            RowCountBinding rowCountBinding,
            Availability availability) {
        return new CudfPrimitive(name, signature, rowCountBinding, availability, true, null);
    }

    /**
     * A primitive the binding exposes that no operator emits yet.
     *
     * <p>Recorded rather than omitted, because the gap between what cuDF can express and what this
     * provider asks it for is exactly the thing the catalogue exists to make visible. Routing
     * refuses these with {@link RefusalCode#PRIMITIVE_NOT_WIRED}, so behaviour is unchanged until
     * someone wires one.
     */
    public static CudfPrimitive bound(
            String name,
            String signature,
            RowCountBinding rowCountBinding,
            Availability availability,
            String notWiredBecause) {
        return new CudfPrimitive(
                name, signature, rowCountBinding, availability, false, notWiredBecause);
    }

    /**
     * How many buffers this primitive writes.
     *
     * <p>The assembler needs it to size a stage before it knows anything else about the query:
     * {@code groupSum} writes keys, sums and a group count, while {@code sortedOrder} writes one
     * permutation. Getting it from the primitive rather than from the node is what lets one
     * assembler wire a shape nobody wrote code for.
     */
    public int outputArity() {
        return outputArity;
    }

    /** Declares a primitive that writes more than one buffer. */
    public CudfPrimitive writing(int outputs) {
        if (outputs < 1) {
            throw new IllegalArgumentException(name + " must write at least one buffer");
        }
        this.outputArity = outputs;
        return this;
    }

    public String name() {
        return name;
    }

    public String signature() {
        return signature;
    }

    public RowCountBinding rowCountBinding() {
        return rowCountBinding;
    }

    public Availability availability() {
        return availability;
    }

    public boolean wired() {
        return wired;
    }

    /** Present only when {@link #wired()} is false. */
    public String notWiredBecause() {
        return notWiredBecause;
    }

    @Override
    public String toString() {
        return "Cudf." + name + (wired ? "" : " (bound, not wired)");
    }
}
