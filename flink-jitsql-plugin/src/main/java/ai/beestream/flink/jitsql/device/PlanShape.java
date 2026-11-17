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

import java.util.Objects;

/**
 * What makes one compiled execution plan unusable in place of another.
 *
 * <p>Three things, and each of them is a wrong answer rather than a slowdown if it is left out.
 *
 * <ul>
 *   <li><b>Row count.</b> Every {@code Cudf} primitive takes its row count as an {@code int}
 *       argument to the descriptor, so the count is fixed when the task graph is built. A plan
 *       built for a full batch handed a short final batch does not adapt; it processes the wrong
 *       number of rows. See {@link ai.beestream.flink.jitsql.catalogue.RowCountBinding}.
 *   <li><b>Capacity dimensions.</b> Anything else the graph was sized against -- a list child's
 *       capacity, a group count -- for the same reason.
 *   <li><b>The device.</b> A plan compiled and queued for one GPU cannot be handed another GPU's
 *       pointers. A thread that changes device gets a different plan rather than silently reusing
 *       one built elsewhere.
 * </ul>
 */
public final class PlanShape {

    private final int rows;
    private final long capacityDims;
    private final int cudaOrdinal;

    public PlanShape(int rows, long capacityDims, int cudaOrdinal) {
        this.rows = rows;
        this.capacityDims = capacityDims;
        this.cudaOrdinal = cudaOrdinal;
    }

    /** The common case: a graph sized only by its row count, on the default device. */
    public static PlanShape ofRows(int rows) {
        return new PlanShape(rows, 0L, 0);
    }

    public int rows() {
        return rows;
    }

    public long capacityDims() {
        return capacityDims;
    }

    public int cudaOrdinal() {
        return cudaOrdinal;
    }

    @Override
    public boolean equals(Object other) {
        if (this == other) {
            return true;
        }
        if (!(other instanceof PlanShape)) {
            return false;
        }
        PlanShape that = (PlanShape) other;
        return rows == that.rows
                && capacityDims == that.capacityDims
                && cudaOrdinal == that.cudaOrdinal;
    }

    @Override
    public int hashCode() {
        return Objects.hash(rows, capacityDims, cudaOrdinal);
    }

    @Override
    public String toString() {
        return "rows=" + rows + ", dims=" + capacityDims + ", device=" + cudaOrdinal;
    }
}
