/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
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

import ai.beestream.flink.jitsql.codegen.GpuValueType;

import org.apache.flink.table.data.RowData;
import org.apache.flink.table.data.binary.BinaryRowData;
import org.apache.flink.table.data.columnar.ColumnarRowData;

/**
 * Copies one column out of a stream of {@link RowData} into a packed, column-major staging buffer.
 *
 * <p>This is the boundary where the input's physical layout stops mattering. Which implementation
 * applies is fixed by what is upstream in the plan, and is knowable at plan time in three of the
 * four positions:
 *
 * <table>
 *   <tr><th>Upstream</th><th>RowData impl</th><th>Tier</th></tr>
 *   <tr><td>Vectorized Parquet/ORC source</td><td>{@link ColumnarRowData}</td><td>1</td></tr>
 *   <tr><td>Downstream of a shuffling Exchange</td><td>{@link BinaryRowData}</td><td>2</td></tr>
 *   <tr><td>Chained after a Calc/Expand</td><td>BoxedWrapperRowData</td><td>4</td></tr>
 * </table>
 *
 * <p><b>Gathering is not an extra pass.</b> Buffering is mandatory — TornadoVM needs a sized,
 * contiguous buffer — so every field is read and written once regardless. Writing column-major
 * rather than row-major is the same traffic to a different address. The tiers therefore differ by
 * per-field access cost, not by an added traversal.
 */
public interface RowGather {

    /**
     * Appends {@code row}'s value for the configured column at {@code position}.
     *
     * <p>Every gather writes the value before returning. A deferred tier existed here until
     * 2026-10-27 -- it held a run's bounds and issued one {@code MemorySegment.copy} per run out of
     * a {@code VectorizedColumnBatch} -- and it was removed because the deferral could not be made
     * correct. A vectorized reader resets and refills the same vectors for each batch, so a pending
     * run stops pointing at its own values the moment the reader moves on; the run had to be
     * flushed at the batch's last row, and a gather only sees the rows that reach it. A filter-only
     * Calc upstream drops that row, and the copy then came out of a refilled vector as plausible,
     * in-range, wrong numbers. No signal rescues it: the operator does not see the boundary either,
     * and by the time the next row arrives the refill has happened.
     *
     * <p>It was worth about 3% to give up. §T-staging A/B'd it with the format held still: 659 ms
     * against 682 ms, ranges 639-713 and 635-902, and the operator's own breakdown identical to the
     * tenth of a millisecond either way. The conclusion recorded there is that per-row accessor
     * cost is not where staging time goes.
     */
    void accept(RowData row, int position);

    /** Human-readable tier name, for the metrics report. */
    String tier();

    /**
     * Picks the cheapest gather for the concrete {@link RowData} implementation actually observed.
     *
     * <p>This is a runtime decision because the physical row class is not part of the {@code
     * ExecNodeGraph}: the planner declares only {@code InternalTypeInfo<RowData>} and the connector
     * chooses the implementation inside {@code getScanRuntimeProvider}. There is no source ability
     * for negotiating layout.
     */
    static RowGather forColumn(RowData sample, int field, GpuValueType type, StagingColumn target) {
        if (sample instanceof ColumnarRowData) {
            return new ColumnarGather(field, type, target);
        }
        if (sample instanceof BinaryRowData) {
            return new BinaryGather(field, type, target);
        }
        return new GenericGather(field, type, target, sample);
    }

    /**
     * Write side of a staging column, so gathers do not depend on the buffer implementation.
     *
     * <p>Values cross as {@code double} whatever the column's width, because {@code INT} and {@code
     * FLOAT} are both exact in one; the implementation narrows to the buffer it holds.
     */
    interface StagingColumn {
        void set(int position, double value);
    }
}
