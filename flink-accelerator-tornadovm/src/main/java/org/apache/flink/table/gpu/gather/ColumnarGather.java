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

package org.apache.flink.table.gpu.gather;

import org.apache.flink.table.data.RowData;
import org.apache.flink.table.data.columnar.ColumnarRowData;
import org.apache.flink.table.gpu.codegen.GpuValueType;

/**
 * Vectorized batches, read one row at a time.
 *
 * <p>The only tier for a columnar source since 2026-10-27. A deferred sibling copied runs of rows
 * with one {@code MemorySegment.copy} and was removed: the run had to be issued before the reader
 * refilled its vectors, and a gather sees only the rows that survive the operators above it. See
 * {@link RowGather#accept} for what that cost and what it bought.
 *
 * <p>The accessor is chosen by the column's declared type. Reading a four-byte field with {@code
 * getDouble} does not fail, it returns eight bytes of whatever is there, so this switch is the
 * difference between a value and garbage.
 */
final class ColumnarGather implements RowGather {

    private final int field;
    private final GpuValueType type;
    private final StagingColumn target;

    ColumnarGather(int field, GpuValueType type, StagingColumn target) {
        this.field = field;
        this.type = type;
        this.target = target;
    }

    @Override
    public void accept(RowData row, int position) {
        switch (type) {
            case INT:
                target.set(position, ((ColumnarRowData) row).getInt(field));
                break;
            case FLOAT:
                target.set(position, ((ColumnarRowData) row).getFloat(field));
                break;
            default:
                target.set(position, ((ColumnarRowData) row).getDouble(field));
                break;
        }
    }

    @Override
    public String tier() {
        // The parenthetical named the reason the bulk tier had not been chosen. There is no
        // bulk tier now, so it would be answering a question nobody asked.
        return "tier1-columnar(per-row)";
    }
}
