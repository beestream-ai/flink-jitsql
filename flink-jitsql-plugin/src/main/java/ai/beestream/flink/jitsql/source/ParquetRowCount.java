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

package ai.beestream.flink.jitsql.source;

import uk.ac.manchester.tornado.cudf.Cudf;

/**
 * The row count in a Parquet footer, narrowed to the {@code int} a device array is indexed by, or
 * refused.
 *
 * <p>All seven readers in this package size their staging buffers at one element per row, so every
 * one of them has to turn the footer's {@code long} into an {@code int} before it can allocate
 * anything. Each did it with a bare cast, and a bare cast is the one thing that cannot be done
 * here: the two outcomes are a silently wrong answer and a silently empty one.
 *
 * <p>Above {@code Integer.MAX_VALUE} rows the narrowing keeps the low 32 bits. A file of 2^31 rows
 * narrows to {@code Integer.MIN_VALUE}, every reader's {@code rows <= 0} guard then returns early,
 * and the file is skipped without a word — a query over it answers from the part files that
 * happened to be smaller. A file of 2^32 + 1000 rows narrows to 1000, and the region reads the
 * first thousand rows of it and reports success.
 *
 * <p>Neither is a case worth surviving, so this refuses instead, naming the file. The ceiling is
 * the device array's, not cuDF's: cuDF reads the file either way.
 */
final class ParquetRowCount {

    private ParquetRowCount() {}

    /**
     * The number of rows in {@code file}, as an {@code int}.
     *
     * @throws IllegalStateException if the file holds more rows than a device array can index
     */
    static int of(String file) {
        final long rows = Cudf.parquetMetadata(file)[0];
        if (rows > Integer.MAX_VALUE) {
            throw new IllegalStateException(
                    "the Parquet part file "
                            + file
                            + " holds "
                            + rows
                            + " rows, past the "
                            + Integer.MAX_VALUE
                            + " a device array can index. Every buffer an accelerated region"
                            + " stages is one element per row, so this file cannot be read on"
                            + " the device whatever the region does with it. Split it into"
                            + " smaller part files.");
        }
        return (int) rows;
    }
}
