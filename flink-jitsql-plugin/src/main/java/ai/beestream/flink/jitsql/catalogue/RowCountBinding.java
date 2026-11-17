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
 * When a cuDF primitive learns how many rows it will process.
 *
 * <p>This is not a detail. Every {@code Cudf} factory method takes its row count as an ordinary
 * {@code int} argument, which means the count is fixed when the {@code LibraryTaskDescriptor} is
 * built and therefore when the {@link uk.ac.manchester.tornado.api.TaskGraph} around it is built. A
 * graph built for 262,144 rows cannot be handed 300 rows on the last batch; it needs a second
 * graph, and a second compilation.
 *
 * <p>Two things follow, and both are load-bearing elsewhere:
 *
 * <ul>
 *   <li>The assembler has to build one graph per distinct row count, not one per query.
 *   <li>The execution-plan cache has to carry the row count in its key, which is why {@code
 *       (rowCount, capacityDims, cudaOrdinal)} is the shape of that key rather than something keyed
 *       on the query alone.
 * </ul>
 *
 * <p>{@code GeneratedKernelEngine} records having been bitten by this for {@code sortedOrder},
 * {@code groupSum} and {@code runningSum}. The reason it applies to all of them is structural --
 * the count is a constructor argument -- rather than a quirk of those three.
 */
public enum RowCountBinding {

    /** The row count is fixed when the task graph is built. Every {@code Cudf} primitive today. */
    CAPTURED_AT_BUILD,

    /**
     * The row count travels with the execution, so one graph serves any batch size. No cuDF
     * primitive behaves this way at present; the constant exists so that a binding which does can
     * be described without the catalogue having to grow a new concept.
     */
    PASSED_AT_EXECUTION
}
