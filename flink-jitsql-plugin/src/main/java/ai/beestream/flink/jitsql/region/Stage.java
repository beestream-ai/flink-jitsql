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
package ai.beestream.flink.jitsql.region;

import ai.beestream.flink.jitsql.catalogue.Route;
import ai.beestream.flink.jitsql.device.BufferKey;

import org.apache.flink.table.accelerator.AccelNode;

import java.util.Collections;
import java.util.List;

/** One step of an assembled region: what runs, what it reads and what it writes. */
public final class Stage {

    private final int index;
    private final AccelNode node;
    private final Route route;
    private final List<BufferKey> inputs;
    private final List<BufferKey> outputs;
    private final int rows;

    Stage(
            int index,
            AccelNode node,
            Route route,
            List<BufferKey> inputs,
            List<BufferKey> outputs,
            int rows) {
        this.index = index;
        this.node = node;
        this.route = route;
        this.inputs = Collections.unmodifiableList(inputs);
        this.outputs = Collections.unmodifiableList(outputs);
        this.rows = rows;
    }

    public int index() {
        return index;
    }

    public AccelNode node() {
        return node;
    }

    public Route route() {
        return route;
    }

    /** The keys this stage reads, which are earlier stages' outputs. */
    public List<BufferKey> inputs() {
        return inputs;
    }

    public List<BufferKey> outputs() {
        return outputs;
    }

    /**
     * How many rows this stage was sized for.
     *
     * <p>Part of the stage rather than of the region because it is part of what the graph is
     * compiled against: every cuDF primitive fixes its row count when the descriptor is built.
     */
    public int rows() {
        return rows;
    }

    @Override
    public String toString() {
        return index + ": " + route + " " + inputs + " -> " + outputs + " (" + rows + " rows)";
    }
}
