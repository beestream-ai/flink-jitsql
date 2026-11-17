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

import ai.beestream.flink.jitsql.catalogue.RefusalCode;
import ai.beestream.flink.jitsql.catalogue.Route;
import ai.beestream.flink.jitsql.device.BufferKey;

import org.apache.flink.table.accelerator.AccelNode;

import java.util.Collections;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

/**
 * An ordered list of stages that together run one subtree on the device, or the reason there is
 * none.
 *
 * <p>This is what replaces the six hand-written regions. Each of those is a class whose fields are
 * its intermediates and whose constructor is its task graph, which is why adding a seventh query
 * shape means writing a seventh class. A plan is the same information with nothing named after the
 * query: stages in order, each one's route from the catalogue, each one's buffers by key.
 */
public final class RegionPlan {

    private final List<Stage> stages;
    private final RefusalCode code;
    private final String because;
    private final AccelNode refusedAt;

    private RegionPlan(List<Stage> stages, RefusalCode code, String because, AccelNode refusedAt) {
        this.stages = stages == null ? null : Collections.unmodifiableList(stages);
        this.code = code;
        this.because = because;
        this.refusedAt = refusedAt;
    }

    static RegionPlan of(List<Stage> stages) {
        return new RegionPlan(stages, null, null, null);
    }

    static RegionPlan refused(AccelNode refusedAt, RefusalCode code, String because) {
        return new RegionPlan(null, code, because, refusedAt);
    }

    public boolean accepted() {
        return stages != null;
    }

    /** Present only when {@link #accepted()}. In execution order, leaves first. */
    public List<Stage> stages() {
        if (stages == null) {
            throw new IllegalStateException("this subtree was declined: " + because);
        }
        return stages;
    }

    public Optional<RefusalCode> code() {
        return Optional.ofNullable(code);
    }

    public Optional<String> because() {
        return Optional.ofNullable(because);
    }

    /** The node that had no route, when one did. */
    public Optional<AccelNode> refusedAt() {
        return Optional.ofNullable(refusedAt);
    }

    /** The key holding the region's result: the last stage's first output. */
    public BufferKey result() {
        List<Stage> all = stages();
        return all.get(all.size() - 1).outputs().get(0);
    }

    /** How many stages run on the device through cuDF rather than through a generated kernel. */
    public long cudfStages() {
        return stages().stream().filter(s -> s.route().kind() == Route.Kind.CUDF).count();
    }

    public long kernelStages() {
        return stages().stream().filter(s -> s.route().kind() == Route.Kind.KERNEL).count();
    }

    @Override
    public String toString() {
        if (!accepted()) {
            return "declined (" + code + "): " + because;
        }
        return stages.stream().map(Stage::toString).collect(Collectors.joining("\n"));
    }
}
