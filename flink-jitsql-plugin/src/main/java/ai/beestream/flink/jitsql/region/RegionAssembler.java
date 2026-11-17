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

import ai.beestream.flink.jitsql.catalogue.CudfCapabilityCatalogue;
import ai.beestream.flink.jitsql.catalogue.CudfPrimitive;
import ai.beestream.flink.jitsql.catalogue.RefusalCode;
import ai.beestream.flink.jitsql.catalogue.Route;
import ai.beestream.flink.jitsql.catalogue.RoutingContext;
import ai.beestream.flink.jitsql.device.BufferKey;

import org.apache.flink.table.accelerator.AccelInput;
import org.apache.flink.table.accelerator.AccelNode;
import org.apache.flink.table.accelerator.AccelWorkProfile;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;

/**
 * Turns a subtree into an ordered list of stages, without being written for any particular shape.
 *
 * <h2>What it does</h2>
 *
 * <p>Walks the subtree leaves-first, asks the catalogue what each node is, and gives each stage a
 * buffer key for its output and the earlier stages' keys for its inputs. Nothing in here mentions a
 * projection feeding a sum, or a top-N, or a grok -- the four hand-written regions that do are each
 * one query shape's worth of exactly this, with the wiring written out by hand.
 *
 * <h2>Admitting the whole subtree at once</h2>
 *
 * <p>A node with no route declines the <em>subtree</em>, not just itself. That is not tidiness: a
 * region whose third stage has to fall back to the host drains the device buffers, rebuilds rows,
 * and hands them back -- paying the staging cost the region exists to avoid while keeping the
 * compilation cost it adds. The Spark integration measured this exact shape at 1.62x against the
 * 4.89x its fully-compiled sibling reached.
 *
 * <p>The refusal names the stage that caused it, because "the provider declined" with eight stages
 * in the subtree is not a diagnosis.
 */
public final class RegionAssembler {

    private RegionAssembler() {}

    /**
     * Plans a region, or says which node stopped it.
     *
     * @param batchRows how many rows a batch carries. Part of the plan rather than of execution:
     *     every cuDF primitive fixes its row count when the graph is built, so a region is planned
     *     for one batch size and a different one needs a different plan.
     */
    public static RegionPlan assemble(
            AccelNode subtree, AccelWorkProfile work, RoutingContext context, int batchRows) {
        if (subtree == null) {
            throw new IllegalArgumentException("there is no subtree to assemble");
        }
        if (batchRows <= 0) {
            throw new IllegalArgumentException(
                    "a region cannot be planned for " + batchRows + " rows a batch");
        }
        List<Stage> stages = new ArrayList<>();
        Map<AccelNode, Integer> stageOf = new IdentityHashMap<>();
        RegionPlan refusal = walk(subtree, work, context.forRegion(), batchRows, stages, stageOf);
        if (refusal != null) {
            return refusal;
        }
        if (stages.isEmpty()) {
            return RegionPlan.refused(
                    subtree,
                    RefusalCode.NO_CUDF_PRIMITIVE,
                    "the subtree is a bare input and has no work to run on a device");
        }
        return RegionPlan.of(stages);
    }

    /** Post-order, so a stage's inputs always already have keys. Returns a refusal or null. */
    private static RegionPlan walk(
            AccelNode node,
            AccelWorkProfile work,
            RoutingContext context,
            int batchRows,
            List<Stage> stages,
            Map<AccelNode, Integer> stageOf) {
        List<BufferKey> inputs = new ArrayList<>();
        for (AccelNode input : node.inputs()) {
            RegionPlan refusal = walk(input, work, context, batchRows, stages, stageOf);
            if (refusal != null) {
                return refusal;
            }
            Integer producer = stageOf.get(input);
            if (producer != null) {
                inputs.add(BufferKey.output(producer));
            }
        }
        if (node instanceof AccelInput) {
            // The boundary, not a stage. Whatever fills it -- a scan stage, or rows staged by the
            // operator -- is the caller's business and deliberately not the assembler's.
            return null;
        }
        Route route = CudfCapabilityCatalogue.route(node, work, context);
        if (!route.accepted()) {
            return RegionPlan.refused(
                    node,
                    route.code().orElse(RefusalCode.NO_CUDF_PRIMITIVE),
                    "stage "
                            + stages.size()
                            + " ("
                            + node.getClass().getSimpleName()
                            + ") has no route: "
                            + route.because().orElse("no reason given"));
        }
        int index = stages.size();
        stages.add(new Stage(index, node, route, inputs, outputsOf(route, index), batchRows));
        stageOf.put(node, index);

        // A filter is one node and two stages: the kernel writes a mask, and selectedIndices turns
        // it into a dense run of survivors. The catalogue names the second, so the assembler emits
        // it without knowing what a filter is.
        if (route.compaction().isPresent()) {
            int compaction = stages.size();
            CudfPrimitive primitive = route.compaction().get();
            stages.add(
                    new Stage(
                            compaction,
                            node,
                            Route.cudf(node, primitive),
                            List.of(BufferKey.output(index)),
                            outputs(compaction, primitive.outputArity()),
                            batchRows));
            stageOf.put(node, compaction);
        }
        return null;
    }

    private static List<BufferKey> outputsOf(Route route, int index) {
        int arity = route.primitive().map(CudfPrimitive::outputArity).orElse(1);
        return outputs(index, arity);
    }

    private static List<BufferKey> outputs(int index, int arity) {
        if (arity == 1) {
            return List.of(BufferKey.output(index));
        }
        List<BufferKey> keys = new ArrayList<>(arity);
        for (int i = 0; i < arity; i++) {
            keys.add(BufferKey.output(index, i));
        }
        return keys;
    }
}
