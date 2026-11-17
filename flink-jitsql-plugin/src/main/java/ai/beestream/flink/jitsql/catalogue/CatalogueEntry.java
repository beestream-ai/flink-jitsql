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

import org.apache.flink.table.accelerator.AccelNode;
import org.apache.flink.table.accelerator.AccelWorkProfile;

/**
 * One kind of IR node, and the question of whether cuDF expresses it directly.
 *
 * <p>An entry answers for a node kind, not for a query. Whether the subtree as a whole is worth
 * admitting is a separate decision made once over the whole tree, because a subtree with one
 * unroutable stage is worse than no subtree at all -- it drains back to the host in the middle.
 */
public interface CatalogueEntry {

    /** The IR node kind this entry answers for. */
    Class<? extends AccelNode> nodeKind();

    /**
     * Routes one node of {@link #nodeKind()}.
     *
     * <p>Must never return null and must never throw: a refusal is an ordinary answer and the
     * caller is a TaskManager starting a task that has a perfectly good CPU operator waiting.
     */
    Route route(AccelNode node, AccelWorkProfile work, RoutingContext context);
}
