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

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;
import java.util.stream.Collectors;

/**
 * What the provider was asked, what it answered, and why.
 *
 * <h2>Why this is not just logging</h2>
 *
 * <p>Flink can only ever report that the provider declined. Everything that would let somebody act
 * on that -- which route was tried, which stage failed, whether cuDF cannot express the shape or
 * this provider merely never wired it -- exists for the length of one method call and then does
 * not. In practice that has meant reproducing a query with debug logging turned on, on a cluster,
 * to answer a question the provider already knew the answer to.
 *
 * <p>So decisions are kept. The log is per thread, because routing happens per subtask and mixing
 * two subtasks' decisions into one list makes neither readable, and it is bounded, because a long
 * job routes a great many nodes and none of this is worth a leak.
 */
public final class RoutingDecisions {

    /** Decisions kept per thread. Enough to explain a query; far too few to grow without end. */
    static final int KEEP = 64;

    private static final ThreadLocal<Deque<Decision>> RECENT =
            ThreadLocal.withInitial(ArrayDeque::new);

    private RoutingDecisions() {}

    /** One node, and what became of it. */
    public static final class Decision {

        private final String node;
        private final Route route;

        Decision(String node, Route route) {
            this.node = node;
            this.route = route;
        }

        public String node() {
            return node;
        }

        public Route route() {
            return route;
        }

        public boolean accepted() {
            return route.accepted();
        }

        @Override
        public String toString() {
            return node + " -> " + route;
        }
    }

    /** Records a decision and hands the route straight back, so call sites stay one expression. */
    public static Route record(AccelNode node, Route route) {
        Deque<Decision> recent = RECENT.get();
        recent.addLast(new Decision(node.getClass().getSimpleName(), route));
        while (recent.size() > KEEP) {
            recent.removeFirst();
        }
        return route;
    }

    /** This thread's decisions, oldest first. */
    public static List<Decision> decisions() {
        return new ArrayList<>(RECENT.get());
    }

    /** Only the refusals, which is what somebody asking "why was this not accelerated" wants. */
    public static List<Decision> refusals() {
        return decisions().stream().filter(d -> !d.accepted()).collect(Collectors.toList());
    }

    /** One line per decision, for a log statement or a test failure message. */
    public static String format() {
        List<Decision> decisions = decisions();
        if (decisions.isEmpty()) {
            return "nothing was offered to this provider on this thread";
        }
        return decisions.stream().map(Decision::toString).collect(Collectors.joining("\n"));
    }

    /** Drops this thread's decisions. Call when a subtask finishes. */
    public static void clear() {
        RECENT.remove();
    }
}
