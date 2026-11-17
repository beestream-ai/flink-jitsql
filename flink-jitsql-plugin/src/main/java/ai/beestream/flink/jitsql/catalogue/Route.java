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

import java.util.Optional;

/**
 * Where one node is to be executed, or why nowhere.
 *
 * <p>The rule this is the answer to:
 *
 * <blockquote>
 *
 * One IR node expressible as <em>one</em> cuDF primitive goes to cuDF. Anything else goes to the
 * kernel generator. What neither can express is refused, with a named reason.
 *
 * </blockquote>
 *
 * <p>The word <em>one</em> carries the weight. A projection is expressible in cuDF, but as a
 * sequence of per-column primitives with a materialised intermediate between each pair, so it is
 * not <em>directly</em> expressible and goes to the generator, which fuses it into a single pass.
 * Phrasing the test as capability rather than cost is what makes this total: there is no node both
 * routes can serve, so no tie needs breaking.
 *
 * <p>There is no fourth case. A catalogue that can return nothing is a catalogue that silently
 * drops a node, which is the failure this type exists to make impossible.
 */
public final class Route {

    public enum Kind {
        /** One cuDF primitive expresses this node. */
        CUDF,
        /** The kernel generator expresses this node. */
        KERNEL,
        /** Neither does. */
        REFUSED
    }

    private final Kind kind;
    private final AccelNode node;
    private final CudfPrimitive primitive;
    private final CudfPrimitive compaction;
    private final RefusalCode code;
    private final String because;

    private Route(
            Kind kind,
            AccelNode node,
            CudfPrimitive primitive,
            CudfPrimitive compaction,
            RefusalCode code,
            String because) {
        this.kind = kind;
        this.node = node;
        this.primitive = primitive;
        this.compaction = compaction;
        this.code = code;
        this.because = because;
    }

    public static Route cudf(AccelNode node, CudfPrimitive primitive) {
        return new Route(Kind.CUDF, node, primitive, null, null, null);
    }

    public static Route kernel(AccelNode node) {
        return new Route(Kind.KERNEL, node, null, null, null, null);
    }

    /**
     * A kernel stage whose output a cuDF primitive then reshapes.
     *
     * <p>A filter is the case that needs this and the reason the rule is stated per node rather
     * than per stage. Evaluating the predicate is expression work, so it belongs to the generator;
     * turning the resulting mask into a dense run of surviving rows is {@code selectedIndices},
     * which no kernel should be asked to do. One node, two stages, and the catalogue names both
     * rather than letting the assembler infer the second.
     */
    public static Route kernelThen(AccelNode node, CudfPrimitive compaction) {
        return new Route(Kind.KERNEL, node, null, compaction, null, null);
    }

    public static Route refused(AccelNode node, RefusalCode code, String because) {
        return new Route(Kind.REFUSED, node, null, null, code, because);
    }

    public Kind kind() {
        return kind;
    }

    public AccelNode node() {
        return node;
    }

    public boolean accepted() {
        return kind != Kind.REFUSED;
    }

    /** Present only when {@link #kind()} is {@link Kind#CUDF}. */
    public Optional<CudfPrimitive> primitive() {
        return Optional.ofNullable(primitive);
    }

    /**
     * A cuDF primitive that reshapes this node's output after the kernel stage, where one applies.
     * Empty for every route but a filter today.
     */
    public Optional<CudfPrimitive> compaction() {
        return Optional.ofNullable(compaction);
    }

    /** Present only when refused. */
    public Optional<RefusalCode> code() {
        return Optional.ofNullable(code);
    }

    /** Present only when refused: the sentence a human reads. */
    public Optional<String> because() {
        return Optional.ofNullable(because);
    }

    @Override
    public String toString() {
        switch (kind) {
            case CUDF:
                return "cuDF " + primitive;
            case KERNEL:
                return compaction == null
                        ? "generated kernel"
                        : "generated kernel, then cuDF " + compaction.name();
            default:
                return "refused (" + code + "): " + because;
        }
    }
}
