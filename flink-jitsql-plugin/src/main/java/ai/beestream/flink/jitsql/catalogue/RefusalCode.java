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
 * Why a subtree, or one node of it, did not reach the device.
 *
 * <p>Flink has its own vocabulary for this in {@code AcceleratorDeclineReason}, and it is not this
 * one. Every constant there describes the Flink-to-provider handshake -- the provider declined, the
 * speed-up was below the floor, the operator could not be built. None of them can say <em>which
 * route inside the provider lost and why</em>, which is the only thing that helps when a query that
 * should have been accelerated was not.
 *
 * <p>So this enum names the provider's own decisions. It is deliberately specific: a code that says
 * {@code PRIMITIVE_SIGNATURE_MISMATCH} is worth little without the sentence beside it, and {@link
 * Route#refused} carries both.
 */
public enum RefusalCode {

    /** TornadoVM itself is not usable in this JVM, so no route exists. */
    DEVICE_UNAVAILABLE,

    /** The cuDF binding is not usable in this JVM, so only the kernel generator is available. */
    CUDF_UNAVAILABLE,

    /** No cuDF primitive is bound for this kind of node at all. */
    NO_CUDF_PRIMITIVE,

    /**
     * A primitive exists for this kind of node, but this instance does not fit its signature -- two
     * grouping columns where the binding takes one, a key of a type it cannot order, and so on.
     */
    PRIMITIVE_SIGNATURE_MISMATCH,

    /** The aggregate function is not one the bound primitive computes. */
    UNSUPPORTED_AGGREGATION,

    /** The primitive is bound in the API but is not yet wired into an operator. */
    PRIMITIVE_NOT_WIRED,

    /** The kernel generator could not express the expression. */
    NO_KERNEL,

    /** The expression is expressible but the resulting row cannot be staged. */
    ROW_NOT_STAGEABLE,

    /** The row is wider than the staging buffers admit. */
    ROW_TOO_WIDE,

    /** A column type no route here handles. */
    TYPE_NOT_SUPPORTED,

    /** The subtree asks for more arithmetic per row than the compiler ceiling admits. */
    OPS_PER_ROW_CEILING,

    /**
     * A predicate whose outcome depends on a device transcendental, which can select other rows.
     */
    DRIFT_SENSITIVE_PREDICATE,

    /** Arithmetic with a DECIMAL operand, which Flink does not evaluate as double arithmetic. */
    DECIMAL_COERCION,

    /** One stage of the subtree had no route, so the whole subtree was declined. See A4. */
    STAGE_HAS_NO_ROUTE
}
