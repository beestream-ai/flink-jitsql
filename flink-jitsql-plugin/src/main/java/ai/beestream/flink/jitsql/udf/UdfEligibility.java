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
package ai.beestream.flink.jitsql.udf;

import java.util.Optional;

/** Whether a user's function can become a kernel, and what stopped it if not. */
public final class UdfEligibility {

    private final boolean eligible;
    private final Construct construct;
    private final String detail;
    private final int line;
    private final int loops;
    private final int arrayReads;

    private UdfEligibility(
            boolean eligible,
            Construct construct,
            String detail,
            int line,
            int loops,
            int arrayReads) {
        this.eligible = eligible;
        this.construct = construct;
        this.detail = detail;
        this.line = line;
        this.loops = loops;
        this.arrayReads = arrayReads;
    }

    static UdfEligibility eligible(int loops, int arrayReads) {
        return new UdfEligibility(true, null, null, -1, loops, arrayReads);
    }

    static UdfEligibility refused(Construct construct, String detail, int line) {
        return new UdfEligibility(false, construct, detail, line, 0, 0);
    }

    public boolean eligible() {
        return eligible;
    }

    public Optional<Construct> construct() {
        return Optional.ofNullable(construct);
    }

    public Optional<String> detail() {
        return Optional.ofNullable(detail);
    }

    /** The source line the refusal came from, or -1 when the class carries no line numbers. */
    public int line() {
        return line;
    }

    /**
     * How many back edges the body has.
     *
     * <p>Reported because it is the whole point: a loop is the construct that makes an expression
     * tree unable to hold this function, and therefore the construct that sends it to the CPU
     * today.
     */
    public int loops() {
        return loops;
    }

    public int arrayReads() {
        return arrayReads;
    }

    @Override
    public String toString() {
        if (eligible) {
            return "eligible (" + loops + " loops, " + arrayReads + " array reads)";
        }
        return construct + (line >= 0 ? " at line " + line : "") + ": " + detail;
    }
}
