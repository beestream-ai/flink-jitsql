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
package ai.beestream.flink.jitsql.device;

import java.util.Objects;

/**
 * What one buffer in an assembled region is for, said in terms of the plan rather than of a field
 * name.
 *
 * <p>The hand-written regions each declare their intermediates as named fields -- {@code packedIn},
 * {@code outputs}, {@code sum}, {@code groupKeys}. That is what makes them hand-written: the names
 * only mean anything to the one shape they were written for, so a new shape needs a new class. A
 * key built from a stage's position and its role in that stage means the same thing in every
 * region, which is what lets one assembler serve shapes nobody wrote code for.
 */
public final class BufferKey {

    private final int stage;
    private final String role;
    private final int ordinal;

    private BufferKey(int stage, String role, int ordinal) {
        this.stage = stage;
        this.role = role;
        this.ordinal = ordinal;
    }

    /** The single output of a stage. */
    public static BufferKey output(int stage) {
        return new BufferKey(stage, "out", 0);
    }

    /** One of several outputs of a stage, such as a group-by's keys, sums and count. */
    public static BufferKey output(int stage, int ordinal) {
        return new BufferKey(stage, "out", ordinal);
    }

    /** A buffer a stage needs that is neither its input nor its output, such as a mask. */
    public static BufferKey scratch(int stage, String role) {
        return new BufferKey(stage, Objects.requireNonNull(role), 0);
    }

    public int stage() {
        return stage;
    }

    public String role() {
        return role;
    }

    public int ordinal() {
        return ordinal;
    }

    @Override
    public boolean equals(Object other) {
        if (this == other) {
            return true;
        }
        if (!(other instanceof BufferKey)) {
            return false;
        }
        BufferKey that = (BufferKey) other;
        return stage == that.stage && ordinal == that.ordinal && role.equals(that.role);
    }

    @Override
    public int hashCode() {
        return Objects.hash(stage, role, ordinal);
    }

    @Override
    public String toString() {
        return "stage" + stage + "." + role + (ordinal == 0 ? "" : "[" + ordinal + "]");
    }
}
