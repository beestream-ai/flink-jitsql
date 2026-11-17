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

import org.apache.flink.table.types.logical.ArrayType;
import org.apache.flink.table.types.logical.LogicalType;
import org.apache.flink.table.types.logical.LogicalTypeRoot;

import java.util.Optional;

/**
 * Which SQL column types have a device representation, and what it is.
 *
 * <p>Two shapes and no others. A fixed-width numeric column is one buffer of elements. An {@code
 * ARRAY} of one is a cuDF {@code LIST}: offsets plus a flattened child, with rows free to differ in
 * length. Everything else -- strings, structs, decimals, anything nullable -- has no representation
 * here and is refused while planning rather than discovered on the device.
 *
 * <p>Nullability is part of the type test rather than a separate one. A nullable column needs a
 * validity mask that no kernel here reads, so admitting one would mean computing over whatever
 * bytes happen to sit in the null slots.
 */
public final class ColumnKinds {

    private ColumnKinds() {}

    /** The element kind of a fixed-width column, or empty when it has no device representation. */
    public static Optional<ElementKind> scalar(LogicalType type) {
        if (type == null || type.isNullable()) {
            return Optional.empty();
        }
        switch (type.getTypeRoot()) {
            case TINYINT:
                return Optional.of(ElementKind.BYTE);
            case INTEGER:
                return Optional.of(ElementKind.INT);
            case BIGINT:
                return Optional.of(ElementKind.LONG);
            case FLOAT:
                return Optional.of(ElementKind.FLOAT);
            case DOUBLE:
                return Optional.of(ElementKind.DOUBLE);
            default:
                return Optional.empty();
        }
    }

    /**
     * The element kind inside an {@code ARRAY}, or empty when the column is not an array of a
     * fixed-width type.
     */
    public static Optional<ElementKind> listElement(LogicalType type) {
        if (type == null || type.isNullable() || type.getTypeRoot() != LogicalTypeRoot.ARRAY) {
            return Optional.empty();
        }
        return scalar(((ArrayType) type).getElementType());
    }

    /** Whether a column of this type can reach a kernel at all, as a scalar or as a list. */
    public static boolean isBindable(LogicalType type) {
        return scalar(type).isPresent() || listElement(type).isPresent();
    }

    /** What to say when it cannot. Names the actual type rather than the category. */
    public static String refuse(String what, LogicalType type) {
        if (type == null) {
            return what + " has no type";
        }
        if (type.isNullable()) {
            return what
                    + " is "
                    + type
                    + ", and a nullable column needs a validity mask no kernel here reads."
                    + " Declaring it NOT NULL is the fix.";
        }
        if (type.getTypeRoot() == LogicalTypeRoot.ARRAY) {
            return what
                    + " is "
                    + type
                    + ", and an array reaches a device only when its elements are a fixed-width"
                    + " numeric type";
        }
        return what + " is " + type + ", which has no device representation";
    }
}
