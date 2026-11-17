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

import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.DoubleArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.api.types.arrays.LongArray;
import uk.ac.manchester.tornado.api.types.arrays.TornadoNativeArray;

/** The fixed-width element types a device view can be opened over. */
public enum ElementKind {
    BYTE(Byte.BYTES),
    INT(Integer.BYTES),
    LONG(Long.BYTES),
    FLOAT(Float.BYTES),
    DOUBLE(Double.BYTES);

    private final int byteWidth;

    ElementKind(int byteWidth) {
        this.byteWidth = byteWidth;
    }

    public int byteWidth() {
        return byteWidth;
    }

    /**
     * A view of {@code elements} of this kind: no allocation, and no TornadoVM array header written
     * anywhere.
     *
     * <p>A view carries no length of its own on purpose. Length is part of what a kernel is
     * compiled against, so a view that carried one would key a new compilation on every batch and
     * give back exactly what caching the plan was meant to save.
     */
    public TornadoNativeArray newView(long elements) {
        switch (this) {
            case BYTE:
                return ByteArray.cudaDeviceView(elements);
            case INT:
                return IntArray.cudaDeviceView(elements);
            case LONG:
                return LongArray.cudaDeviceView(elements);
            case FLOAT:
                return FloatArray.cudaDeviceView(elements);
            case DOUBLE:
                return DoubleArray.cudaDeviceView(elements);
            default:
                throw new IllegalStateException("no view factory for " + this);
        }
    }

    /** The kind a TornadoVM array of this class holds, or null when it is not one of the five. */
    public static ElementKind of(TornadoNativeArray array) {
        if (array instanceof ByteArray) {
            return BYTE;
        }
        if (array instanceof IntArray) {
            return INT;
        }
        if (array instanceof LongArray) {
            return LONG;
        }
        if (array instanceof FloatArray) {
            return FLOAT;
        }
        if (array instanceof DoubleArray) {
            return DOUBLE;
        }
        return null;
    }
}
