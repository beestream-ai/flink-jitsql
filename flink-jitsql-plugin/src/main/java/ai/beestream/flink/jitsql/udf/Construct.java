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

/**
 * The thing in a user's function that decided whether it could become a kernel.
 *
 * <p>Named rather than numbered because the person who has to act on it wrote the method. "The
 * provider declined" sends them to read the whole body; {@code OBJECT_ALLOCATION at line 34} sends
 * them to the line.
 *
 * <p>Note which of these are <em>not</em> here. A loop is not a refusal, and neither is a branch, a
 * switch, an accumulator or an array read. Those are exactly the constructs that send a UDF to the
 * CPU today, because an expression tree cannot hold them -- and they are the reason this analyser
 * exists.
 */
public enum Construct {

    /** Allocating anything: a kernel has no heap. */
    OBJECT_ALLOCATION,

    /** Reading or writing a field of an object, including {@code this}. */
    OBJECT_ACCESS,

    /** A call to a method this analyser will not follow. */
    UNSUPPORTED_CALL,

    /** A {@code java.lang.Math} entry point with no device equivalent. */
    UNSUPPORTED_MATH,

    /** A string anywhere in the body. */
    STRING,

    /** {@code throw}, or a handler. A kernel has nowhere to unwind to. */
    EXCEPTION,

    /** A monitor. There is nothing to synchronise with. */
    SYNCHRONIZATION,

    /** The method calls itself, directly or through a helper. */
    RECURSION,

    /** An array access this analyser could not prove in range. */
    UNPROVABLE_BOUNDS,

    /** A parameter or return type with no device representation. */
    UNSUPPORTED_TYPE,

    /** The bytecode could not be read at all. */
    UNREADABLE
}
