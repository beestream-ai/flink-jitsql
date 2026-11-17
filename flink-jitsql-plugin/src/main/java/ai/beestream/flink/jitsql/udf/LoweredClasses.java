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

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

/**
 * Where generated classes live, with no bytecode library anywhere in its signature.
 *
 * <h2>Why this is its own class</h2>
 *
 * <p>The kernel compiler needs two things from the UDF machinery: a directory to put on its
 * classpath, and the bytes of a lowered class when its loader is asked for one. Neither has
 * anything to do with ASM.
 *
 * <p>Taking them from {@code StaticUdfLowering} instead made ASM a hard runtime requirement of
 * <em>every</em> kernel compile, because verifying that class resolves the ASM types in its method
 * signatures. A query with no user function in it then failed to open with {@code
 * NoClassDefFoundError: org/objectweb/asm/tree/ClassNode} on a deployment that had no reason to
 * carry ASM at all -- caught by the fallback operator, so the answers stayed right and the
 * accelerated arm quietly became the CPU arm. That is the worst kind of regression: correct,
 * silent, and only visible as a number that got worse.
 *
 * <p>So the coupling is this class, which mentions nothing but paths and bytes.
 */
public final class LoweredClasses {

    private static final Path OUTPUT = createOutputDirectory();

    private static final ConcurrentMap<String, byte[]> PUBLISHED = new ConcurrentHashMap<>();

    private LoweredClasses() {}

    /** The directory holding generated classes, for the kernel compiler's classpath. */
    public static Path outputDirectory() {
        return OUTPUT;
    }

    /** The bytes of a generated class, or null when this is not one of ours. */
    public static byte[] bytesOf(String binaryName) {
        return PUBLISHED.get(binaryName);
    }

    /**
     * Records a generated class and writes it where {@code javac} can see it.
     *
     * <p>Both, because they answer different lookups: the directory is what the compiler resolves a
     * name on, and the map is what TornadoVM's compiler asks the kernel's own classloader for.
     */
    public static void publish(String generatedName, byte[] bytes) {
        PUBLISHED.put(generatedName.replace('/', '.'), bytes);
        try {
            Path target = OUTPUT.resolve(generatedName + ".class");
            Files.createDirectories(target.getParent());
            Files.write(target, bytes);
            target.toFile().deleteOnExit();
        } catch (IOException e) {
            throw new IllegalStateException("could not publish " + generatedName, e);
        }
    }

    private static Path createOutputDirectory() {
        try {
            Path directory = Files.createTempDirectory("jitsql-udf-");
            directory.toFile().deleteOnExit();
            return directory;
        } catch (IOException e) {
            throw new IllegalStateException("nowhere to write lowered user functions", e);
        }
    }
}
