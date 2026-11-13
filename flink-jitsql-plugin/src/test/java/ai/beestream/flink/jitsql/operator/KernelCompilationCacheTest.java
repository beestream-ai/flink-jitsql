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
package ai.beestream.flink.jitsql.operator;

import ai.beestream.flink.jitsql.codegen.GpuKernelSource;
import ai.beestream.flink.jitsql.codegen.GpuValueType;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.lang.reflect.Method;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * That a kernel is compiled once per JVM however many subtasks ask for it.
 *
 * <h2>Why this test and not a change</h2>
 *
 * <p>Compiling costs a few hundred milliseconds and happens on the task-startup path, so at
 * parallelism p a TaskManager used to pay for the same kernel p times. {@code GeneratedKernel}
 * already caches the compiled class for the whole JVM, keyed on the generated source and reference
 * counted so that one operator closing cannot take the class loader from another still running.
 * Nothing asserted it, and an invariant that only holds while nobody disturbs it is one {@code
 * javac} call away from quietly not holding.
 *
 * <p>{@code COMPILATIONS} is documented as existing "for tests, and for nothing else"; until now no
 * test used it.
 *
 * <h2>What this is not</h2>
 *
 * <p>This is a <em>start-up</em> cost, not a per-batch one. It does not move throughput and is not
 * an answer to where the time goes in an offloaded operator. What it buys is a shorter job start at
 * parallelism and one copy of the kernel in the device's code cache instead of p.
 *
 * <p>The execution <em>plan</em> is deliberately not shared. Each operator builds its own, because
 * a plan holds the buffers it was prepared with and two subtasks running one plan would be a race
 * over those rather than a saving. That is the same split spark-jitudf's TornadoKernelCache makes:
 * the class is per JVM, the plan is per thread.
 */
class KernelCompilationCacheTest {

    /** A kernel that compiles with javac alone — no device, no TornadoVM runtime needed. */
    private static GpuKernelSource kernelOf(String className, String body) {
        final String source =
                "public class "
                        + className
                        + " {\n"
                        + "  public static void evaluate(double[] in, double[] out) {\n"
                        + "    for (int i = 0; i < in.length; i++) { out[i] = "
                        + body
                        + "; }\n"
                        + "  }\n"
                        + "}\n";
        return new GpuKernelSource(
                className,
                "evaluate",
                source,
                new int[] {0},
                new GpuValueType[] {GpuValueType.DOUBLE},
                new GpuValueType[] {GpuValueType.DOUBLE},
                false,
                new int[] {0});
    }

    @Test
    @DisplayName("eight subtasks with the same kernel run javac once, and share one class")
    void sameSourceCompilesOnce() throws Exception {
        final GpuKernelSource kernel = kernelOf("CacheK1", "in[i] * 2.0");
        final int before = GeneratedKernel.COMPILATIONS.get();

        final GeneratedKernel[] handles = new GeneratedKernel[8];
        final Method[] entries = new Method[handles.length];
        try {
            for (int i = 0; i < handles.length; i++) {
                handles[i] = new GeneratedKernel(kernelOf("CacheK1", "in[i] * 2.0"));
                entries[i] = handles[i].compile();
            }

            assertThat(GeneratedKernel.COMPILATIONS.get() - before)
                    .as("javac ran more than once for one kernel source")
                    .isEqualTo(1);

            for (int i = 1; i < entries.length; i++) {
                assertThat(entries[i].getDeclaringClass())
                        .as("subtask %d got its own copy of the class", i)
                        .isSameAs(entries[0].getDeclaringClass());
            }
        } finally {
            for (GeneratedKernel handle : handles) {
                if (handle != null) {
                    handle.close();
                }
            }
        }
        assertThat(kernel).isNotNull();
    }

    @Test
    @DisplayName("two different kernels are two compilations, not one shared by mistake")
    void differentSourcesDoNotShare() throws Exception {
        final int before = GeneratedKernel.COMPILATIONS.get();
        try (GeneratedKernel a = new GeneratedKernel(kernelOf("CacheK2", "in[i] * 2.0"));
                GeneratedKernel b = new GeneratedKernel(kernelOf("CacheK3", "in[i] + 1.0"))) {
            final Method ea = a.compile();
            final Method eb = b.compile();

            assertThat(GeneratedKernel.COMPILATIONS.get() - before).isEqualTo(2);
            assertThat(ea.getDeclaringClass()).isNotSameAs(eb.getDeclaringClass());
        }
    }

    @Test
    @DisplayName(
            "a kernel compiled after the last handle closed is compiled again, not resurrected")
    void theCacheDoesNotOutliveItsLastUser() throws Exception {
        final int before = GeneratedKernel.COMPILATIONS.get();
        try (GeneratedKernel first = new GeneratedKernel(kernelOf("CacheK4", "in[i] - 3.0"))) {
            first.compile();
        }
        try (GeneratedKernel second = new GeneratedKernel(kernelOf("CacheK4", "in[i] - 3.0"))) {
            second.compile();
        }
        // Two, because the entry is reference counted rather than kept for the life of the JVM:
        // nothing outlives the last operator using it, which is what bounds the cache.
        assertThat(GeneratedKernel.COMPILATIONS.get() - before).isEqualTo(2);
    }
}
