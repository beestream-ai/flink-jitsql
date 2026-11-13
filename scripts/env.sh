#!/usr/bin/env bash
#
# Copyright (c) 2026 Beestream Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# The one place this project states where things are.
#
# Every value below can be overridden from the environment, and ROOT is derived from this
# file's own location rather than written down, so a clone anywhere works without editing
# anything. Source it; do not run it:
#
#     source scripts/env.sh
#
# Only two have no sensible default on another machine -- the JDK and the TornadoVM SDK --
# and scripts/preflight.sh says so plainly when they are wrong rather than failing later
# for a reason that looks unrelated.

# This file's directory, even when sourced rather than executed.
if [ -n "${BASH_SOURCE[0]:-}" ]; then
    export JITSQL_ROOT="${JITSQL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
else
    export JITSQL_ROOT="${JITSQL_ROOT:-$(pwd)}"
fi

# --- the two with no default -------------------------------------------------------------

# JDK 21, and exactly 21: TornadoVM's off-heap arrays are java.lang.foreign, preview on 21
# and final on 22, so the SDK, this project and the TaskManager must all be on the same one.
# Derived from javac on PATH when unset.
export JAVA_HOME="${JAVA_HOME:-$(dirname "$(dirname "$(readlink -f "$(command -v javac 2>/dev/null)" 2>/dev/null)")" 2>/dev/null)}"

# The TornadoVM SDK: the directory holding bin/, lib/, share/java/tornado and etc/.
#     sdk install tornadovm 8.1.0-jdk21-cuda
# Point this at the versioned directory, never at ~/.sdkman/candidates/tornadovm/current --
# `current` follows whatever was installed last, and on this machine it resolved to a 2.2.0
# OpenCL SDK while every artifact here was built against 8.1.0 CUDA.
export TORNADOVM_HOME="${TORNADOVM_HOME:-${TORNADO_SDK:-}}"
export TORNADO_SDK="${TORNADO_SDK:-$TORNADOVM_HOME}"

# --- versions ----------------------------------------------------------------------------

export JITSQL_VERSION="${JITSQL_VERSION:-0.1.0-SNAPSHOT}"
export TORNADO_VERSION="${TORNADO_VERSION:-8.1.0-jdk21}"
export FLINK_VERSION="${FLINK_VERSION:-2.3.0}"

# --- this project's own build output -------------------------------------------------------

export PLUGIN_JAR="${PLUGIN_JAR:-$JITSQL_ROOT/flink-jitsql-plugin/target/flink-jitsql-plugin-$JITSQL_VERSION.jar}"
export BENCH_DIR="${BENCH_DIR:-$JITSQL_ROOT/flink-jitsql-bench/target}"

# --- the other checkouts --------------------------------------------------------------------

# Flink with the accelerator SPI: branch gpu-offload. Not a release, and it has to be built
# and installed before this project compiles against it.
export FLINK_SRC="${FLINK_SRC:-$(dirname "$JITSQL_ROOT")/flink}"
# A built Flink distribution, for the demos that start a cluster rather than a MiniCluster.
export FLINK_DIST="${FLINK_DIST:-$FLINK_SRC/build-target}"
# TornadoVM sources. Needed only to build an SDK; the released one is enough for everything
# here, including the cuDF path, since 8.1.0 ships lib/libtornado-cudf.so already built.
export TORNADOVM_SRC="${TORNADOVM_SRC:-$(dirname "$JITSQL_ROOT")/TornadoVM}"

# Generated data sets. Not under /tmp: it is tmpfs here, so a large data set is written to RAM.
export DATA_ROOT="${DATA_ROOT:-$HOME/gpu-bench-data}"

# --- RAPIDS -----------------------------------------------------------------------------

# libcudf and its dependencies, for the operators cuDF serves. scripts/cudf-env.sh installs
# them here from NVIDIA's wheels if the directory does not exist.
export RAPIDS_HOME="${RAPIDS_HOME:-$HOME/.local/share/rapids-libcudf}"

# Six directories, not the obvious one. libcudf pulls in librapids_logger and libnvcomp, and
# libkvikio's wheel carries an auditwheel-mangled libzstd of its own in a side-car directory.
# A missing one of these surfaces as isAvailable() == false and nothing else, so when the
# cuDF path declines for no stated reason, check with
#     ldd "$TORNADOVM_HOME/lib/libtornado-cudf.so" | grep 'not found'
jitsql_rapids_ld_path() {
    printf '%s' \
        "$RAPIDS_HOME/libcudf/lib64:$RAPIDS_HOME/librmm/lib64:$RAPIDS_HOME/libkvikio/lib64:$RAPIDS_HOME/rapids_logger/lib64:$RAPIDS_HOME/nvidia/libnvcomp/lib64:$RAPIDS_HOME/libkvikio_cu12.libs"
}

if [ -d "$RAPIDS_HOME" ] && [[ ":${LD_LIBRARY_PATH:-}:" != *":$RAPIDS_HOME/libcudf/lib64:"* ]]; then
    export LD_LIBRARY_PATH="$(jitsql_rapids_ld_path):${LD_LIBRARY_PATH:-}"
fi

# --- the argfile ---------------------------------------------------------------------------

# TornadoVM's own JVM flags, generated by its build from `tornado --printJavaFlags`, so they
# stay correct across versions instead of being a hand-copied list that rots.
#
# A released SDK ships only the template: the expanded file exists in an SDK you built
# yourself, which is why `@$TORNADOVM_HOME/tornado-argfile` used to work here and fails on a
# fresh install. Expanding it is what the template itself tells you to do.
jitsql_argfile() {
    local t="$TORNADOVM_HOME/tornado-argfile"
    if [ ! -s "$t" ] && [ -s "$t.template" ]; then
        TORNADOVM_HOME="$TORNADOVM_HOME" envsubst < "$t.template" > "$t" || return 1
    fi
    printf '%s' "$t"
}
