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
# Checks everything a device run needs, before the run:
#
#     scripts/preflight.sh
#
# One line per check -- OK, WARN or FAIL -- and exit 1 if any check fails.
#
# This exists because the failures it catches do not look like failures. A missing RAPIDS
# directory makes the cuDF operators decline, so the query returns the right answer on the
# host and the suite reports 23 of 43 tests skipped with BUILD SUCCESS. A TORNADOVM_HOME
# pointing at the wrong SDK measures the wrong TornadoVM. Both have happened here.

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"

JDK=21              # preview features tie this project, the SDK and the TaskManager to one release
MIN_COMPUTE=7.0     # libcudf 26.08 ships code for sm_70 and newer
MIN_DRIVER=525      # CUDA 12

fails=0; warns=0
ok()   { printf '  OK    %-12s %s\n' "$1" "$2"; }
warn() { printf '  WARN  %-12s %s\n' "$1" "$2"; warns=$((warns+1)); }
fail() { printf '  FAIL  %-12s %s\n' "$1" "$2"; fails=$((fails+1)); }
ver_ge() { [ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -1)" = "$2" ]; }

echo "flink-jitsql preflight"

# --- JDK
JAVA_BIN="${JAVA_HOME:+$JAVA_HOME/bin/}java"
if ! command -v "$JAVA_BIN" >/dev/null 2>&1; then
    fail jdk "no java found; set JAVA_HOME to a JDK $JDK"
else
    jv=$("$JAVA_BIN" -XshowSettings:properties -version 2>&1 | awk -F'= ' '/java.specification.version/ {print $2}')
    if [ "$jv" = "$JDK" ]; then ok jdk "JDK $jv${JAVA_HOME:+ at $JAVA_HOME}"
    else fail jdk "JDK $jv; need exactly $JDK (java.lang.foreign is preview on 21 and final on 22)"; fi
fi

# --- TornadoVM SDK
if [ -z "${TORNADOVM_HOME:-}" ] || [ ! -d "${TORNADOVM_HOME:-/nonexistent}" ]; then
    fail tornadovm "TORNADOVM_HOME is not set to an SDK; sdk install tornadovm $TORNADO_VERSION-cuda"
else
    case "$(readlink -f "$TORNADOVM_HOME")" in
        *"$(basename "$TORNADOVM_HOME")"*) ;;
    esac
    if [ "$(basename "$TORNADOVM_HOME")" = current ]; then
        warn tornadovm "TORNADOVM_HOME ends in 'current', which follows whatever was installed last; point it at the versioned directory"
    fi
    rel=$(awk -F= '/^version=/ {print $2}' "$TORNADOVM_HOME/etc/tornado.release" 2>/dev/null)
    backends=$(awk -F= '/^tornado.backends=/ {print $2}' "$TORNADOVM_HOME"/etc/tornado.backend 2>/dev/null)
    if [[ "$backends" != *cuda* ]]; then
        fail tornadovm "${rel:-the SDK} at $TORNADOVM_HOME has backends '${backends:-none}'; the CUDA backend is required"
    else
        want="${TORNADO_VERSION%%-*}"
        if [ -n "$rel" ] && [ "$rel" != "$want" ]; then
            fail tornadovm "SDK is $rel but this project compiles against $TORNADO_VERSION. A kernel's writeReplace() is emitted against one tornado-api; a mismatch fails at run time as 'Kernel entry ... has no writeReplace()'"
        else
            ok tornadovm "TornadoVM ${rel:-?}, backends $backends"
        fi
    fi

    # the argfile: a released SDK ships only the template
    if [ -s "$TORNADOVM_HOME/tornado-argfile" ]; then
        ok argfile "tornado-argfile present"
    elif [ -s "$TORNADOVM_HOME/tornado-argfile.template" ]; then
        if command -v envsubst >/dev/null 2>&1; then
            warn argfile "only the template is present; run: envsubst < \$TORNADOVM_HOME/tornado-argfile.template > \$TORNADOVM_HOME/tornado-argfile"
        else
            fail argfile "only the template is present and envsubst is not installed (gettext)"
        fi
    else
        fail argfile "neither tornado-argfile nor its template is in $TORNADOVM_HOME"
    fi

    # the cuDF shim: 8.1.0 ships it built; earlier releases did not
    SHIM="$TORNADOVM_HOME/lib/libtornado-cudf.so"
    if [ ! -f "$SHIM" ]; then
        warn cudf-shim "no lib/libtornado-cudf.so; the cuDF-served operators will decline and their tests will skip"
    else
        missing=$(ldd "$SHIM" 2>/dev/null | awk '/not found/ {print $1}' | tr '\n' ' ')
        if [ -n "$missing" ]; then
            fail cudf-shim "the shim cannot resolve: $missing -- run 'source scripts/cudf-env.sh' (six directories, not one)"
        else
            ok cudf-shim "libtornado-cudf.so resolves against RAPIDS_HOME=$RAPIDS_HOME"
        fi
    fi
fi

# --- RAPIDS libcudf, checked whether or not an SDK is present
#
# The shim check above lives inside the SDK branch, so without TORNADOVM_HOME it never runs
# and a bare machine would be told nothing about libcudf at all -- then find out later, as a
# silent decline with the right answer. This check is independent of the SDK.
if [ ! -d "$RAPIDS_HOME" ]; then
    warn rapids "no libcudf at RAPIDS_HOME=$RAPIDS_HOME; run 'source scripts/cudf-env.sh' or the cuDF-served operators will decline and their tests will skip"
elif [ ! -f "$RAPIDS_HOME/libcudf/lib64/libcudf.so" ]; then
    warn rapids "$RAPIDS_HOME exists but has no libcudf/lib64/libcudf.so; re-run 'source scripts/cudf-env.sh'"
else
    ok rapids "libcudf at $RAPIDS_HOME"
fi

# --- GPU
if ! command -v nvidia-smi >/dev/null 2>&1; then
    fail gpu "nvidia-smi not found; no NVIDIA driver on this machine"
else
    name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
    cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)
    mem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ -n "$name" ] && ok gpu "$name, ${mem:-?} MiB" || fail gpu "nvidia-smi returned no device"
    if [ -n "$drv" ]; then
        ver_ge "$drv" "$MIN_DRIVER" && ok driver "$drv" || fail driver "$drv; need $MIN_DRIVER or newer for CUDA 12"
    fi
    if [ -n "$cc" ]; then
        ver_ge "$cc" "$MIN_COMPUTE" && ok compute "capability $cc" || fail compute "capability $cc; libcudf needs $MIN_COMPUTE or newer"
    fi
    [ -n "$mem" ] && [ "$mem" -lt 8000 ] && warn memory "${mem} MiB; the benchmarks assume 8 GiB and will decline on size rather than fail"
fi

# --- the Flink SPI, which is not in any release
SPI=~/.m2/repository/org/apache/flink/flink-table-runtime/$FLINK_VERSION
if [ -d "$SPI" ]; then ok flink-spi "flink-table-runtime $FLINK_VERSION installed"
else fail flink-spi "no flink-table-runtime $FLINK_VERSION in ~/.m2; build branch gpu-offload of beestream-ai/flink with 'mvn install -DskipTests'"; fi

# --- this project
if [ -f "$PLUGIN_JAR" ]; then ok plugin "$(basename "$PLUGIN_JAR")"
else warn plugin "not built yet; run 'mvn install' here (only the device runs need it)"; fi

echo
if [ "$fails" -gt 0 ]; then
    echo "$fails check(s) failed, $warns warning(s). A device run will not do what it says."
    exit 1
fi
if [ "$warns" -gt 0 ]; then echo "All checks passed, $warns warning(s)."; else echo "All checks passed."; fi
