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
# Installs RAPIDS libcudf from NVIDIA's Python wheels -- no conda, no Python environment --
# and puts it on LD_LIBRARY_PATH. Source it; do not run it:
#
#     source scripts/cudf-env.sh
#
# Safe to source repeatedly: it downloads only when RAPIDS_HOME has no marker file.
#
# Why this exists rather than a sentence in the README: libcudf needs SIX directories on
# LD_LIBRARY_PATH, not the one you would guess. It pulls in librapids_logger and libnvcomp,
# and libkvikio's wheel carries an auditwheel-mangled libzstd of its own in a side-car
# directory. Get one wrong and TornadoVM's shim reports isAvailable() == false with no
# further explanation, the provider declines every cuDF-served operator, the query returns
# the right answer on the host, and the test suite skips rather than fails. Nothing tells
# you anything is wrong.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "cudf-env.sh sets variables in the current shell, so it has to be sourced:" >&2
    echo "    source ${BASH_SOURCE[0]}" >&2
    exit 1
fi

_jitsql_env="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/env.sh"
[ -r "$_jitsql_env" ] && source "$_jitsql_env"
: "${RAPIDS_HOME:=$HOME/.local/share/rapids-libcudf}"

# libcudf-cu12 is the default because it is what this project's device results were measured
# against, including the 43-of-43 device run on TornadoVM 8.1.0's released CUDA SDK. Set
# CUDF_PACKAGE=libcudf-cu13 for a CUDA 13 build; then the CUDA runtime and NVRTC wheels are
# needed too, and this fetches them.
: "${CUDF_VERSION:=26.8.*}"
: "${CUDF_PACKAGE:=libcudf-cu12}"

if [ ! -s "$RAPIDS_HOME/.installed" ]; then
    echo "Installing $CUDF_PACKAGE $CUDF_VERSION into $RAPIDS_HOME" >&2
    command -v python3 >/dev/null || { echo "cudf-env.sh: python3 is needed to fetch the wheels" >&2; return 1; }
    mkdir -p "$RAPIDS_HOME/wheels" || return 1
    python3 -m pip download --quiet --timeout 120 --retries 10 --only-binary=:all: --python-version 3.12 \
        --platform manylinux_2_28_x86_64 --platform manylinux_2_27_x86_64 \
        --platform manylinux_2_17_x86_64 --platform manylinux2014_x86_64 \
        --extra-index-url https://pypi.nvidia.com -d "$RAPIDS_HOME/wheels" "$CUDF_PACKAGE==$CUDF_VERSION" \
        || { echo "cudf-env.sh: downloading the libcudf wheels failed" >&2; return 1; }
    if [ "$CUDF_PACKAGE" = libcudf-cu13 ]; then
        python3 -m pip download --quiet --timeout 120 --retries 10 --only-binary=:all: --no-deps \
            --python-version 3.12 --platform manylinux_2_28_x86_64 --platform manylinux_2_17_x86_64 \
            --platform manylinux2014_x86_64 -d "$RAPIDS_HOME/wheels" \
            "nvidia-cuda-runtime==13.*" "nvidia-cuda-nvrtc==13.*" \
            || { echo "cudf-env.sh: downloading the CUDA 13 runtime wheels failed" >&2; return 1; }
    fi
    for wheel in "$RAPIDS_HOME"/wheels/*.whl; do
        python3 -m zipfile -e "$wheel" "$RAPIDS_HOME" || return 1
    done
    rm -rf "$RAPIDS_HOME/wheels"
    date -u +%FT%TZ > "$RAPIDS_HOME/.installed"
    echo "$CUDF_PACKAGE $CUDF_VERSION" >> "$RAPIDS_HOME/.installed"
fi

# The six, in the order env.sh states them, so there is one definition of the list.
if declare -F jitsql_rapids_ld_path >/dev/null; then
    _rapids_path="$(jitsql_rapids_ld_path)"
else
    _rapids_path="$RAPIDS_HOME/libcudf/lib64:$RAPIDS_HOME/librmm/lib64:$RAPIDS_HOME/libkvikio/lib64:$RAPIDS_HOME/rapids_logger/lib64:$RAPIDS_HOME/nvidia/libnvcomp/lib64:$RAPIDS_HOME/libkvikio_cu12.libs"
fi
case ":${LD_LIBRARY_PATH:-}:" in
    *":$RAPIDS_HOME/libcudf/lib64:"*) ;;
    *) export LD_LIBRARY_PATH="$_rapids_path:${LD_LIBRARY_PATH:-}" ;;
esac
export RAPIDS_HOME CUDF_HOME="$RAPIDS_HOME/libcudf" RMM_HOME="$RAPIDS_HOME/librmm"
unset _rapids_path
