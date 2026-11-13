#!/usr/bin/env bash
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

set -euo pipefail

FLINK_HOME="${1:-}"
REFS="${2:-32768}"
QUERIES="${3:-65536}"
DIMS="${4:-128}"
RUNS="${5:-5}"
SCORE_LIMIT_MB="${SCORE_LIMIT_MB:-512}"
PARALLELISM="${PARALLELISM:-1}"
VERIFY="${VERIFY:-false}"
DATA="${DATA:-/tmp/flink-gpu-refjoin-${REFS}-${DIMS}-${QUERIES}}"

if [[ -z "${FLINK_HOME}" || ! -x "${FLINK_HOME}/bin/flink" ]]; then
    echo "usage: $0 <flink-dist-dir> [refs] [queries] [dims] [runs]" >&2
    exit 1
fi

JAR=$(ls "${FLINK_HOME}"/examples/table/ReferenceJoinBenchmark.jar \
         "${FLINK_HOME}"/examples/table/*ReferenceJoinBenchmark*.jar 2>/dev/null | head -1 || true)
if [[ -z "${JAR}" ]]; then
    echo "ReferenceJoinBenchmark jar not found under ${FLINK_HOME}/examples/table" >&2
    exit 1
fi

"${FLINK_HOME}/bin/start-cluster.sh"
# stop the cluster however this script exits, so a failed run does not leave one behind
trap '"${FLINK_HOME}/bin/stop-cluster.sh" >/dev/null 2>&1 || true' EXIT

COMMON=(--data "${DATA}" --ref-count "${REFS}" --query-rows "${QUERIES}" --dims "${DIMS}")

if [[ ! -d "${DATA}/queries" ]]; then
    echo "### generating ${REFS} references and ${QUERIES} queries x ${DIMS} dims into ${DATA}"
    "${FLINK_HOME}/bin/flink" run "${JAR}" --generate "${COMMON[@]}"
else
    echo "### reusing ${DATA}"
fi

if [[ "${VERIFY}" == "true" ]]; then
    echo
    echo "############ verifying device answers against vanilla Flink SQL ############"
    "${FLINK_HOME}/bin/flink" run "${JAR}" --verify "${COMMON[@]}" \
        --parallelism "${PARALLELISM}"
fi

echo
echo "############ refs=${REFS}x${DIMS}  queries=${QUERIES}  parallelism=${PARALLELISM}  runs=${RUNS} ############"
"${FLINK_HOME}/bin/flink" run "${JAR}" --sweep "${COMMON[@]}" \
    --parallelism "${PARALLELISM}" \
    --runs "${RUNS}" \
    --score-limit-mb "${SCORE_LIMIT_MB}"
