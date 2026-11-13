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

usage() {
    echo "usage: $0 <check|smoke|haversine|cudf-sort|cudf-groupby|gram> [--printKernel] [--printBytecodes]" \
         "[--rows N] [--data DIR]" >&2
    exit 1
}

DEMO="${1:-}"
shift || true
case "${DEMO}" in
    # The two that answer "does this work here", before any demo that assumes it does.
    # check runs every operator shape in both arms and ends CHECKS: n, FAILURES: 0.
    # smoke runs one kernel and nothing of this project's, so a failure means the machine.
    check)      MAIN=ai.beestream.flink.jitsql.bench.IntegrationCheck ;;
    smoke)      MAIN=ai.beestream.flink.jitsql.bench.FrameworkSmokeTest ;;
    haversine)  MAIN=ai.beestream.flink.jitsql.bench.HaversineSQLExample ;;
    cudf-sort)  MAIN=ai.beestream.flink.jitsql.bench.CudfSortSQLExample ;;
    cudf-groupby) MAIN=ai.beestream.flink.jitsql.bench.CudfGroupBySQLExample ;;
    gram)       MAIN=ai.beestream.flink.jitsql.bench.GramMatrixSQLExample ;;
    *)          usage ;;
esac

ROWS=""
DATA=""
TORNADO_FLAGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --printKernel|--print-kernel)
            TORNADO_FLAGS+=("-Dtornado.printKernel=true")
            ;;
        --printBytecodes|--print-bytecodes)
            TORNADO_FLAGS+=("-Dtornado.print.bytecodes=true")
            ;;
        --rows) ROWS="${2:-}"; shift ;;
        --data) DATA="${2:-}"; shift ;;
        *) echo "unknown argument: $1" >&2; usage ;;
    esac
    shift
done

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "${HERE}/.." && pwd)   # scripts/ sits at the repository root

# Derived from where this script is, not from one machine's home directory: the
# checkouts sit beside this repository, which is how 1-fetch.sh lays them out.
SIBLINGS=$(dirname "${REPO}")
FLINK_SRC="${FLINK_SRC:-${SIBLINGS}/flink}"
FLINK_DIST="${FLINK_DIST:-${FLINK_SRC}/flink-dist/target/flink-2.3.0-bin/flink-2.3.0}"
# The repository's single statement of where things are; see scripts/env.sh.
_jitsql_env="${REPO}/scripts/env.sh"
[ -r "$_jitsql_env" ] && source "$_jitsql_env"

RAPIDS_HOME="${RAPIDS_HOME:-$HOME/.local/share/rapids-libcudf}"

# A JDK 21: whatever is already in the environment if it is one, then the usual
# install locations. TornadoVM's SDK is built with 21 and the cluster must match.
jdk_is_21() { [[ -x "$1/bin/java" ]] && "$1/bin/java" -version 2>&1 | grep -q '"21\.'; }
if ! jdk_is_21 "${JAVA_HOME:-}"; then
    for _c in /usr/lib/jvm/java-21-openjdk /usr/lib/jvm/jdk-21 /usr/lib/jvm/temurin-21-jdk \
              "$HOME/.sdkman/candidates/java/21"*; do
        if jdk_is_21 "$_c"; then JAVA_HOME="$_c"; break; fi
    done
fi
[[ -n "${JAVA_HOME:-}" ]] || { echo "no JDK 21 found -- set JAVA_HOME to one" >&2; exit 1; }

# The SDK, resolved rather than assumed. Explicit TORNADO_SDK wins; otherwise the newest
# CUDA distribution under TORNADO_SRC, which is where `make BACKEND=cuda` leaves it.
TORNADO_SRC="${TORNADO_SRC:-${SIBLINGS}/TornadoVM}"
if [[ -z "${TORNADO_SDK:-}" ]]; then
    TORNADO_SDK=$(ls -d "${TORNADO_SRC}"/dist/*cuda*/*cuda* 2>/dev/null | sort | tail -1 || true)
fi
if [[ -z "${TORNADO_SDK}" || ! -f "${TORNADO_SDK}/tornado-argfile" ]]; then
    echo "no TornadoVM CUDA SDK: set TORNADO_SDK, or build one with 'make BACKEND=cuda' in ${TORNADO_SRC}" >&2
    exit 1
fi
# An SDK without the CUDA backend loads, runs, and accelerates nothing; one without the
# cuDF module declines the sort with a reason that sounds like a missing card. Both are
# worth catching here rather than in the middle of a demo.
if ! grep -q 'tornado.drivers.cuda' "${TORNADO_SDK}/tornado-argfile"; then
    echo "${TORNADO_SDK} is not a CUDA build (no tornado.drivers.cuda in its argfile)" >&2
    exit 1
fi
if [[ "${DEMO}" == gram ]] && ! ls "${TORNADO_SDK}"/share/java/tornado/tornado-cublas-*.jar >/dev/null 2>&1; then
    # Same trap as the cuDF one below: without the binding the provider declines the Gram
    # matrix, the query runs on the host, and it still prints the right matrix.
    echo "no tornado-cublas jar in ${TORNADO_SDK}, so ${DEMO} would run on the host and still" \
         "print the right answer. Rebuild TornadoVM with the cuBLAS module." >&2
    exit 1
fi
if [[ "${DEMO}" == cudf-* && ! -f "${TORNADO_SDK}/lib/libtornado-cudf.so" ]]; then
    # Fatal for a demo rather than a note. Without the shim the provider declines and the query
    # runs on the CPU, printing the same numbers -- which is a CPU run presented as a device one.
    echo "${TORNADO_SDK}/lib/libtornado-cudf.so is missing, so ${DEMO} would run on the host" \
         "and still print the right answer. Install RAPIDS libcudf and rebuild TornadoVM with" \
         "'make BACKEND=cuda'." >&2
    exit 1
fi
# Exported because the provider and BenchmarkRun read it to report what they ran against.
TORNADOVM_HOME="${TORNADO_SDK}"
export TORNADOVM_HOME

# The module jar, the one holding every example class -- not one of the many single-class
# jars the build also produces for `flink run`, whose names carry a classifier.
EXAMPLES=$(ls "${REPO}"/flink-jitsql-bench/target/flink-jitsql-bench-*-SNAPSHOT.jar 2>/dev/null \
           | grep -vE -- '-sources|-javadoc|original-' | head -1 || true)

# check and smoke run in this repository's own JVM against Maven's classpath, so they need
# no Flink distribution -- which matters, because the README's build step installs two Flink
# modules and never produces one. Requiring a distribution here made the one command that is
# supposed to say "it works" the one command a reader could not run.
REQUIRED=("${JAVA_HOME}/bin/java" "${TORNADO_SDK}/tornado-argfile")
case "${DEMO}" in
    check|smoke) ;;
    *) REQUIRED+=("${FLINK_DIST}/lib") ;;
esac
for required in "${REQUIRED[@]}"; do
    if [[ ! -e "${required}" ]]; then
        echo "not found: ${required}" >&2
        exit 1
    fi
done
if [[ -z "${EXAMPLES}" ]]; then
    echo "no examples jar: run 'mvn install -DskipTests' in ${REPO} first" >&2
    exit 1
fi
# The provider, taken from THIS checkout's target/ and put ahead of the distribution's
# lib/ on the classpath. Not from lib/, and the difference has already cost time: the
# distribution keeps whatever gpu-cluster-setup.sh last copied there, so a provider change
# that has been rebuilt and installed is still invisible to a demo reading lib/. That is
# the trap DEPLOYMENT.md records for clusters, and it applies here for the same reason.
# Deleting a class makes it worse rather than better -- the stale jar still has it.
PROVIDER=$(ls "${REPO}"/flink-jitsql-plugin/target/flink-jitsql-plugin-*.jar 2>/dev/null \
           | grep -vE -- '-sources|-javadoc|original-' | head -1 || true)
if [[ -z "${PROVIDER}" ]]; then
    echo "no provider jar: run 'mvn install -DskipTests' in ${REPO} first" >&2
    exit 1
fi

# RAPIDS, for the cuDF demo. Six directories rather than the three first tried here:
# libcudf pulls in librapids_logger and libnvcomp, and libkvikio's wheel carries an
# auditwheel-mangled libzstd of its own in a side-car directory. A missing one of these
# fails as "isAvailable() == false" with no further explanation, so check with
#     ldd "${TORNADO_SDK}/lib/libtornado-cudf.so" | grep 'not found'
if [[ -d "${RAPIDS_HOME}" ]]; then
    export LD_LIBRARY_PATH="${RAPIDS_HOME}/libcudf/lib64:${RAPIDS_HOME}/librmm/lib64:\
${RAPIDS_HOME}/libkvikio/lib64:${RAPIDS_HOME}/rapids_logger/lib64:\
${RAPIDS_HOME}/nvidia/libnvcomp/lib64:${RAPIDS_HOME}/libkvikio_cu12.libs:${LD_LIBRARY_PATH:-}"
elif [[ "${DEMO}" == cudf-* ]]; then
    echo "RAPIDS_HOME=${RAPIDS_HOME} does not exist, so cuDF will report itself unavailable and" \
         "${DEMO} will run on the host while printing the right answer. Set RAPIDS_HOME." >&2
    exit 1
fi

# Quiet by default. These are meant to be watched, and the accelerator's own INFO logging
# is a paragraph of provenance per job; the evidence a demo wants is --printKernel or
# --printBytecodes, not a log line. VERBOSE=1 turns the logging back on, which is what to
# do when neither of those printed anything.
DECISION_LEVEL=WARN
PROVIDER_LEVEL=WARN
if [[ -n "${VERBOSE:-}" ]]; then
    DECISION_LEVEL=INFO
    PROVIDER_LEVEL=DEBUG
fi

LOGCONF=$(mktemp -d)
trap 'rm -rf "${LOGCONF}"' EXIT
cat > "${LOGCONF}/log4j2.properties" <<EOF
rootLogger.level = WARN
rootLogger.appenderRef.console.ref = ConsoleAppender
appender.console.name = ConsoleAppender
appender.console.type = CONSOLE
appender.console.target = SYSTEM_OUT
appender.console.layout.type = PatternLayout
appender.console.layout.pattern = %-5p %c{1} - %m%n
# "Accelerated on this TaskManager" / "Accelerator declined on this TaskManager".
logger.decision.name = org.apache.flink.table.runtime.gpu
logger.decision.level = ${DECISION_LEVEL}
# The provider's own reasons, which are where a decline is explained.
logger.provider.name = ai.beestream.flink.jitsql
logger.provider.level = ${PROVIDER_LEVEL}
# Teardown noise. The MiniCluster stops as soon as the last row is collected, and the
# fetcher then finds it gone and says so; it is not a failure and not a lost row.
logger.fetcher.name = org.apache.flink.streaming.api.operators.collect
logger.fetcher.level = ERROR
# Start-up noise a MiniCluster emits and a demo cannot act on: no Kerberos tokens, no
# log.file to point the dashboard at, no checkpointing on a batch job. All three are
# WARN, all three are correct, and none of them is about this query.
logger.tokens.name = org.apache.flink.runtime.security.token
logger.tokens.level = ERROR
logger.webmonitor.name = org.apache.flink.runtime.webmonitor
logger.webmonitor.level = ERROR
logger.jobmaster.name = org.apache.flink.runtime.jobmaster
logger.jobmaster.level = ERROR
EOF

# The classpath. For check and smoke, Maven's own runtime classpath for the bench module,
# so a fresh clone needs nothing but this repository and an SDK. For the demos, the Flink
# distribution's lib/, because they exercise the planner the way a cluster loads it.
if [[ "${DEMO}" == check || "${DEMO}" == smoke ]]; then
    CP_FILE="${REPO}/flink-jitsql-bench/target/.runtime-classpath"
    if [[ ! -s "${CP_FILE}" ]]; then
        (cd "${REPO}" && mvn -B -q -pl flink-jitsql-bench dependency:build-classpath \
            -Dmdep.outputFile="${CP_FILE}" -Dmdep.includeScope=runtime) \
            || { echo "could not resolve the bench module's classpath; run 'mvn install' first" >&2; exit 1; }
    fi
    RUN_CP="${LOGCONF}:${EXAMPLES}:${PROVIDER}:$(cat "${CP_FILE}")"
else
    RUN_CP="${LOGCONF}:${EXAMPLES}:${PROVIDER}:${FLINK_DIST}/lib/*"
fi

exec "${JAVA_HOME}/bin/java" \
    "@${TORNADO_SDK}/tornado-argfile" \
    -Dtornado.enable.fma=false \
    -Dtornado.cuda.compile.profile=repro \
    ${TORNADO_FLAGS[@]+"${TORNADO_FLAGS[@]}"} \
    ${EXTRA_JVM:-} \
    -cp "${RUN_CP}" \
    "${MAIN}" ${ROWS:+"${ROWS}"} ${DATA:+"${DATA}"}
