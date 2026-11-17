#!/usr/bin/env bash
# Every task in one run: the host suites, the Python suites, and the device checks.
#
# Separate runs leave "did they all pass together" open. This answers it, and it fails loudly
# when a dependency is missing rather than stopping with no message -- which is what the cuDF
# stages do without the RAPIDS libraries on LD_LIBRARY_PATH.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH="${BENCH_REPO:-$HOME/Projects/flink-jitsql-benchmarking}"

export JAVA_HOME="${JAVA_HOME:-$HOME/.sdkman/candidates/java/21.0.2-open}"
export TORNADO_SDK="${TORNADO_SDK:-$HOME/.sdkman/candidates/tornadovm/8.1.0-jdk21-cuda}"

fail() { echo "PRECONDITION FAILED: $*" >&2; exit 2; }

command -v nvidia-smi >/dev/null 2>&1 || fail "no nvidia-smi, so there is no device to verify on"
[[ -x "$JAVA_HOME/bin/java" ]] || fail "no JDK 21 at $JAVA_HOME"
[[ -f "$TORNADO_SDK/tornado-argfile" ]] || fail "no TornadoVM SDK at $TORNADO_SDK"
grep -q 'tornado.drivers.cuda' "$TORNADO_SDK/tornado-argfile" \
    || fail "$TORNADO_SDK is not a CUDA build"

# shellcheck disable=SC1091
source "$HERE/scripts/cudf-env.sh" >/dev/null 2>&1 \
    || fail "could not set up the RAPIDS libraries; the cuDF stages need six directories on LD_LIBRARY_PATH"

echo "device:  $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
echo "sdk:     $TORNADO_SDK"
echo

status=0

echo "--- host test suites ---"
( cd "$HERE" && mvn -q -o -pl flink-jitsql-plugin test ) \
    && echo "provider suite: PASS" || { echo "provider suite: FAIL"; status=1; }

if [[ -d "$BENCH" ]]; then
    echo
    echo "--- B7 measurement protocol ---"
    ( cd "$BENCH" && python3 scripts/test-analyse-runs.py >/dev/null 2>&1 ) \
        && echo "analyse-runs: PASS" || { echo "analyse-runs: FAIL"; status=1; }
    ( cd "$BENCH" && python3 scripts/test-check-correctness.py >/dev/null 2>&1 ) \
        && echo "check-correctness: PASS" || { echo "check-correctness: FAIL"; status=1; }
    rm -rf "$BENCH/scripts/__pycache__"
else
    echo "B7: benchmarking repo not at $BENCH, skipped"
fi

echo
echo "--- device checks ---"
CPF="$(mktemp)"
( cd "$HERE" && mvn -q -o -pl flink-jitsql-bench dependency:build-classpath \
      -Dmdep.outputFile="$CPF" ) || fail "could not resolve the bench classpath"
RUN_CP="$HERE/flink-jitsql-bench/target/classes:$HERE/flink-jitsql-plugin/target/classes:$(cat "$CPF")"
rm -f "$CPF"

"$JAVA_HOME/bin/java" "@${TORNADO_SDK}/tornado-argfile" \
    --enable-native-access=ALL-UNNAMED \
    -Dtornado.enable.fma=false -Dtornado.cuda.compile.profile=repro \
    -cp "$RUN_CP" ai.beestream.flink.jitsql.bench.DeviceAcceptance || status=1

exit "$status"
