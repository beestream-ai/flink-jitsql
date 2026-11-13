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
# Stop a cluster left running by --keep-cluster.
#
#   ./stop.sh
#
# The demos stop the cluster themselves unless --keep-cluster was passed, so
# this is only needed after that.
set -euo pipefail
cd "$(dirname "$0")"
source ./common.sh

if curl -s -m 2 "http://localhost:8081/overview" >/dev/null 2>&1; then
    demo_banner "stopping the cluster"
    "$FLINK_HOME/bin/stop-cluster.sh"
else
    demo_banner "nothing to stop"
    echo "  no cluster answering on http://localhost:8081"
    # Stop anyway: the UI may be bound elsewhere, or mid-shutdown.
    "$FLINK_HOME/bin/stop-cluster.sh" >/dev/null 2>&1 || true
fi
