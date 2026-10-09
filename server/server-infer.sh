#!/usr/bin/env bash
set -euo pipefail
H3_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export H3_ENTRYPOINT=infer.server
exec "$H3_ROOT/infer/run.sh" infer "$@"
