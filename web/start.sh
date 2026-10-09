#!/usr/bin/env bash
set -euo pipefail
H3_WEB_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
H3_WEB_PYTHON=${H3_PYTHON:-"$H3_WEB_ROOT/.venv/bin/python"}
if [[ ! -x "$H3_WEB_PYTHON" ]]; then H3_WEB_PYTHON=$(command -v python3); fi
cd -- "$H3_WEB_ROOT"
exec "$H3_WEB_PYTHON" -m web.app "$@"
