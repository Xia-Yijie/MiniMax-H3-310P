#!/usr/bin/env bash
set -euo pipefail

H3_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if [[ -z "${H3_CANN_ENV:-}" ]]; then
    for candidate in "${ASCEND_HOME_PATH:-/nonexistent}/set_env.sh" \
        /usr/local/Ascend/ascend-toolkit/set_env.sh \
        /usr/local/Ascend/cann/set_env.sh "$HOME/Ascend/cann-8.5.0/set_env.sh"; do
        if [[ -f "$candidate" ]]; then H3_CANN_ENV=$candidate; break; fi
    done
fi
H3_PYTHON=${H3_PYTHON:-"$H3_ROOT/.venv/bin/python"}
if [[ ! -x "$H3_PYTHON" ]]; then H3_PYTHON=$(command -v python3); fi
# Vendor setup scripts may reference unset variables.
if [[ -n "${H3_CANN_ENV:-}" ]]; then
    if [[ ! -f "$H3_CANN_ENV" ]]; then
        echo "Missing CANN environment script: $H3_CANN_ENV" >&2; exit 1
    fi
    set +u
    source "$H3_CANN_ENV"
    set -u
elif [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    echo "Set H3_CANN_ENV to your CANN set_env.sh, or load CANN before running." >&2
    exit 1
fi
cd -- "$H3_ROOT"
export PYTHONPATH="$H3_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TE_PARALLEL_COMPILER=${TE_PARALLEL_COMPILER:-2}
export MAX_COMPILE_CORE_NUMBER=${MAX_COMPILE_CORE_NUMBER:-2}
exec "$H3_PYTHON" -u -m "${H3_ENTRYPOINT:-infer.generate}" "$@"
