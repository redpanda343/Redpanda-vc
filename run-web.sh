#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

if [ -x env/bin/python ]; then
    exec env/bin/python -u -m shared.desktop.web "$@"
fi
if [ -x .venv/bin/python ]; then
    exec .venv/bin/python -u -m shared.desktop.web "$@"
fi
exec python3 -u -m shared.desktop.web "$@"
