#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

if [ -x ./redpanda ]; then
    exec ./redpanda "$@"
fi
if [ -x desktop/target/release/redpanda ]; then
    exec desktop/target/release/redpanda "$@"
fi
printf '%s\n' 'Build with cargo build --release --locked --manifest-path desktop/Cargo.toml'
exit 1
