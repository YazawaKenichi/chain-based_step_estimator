#!/bin/bash
# 使い方: ./train.bash [データセットのディレクトリ] [train.py のオプション...]
#   ディレクトリを省略すると TARGET_DIR を使う。どこから実行してもよい。

TARGET_DIR="$HOME/raspicat2/rosbags/chain/tsudanuma/tooi/dataset_circuit_v2"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ $# -gt 0 ] && [ "${1#-}" = "$1" ]; then
    TARGET="$1"
    shift
else
    TARGET="$TARGET_DIR"
fi

exec python3 "$SCRIPT_DIR/train.py" "$TARGET" "$@"
