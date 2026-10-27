"""bag_labeler が出力した *_dataset.csv を読み込む."""

from __future__ import annotations

import csv
import fnmatch
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_COLUMN = '/hx711/raw/data'
# シナリオのディレクトリ名から段差の高さと向きを取る (例: step_150_up_choku_02)
STEP_RE = re.compile(r'step_(\d+)_(up|down)')


@dataclass
class Bag:
    rel: str                  # データセットのルートからの bag ディレクトリの相対パス
    scenario: str             # bag の親ディレクトリ (例: ijo/step_150_up_choku_02)
    elapsed: np.ndarray       # [T] bag 先頭からの秒
    raw: np.ndarray           # [T] センサ生値 (float64)
    label: np.ndarray         # [T] bag_labeler のラベル (0: 区間外, 1: 接触〜odom 停止, 2: 停止後)
    step_height: int | None   # 段差の高さ [mm]。段差のないシナリオは None
    direction: str | None     # 'up' / 'down'。段差のないシナリオは None

    def __len__(self) -> int:
        return len(self.raw)

    def onset_index(self) -> int | None:
        """最初に接触した (label != 0 になった) 行。接触しない bag は None."""
        nz = np.flatnonzero(self.label != 0)
        return int(nz[0]) if len(nz) else None


def load_bag(csv_path: Path, root: Path, column: str = DEFAULT_COLUMN) -> Bag:
    elapsed, raw, label = [], [], []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        if column not in reader.fieldnames:
            raise ValueError(f'{csv_path}: 列 {column} がない (列: {reader.fieldnames})')
        for row in reader:
            # 基準トピック以外の列を指定したとき、まだ受信していない行は空欄になる
            if row[column] == '':
                continue
            elapsed.append(float(row['elapsed']))
            raw.append(float(row[column]))
            label.append(int(row['label']))

    bag_dir = csv_path.parent
    rel = bag_dir.relative_to(root).as_posix()
    scenario = bag_dir.parent.relative_to(root).as_posix()
    m = STEP_RE.search(scenario)
    return Bag(
        rel=rel,
        scenario=scenario,
        elapsed=np.asarray(elapsed, dtype=np.float64),
        raw=np.asarray(raw, dtype=np.float64),
        label=np.asarray(label, dtype=np.int64),
        step_height=int(m.group(1)) if m else None,
        direction=m.group(2) if m else None,
    )


def load_dataset(root: Path, column: str = DEFAULT_COLUMN,
                 exclude: list[str] | None = None, min_len: int = 1) -> list[Bag]:
    """root 以下の *_dataset.csv をすべて読む.

    exclude には bag の相対パスに対する fnmatch パターンを渡す (例: 'sakamichi/*')。
    """
    root = root.expanduser().resolve()
    bags = []
    for csv_path in sorted(root.rglob('*_dataset.csv')):
        rel = csv_path.parent.relative_to(root).as_posix()
        if any(fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(rel, pat.rstrip('/') + '/*')
               for pat in exclude or []):
            continue
        bag = load_bag(csv_path, root, column)
        if len(bag) < min_len:
            print(f'[skip] {rel}: {len(bag)} 行しかない')
            continue
        bags.append(bag)
    if not bags:
        raise SystemExit(f'{root} 以下に *_dataset.csv がない')
    return bags
