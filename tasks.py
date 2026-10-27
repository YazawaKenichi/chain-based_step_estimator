"""学習タスク (何を推定するか) の定義.

タスクは「出力ヘッドの一覧を作る関数」と「bag のラベルから各ヘッドの教師を作る関数」の組。
新しいタスクを足すときは 2 つの関数を書いて register() するだけでよい。
教師の IGNORE の行は損失・評価から外れる (行を消すのではなく無視するので、時系列は途切れない)。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from dataset import Bag

IGNORE = -100


@dataclass(frozen=True)
class Head:
    name: str
    classes: tuple[str, ...]


@dataclass(frozen=True)
class Task:
    name: str
    description: str
    build_heads: Callable[[list[Bag]], list[Head]]
    make_targets: Callable[[Bag, list[Head]], dict[str, np.ndarray]]
    # 接触検出の評価 (検出遅れ・誤検出) に使うヘッド。クラス 0 以外を「接触」とみなす
    onset_head: str


TASKS: dict[str, Task] = {}


def register(task: Task) -> Task:
    TASKS[task.name] = task
    return task


def get_task(name: str) -> Task:
    if name not in TASKS:
        raise KeyError(f'未知のタスク {name} (候補: {", ".join(TASKS)})')
    return TASKS[name]


def _fixed_heads(*heads: Head) -> Callable[[list[Bag]], list[Head]]:
    return lambda bags: list(heads)


# --- contact_state: 0 vs 1+2 ---------------------------------------------

def _contact_state_targets(bag: Bag, heads: list[Head]) -> dict[str, np.ndarray]:
    return {'contact': (bag.label != 0).astype(np.int64)}


register(Task(
    name='contact_state',
    description='接触しているか (0 vs 1+2)',
    build_heads=_fixed_heads(Head('contact', ('free', 'contact'))),
    make_targets=_contact_state_targets,
    onset_head='contact',
))


# --- contact_onset: 0 vs 1 (label 2 は無視) --------------------------------

def _contact_onset_targets(bag: Bag, heads: list[Head]) -> dict[str, np.ndarray]:
    y = (bag.label == 1).astype(np.int64)
    y[bag.label == 2] = IGNORE
    return {'contact': y}


register(Task(
    name='contact_onset',
    description='走行中の接触を検出 (0 vs 1, label 2 は無視)',
    build_heads=_fixed_heads(Head('contact', ('free', 'contact'))),
    make_targets=_contact_onset_targets,
    onset_head='contact',
))


# --- three_class: 0 / 1 / 2 ---------------------------------------------

def _three_class_targets(bag: Bag, heads: list[Head]) -> dict[str, np.ndarray]:
    return {'phase': bag.label.astype(np.int64)}


register(Task(
    name='three_class',
    description='ラベルをそのまま 3 クラス分類 (0 / 1 / 2)',
    build_heads=_fixed_heads(Head('phase', ('free', 'contact_moving', 'contact_stopped'))),
    make_targets=_three_class_targets,
    onset_head='phase',
))


# --- step_multitask: 接触 + 段差の高さ + 向き ------------------------------

DIRECTIONS = ('up', 'down')


def _step_multitask_heads(bags: list[Bag]) -> list[Head]:
    heights = sorted({b.step_height for b in bags if b.step_height is not None})
    return [
        Head('contact', ('free', 'contact')),
        Head('height', tuple(str(h) for h in heights)),
        Head('direction', DIRECTIONS),
    ]


def _step_multitask_targets(bag: Bag, heads: list[Head]) -> dict[str, np.ndarray]:
    contact = bag.label != 0
    height = np.full(len(bag), IGNORE, dtype=np.int64)
    direction = np.full(len(bag), IGNORE, dtype=np.int64)
    # 高さ・向きは接触している行だけで学習する (接触前は段差の情報がないため)
    heights = next(h for h in heads if h.name == 'height').classes
    if bag.step_height is not None and str(bag.step_height) in heights:
        height[contact] = heights.index(str(bag.step_height))
        direction[contact] = DIRECTIONS.index(bag.direction)
    return {'contact': contact.astype(np.int64), 'height': height, 'direction': direction}


register(Task(
    name='step_multitask',
    description='接触 (0 vs 1+2) + 接触中の段差の高さ・向き (ディレクトリ名から)',
    build_heads=_step_multitask_heads,
    make_targets=_step_multitask_targets,
    onset_head='contact',
))
