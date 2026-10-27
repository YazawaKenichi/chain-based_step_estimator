"""特徴量・RNN モデル・チェックポイント・逐次推論.

特徴量はすべて因果的 (過去と現在のサンプルだけを使う) なので、
bag 全体を一度に入れた結果と StreamingEstimator で 1 サンプルずつ入れた結果は一致する。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from tasks import Head

FEATURE_NAMES = ('level', 'diff')
CKPT_FORMAT = 1


# --- 特徴量 -------------------------------------------------------------

def causal_baseline(raw: np.ndarray, baseline_n: int) -> np.ndarray:
    """各時刻のベースライン。先頭 baseline_n サンプルの中央値 (揃うまではそれまでの中央値).

    hx711 の生値は bag ごとにオフセットが数万違うので、起動直後の値を基準に引く。
    """
    out = np.empty_like(raw, dtype=np.float64)
    n = min(baseline_n, len(raw))
    for t in range(n):
        out[t] = np.median(raw[:t + 1])
    out[n:] = out[n - 1] if n else 0.0
    return out


def raw_features(raw: np.ndarray, baseline_n: int) -> np.ndarray:
    """スケーリング前の特徴量 [T, 2]: (ベースラインからの差, 前サンプルとの差)."""
    level = raw - causal_baseline(raw, baseline_n)
    diff = np.diff(raw, prepend=raw[:1])
    return np.stack([level, diff], axis=1)


def fit_feature_config(raws: list[np.ndarray], baseline_n: int) -> dict:
    f = np.concatenate([raw_features(r, baseline_n) for r in raws])
    scale = f.std(axis=0)
    scale[scale == 0] = 1.0
    return {'baseline_n': baseline_n, 'names': list(FEATURE_NAMES), 'scale': scale.tolist()}


def compute_features(raw: np.ndarray, feat_cfg: dict) -> np.ndarray:
    f = raw_features(raw, feat_cfg['baseline_n']) / np.asarray(feat_cfg['scale'])
    return f.astype(np.float32)


# --- モデル -------------------------------------------------------------

class StepRNN(nn.Module):
    """単方向 RNN + 時刻ごとの分類ヘッド (ヘッドは複数持てる)."""

    def __init__(self, in_dim: int, heads: dict[str, int], rnn: str = 'gru',
                 hidden: int = 64, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        rnn_cls = {'gru': nn.GRU, 'lstm': nn.LSTM}[rnn]
        self.rnn = rnn_cls(in_dim, hidden, num_layers=layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.heads = nn.ModuleDict({name: nn.Linear(hidden, n) for name, n in heads.items()})

    def forward(self, x: torch.Tensor, h=None):
        """x: [B, T, F] → ({ヘッド名: logits [B, T, C]}, 隠れ状態)."""
        y, h = self.rnn(x, h)
        y = self.drop(y)
        return {name: head(y) for name, head in self.heads.items()}, h


# --- チェックポイント -----------------------------------------------------

def save_checkpoint(path: Path, model: StepRNN, model_cfg: dict, task_name: str,
                    heads: list[Head], feat_cfg: dict, **extra) -> None:
    torch.save({
        'format': CKPT_FORMAT,
        'task': task_name,
        'heads': [{'name': h.name, 'classes': list(h.classes)} for h in heads],
        'model_cfg': model_cfg,
        'features': feat_cfg,
        'state_dict': model.state_dict(),
        **extra,
    }, path)


def load_checkpoint(path: Path, device: str | torch.device = 'cpu'):
    """→ (model (eval モード), ckpt の dict, ヘッドの一覧)."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    heads = [Head(h['name'], tuple(h['classes'])) for h in ckpt['heads']]
    model = StepRNN(heads={h.name: len(h.classes) for h in heads}, **ckpt['model_cfg'])
    model.load_state_dict(ckpt['state_dict'])
    model.to(device).eval()
    return model, ckpt, heads


# --- 推論 ---------------------------------------------------------------

@torch.no_grad()
def predict_sequence(model: StepRNN, raw: np.ndarray, feat_cfg: dict,
                     device: str | torch.device = 'cpu') -> dict[str, np.ndarray]:
    """bag 全体を一度に推論する。→ {ヘッド名: クラス確率 [T, C]}."""
    x = torch.from_numpy(compute_features(raw, feat_cfg))[None].to(device)
    logits, _ = model(x)
    return {k: v[0].softmax(-1).cpu().numpy() for k, v in logits.items()}


class StreamingEstimator:
    """1 サンプルずつ推論する (オンライン用).

        est = StreamingEstimator('runs/xxx/best.ckpt')
        for value in hx711_stream:
            probs = est.update(value)   # {'contact': array([p_free, p_contact])}
    """

    def __init__(self, ckpt_path: str | Path, device: str = 'cpu'):
        self.device = device
        self.model, self.ckpt, self.heads = load_checkpoint(Path(ckpt_path), device)
        self.feat_cfg = self.ckpt['features']
        self._scale = np.asarray(self.feat_cfg['scale'])
        self.reset()

    def reset(self) -> None:
        self._first: list[float] = []
        self._baseline = 0.0
        self._prev: float | None = None
        self._h = None

    @torch.no_grad()
    def update(self, raw: float) -> dict[str, np.ndarray]:
        raw = float(raw)
        if len(self._first) < self.feat_cfg['baseline_n']:
            self._first.append(raw)
            self._baseline = float(np.median(self._first))
        diff = 0.0 if self._prev is None else raw - self._prev
        self._prev = raw
        f = np.array([raw - self._baseline, diff]) / self._scale
        x = torch.tensor(f, dtype=torch.float32, device=self.device)[None, None]
        logits, self._h = self.model(x, self._h)
        return {k: v[0, 0].softmax(-1).cpu().numpy() for k, v in logits.items()}
