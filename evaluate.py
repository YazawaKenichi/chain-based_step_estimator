"""評価指標とプロット (train.py / predict.py 共通)."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from dataset import Bag
from estimator import StepRNN, predict_sequence, raw_features
from tasks import IGNORE, Head, Task


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, classes: tuple[str, ...]) -> dict:
    keep = y_true != IGNORE
    y_true, y_pred = y_true[keep], y_pred[keep]
    n = len(classes)
    cm = np.zeros((n, n), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    per_class = {}
    f1s = []
    for i, c in enumerate(classes):
        tp = cm[i, i]
        prec = tp / cm[:, i].sum() if cm[:, i].sum() else 0.0
        rec = tp / cm[i].sum() if cm[i].sum() else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class[c] = {'precision': float(prec), 'recall': float(rec), 'f1': float(f1),
                        'support': int(cm[i].sum())}
        if cm[i].sum():
            f1s.append(f1)
    return {
        'accuracy': float(np.trace(cm) / cm.sum()) if cm.sum() else 0.0,
        'macro_f1': float(np.mean(f1s)) if f1s else 0.0,
        'per_class': per_class,
        'confusion': cm.tolist(),  # 行: 正解, 列: 予測
        'classes': list(classes),
    }


def detect_onset(pred_contact: np.ndarray, warmup: int, k: int) -> int | None:
    """陽性が k 行続いたら検出とし、その k 行目の行を返す (因果的なのでオンラインでも同じ規則で使える).

    先頭 warmup 行 (ベースラインが決まるまで) は判定しない。
    """
    run = 0
    for t in range(warmup, len(pred_contact)):
        run = run + 1 if pred_contact[t] else 0
        if run >= k:
            return t
    return None


def onset_result(bag: Bag, pred_i: int | None, tolerance: float) -> dict:
    """1 bag の接触検出の結果 (pred_i は detect_onset() の結果).

    status:
      ok          接触後 (または正解の tolerance 秒前以降) に初めて検出した
      early       正解の接触時刻より tolerance 秒以上前に検出した (誤検出)
      miss        接触したのに一度も検出しなかった
      clean       接触しない bag で一度も検出しなかった
      false_alarm 接触しない bag で検出した
    delay は検出時刻 - 正解の接触時刻 [s]。
    """
    true_i = bag.onset_index()
    true_t = float(bag.elapsed[true_i]) if true_i is not None else None
    pred_t = float(bag.elapsed[pred_i]) if pred_i is not None else None
    if true_i is None:
        status = 'clean' if pred_i is None else 'false_alarm'
        delay = None
    elif pred_i is None:
        status, delay = 'miss', None
    else:
        delay = pred_t - true_t
        status = 'early' if delay < -tolerance else 'ok'
    return {'bag': bag.rel, 'true_onset': true_t, 'pred_onset': pred_t, 'delay': delay, 'status': status}


def summarize_onsets(rows: list[dict]) -> dict:
    counts = {s: sum(r['status'] == s for r in rows)
              for s in ('ok', 'early', 'miss', 'clean', 'false_alarm')}
    delays = np.array([r['delay'] for r in rows if r['status'] == 'ok'])
    return {
        'counts': counts,
        'delay_mean': float(delays.mean()) if len(delays) else None,
        'delay_median': float(np.median(delays)) if len(delays) else None,
        'delay_max': float(delays.max()) if len(delays) else None,
    }


def evaluate(model: StepRNN, bags: list[Bag], task: Task, heads: list[Head], feat_cfg: dict,
             device, tolerance: float = 0.5, onset_k: int = 3
             ) -> tuple[dict, dict[str, dict[str, np.ndarray]]]:
    """→ (指標, {bag: {ヘッド名: 確率 [T, C]}})."""
    model.eval()
    preds = {b.rel: predict_sequence(model, b.raw, feat_cfg, device) for b in bags}
    targets = {b.rel: task.make_targets(b, heads) for b in bags}
    metrics = {'heads': {}}
    for h in heads:
        y_true = np.concatenate([targets[b.rel][h.name] for b in bags])
        y_pred = np.concatenate([preds[b.rel][h.name].argmax(-1) for b in bags])
        metrics['heads'][h.name] = classification_metrics(y_true, y_pred, h.classes)
    warmup = feat_cfg['baseline_n']
    rows = [onset_result(b, detect_onset(preds[b.rel][task.onset_head].argmax(-1) != 0, warmup, onset_k),
                         tolerance) for b in bags]
    metrics['onset'] = summarize_onsets(rows)
    metrics['onset_per_bag'] = rows
    return metrics, preds


def format_metrics(metrics: dict) -> str:
    lines = []
    for name, m in metrics['heads'].items():
        lines.append(f'[{name}] acc={m["accuracy"]:.3f} macro_f1={m["macro_f1"]:.3f}')
        w = max(len(c) for c in m['classes'])
        lines.append('  ' + ' ' * w + '  ' + ' '.join(f'{c[:8]:>8}' for c in m['classes'])
                     + '   recall  precision')
        for c, row in zip(m['classes'], m['confusion']):
            pc = m['per_class'][c]
            lines.append(f'  {c:>{w}}  ' + ' '.join(f'{v:>8}' for v in row)
                         + f'   {pc["recall"]:.3f}   {pc["precision"]:.3f}')
    o = metrics['onset']
    lines.append('[onset] ' + ' '.join(f'{k}={v}' for k, v in o['counts'].items()))
    if o['delay_mean'] is not None:
        lines.append(f'  delay mean={o["delay_mean"]:+.3f}s median={o["delay_median"]:+.3f}s '
                     f'max={o["delay_max"]:+.3f}s')
    return '\n'.join(lines)


def write_predictions_csv(path: Path, bag: Bag, probs: dict[str, np.ndarray], heads: list[Head]) -> None:
    cols = [f'{h.name}/{c}' for h in heads for c in h.classes]
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['elapsed', 'raw', 'label'] + cols)
        p = np.concatenate([probs[h.name] for h in heads], axis=1)
        for t in range(len(bag)):
            w.writerow([f'{bag.elapsed[t]:.6f}', int(bag.raw[t]), int(bag.label[t])]
                       + [f'{v:.4f}' for v in p[t]])


LABEL_COLORS = {1: '#f4a261', 2: '#e76f51'}


def plot_bag(path: Path, bag: Bag, probs: dict[str, np.ndarray], task: Task, heads: list[Head],
             feat_cfg: dict, onset: dict | None = None) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    t = bag.elapsed
    fig, axes = plt.subplots(len(heads) + 1, 1, figsize=(10, 2.2 * (len(heads) + 1)), sharex=True)
    level = raw_features(bag.raw, feat_cfg['baseline_n'])[:, 0]
    ax = axes[0]
    ax.plot(t, level, lw=1, color='k')
    ax.set_ylabel('raw - baseline')
    targets = task.make_targets(bag, heads)
    for ax in axes:
        for lab, color in LABEL_COLORS.items():
            ax.fill_between(t, 0, 1, where=bag.label == lab, color=color, alpha=0.25,
                            transform=ax.get_xaxis_transform(), step='post', lw=0)
        if onset and onset['pred_onset'] is not None:
            ax.axvline(onset['pred_onset'], color='tab:blue', ls='--', lw=1)
    for ax, h in zip(axes[1:], heads):
        p = probs[h.name]
        for i, c in enumerate(h.classes):
            if len(h.classes) == 2 and i == 0:
                continue  # 2 クラスは陽性側だけ描く
            ax.plot(t, p[:, i], lw=1, label=c)
        y = targets[h.name].astype(float)
        y[y == IGNORE] = np.nan
        if len(h.classes) == 2:
            ax.plot(t, y, color='k', lw=1, ls=':', label='target')
        else:
            ax.plot(t, y / (len(h.classes) - 1), color='k', lw=1, ls=':', label='target (scaled)')
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel(h.name)
        ax.legend(loc='upper left', fontsize=7, ncol=len(h.classes) + 1)
    axes[-1].set_xlabel('elapsed [s]')
    title = bag.rel
    if onset:
        d = f' delay={onset["delay"]:+.2f}s' if onset['delay'] is not None else ''
        title += f'  [{onset["status"]}{d}]'
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def save_reports(out_dir: Path, bags: list[Bag], preds: dict, metrics: dict, task: Task,
                 heads: list[Head], feat_cfg: dict, plot: bool = True) -> None:
    """bag ごとの予測 CSV と (plot=True なら) プロットを out_dir に書く."""
    pred_dir = out_dir / 'predictions'
    pred_dir.mkdir(parents=True, exist_ok=True)
    onsets = {r['bag']: r for r in metrics['onset_per_bag']}
    for b in bags:
        stem = b.rel.replace('/', '__')
        write_predictions_csv(pred_dir / f'{stem}.csv', b, preds[b.rel], heads)
        if plot:
            plot_bag(pred_dir / f'{stem}.png', b, preds[b.rel], task, heads, feat_cfg, onsets[b.rel])
