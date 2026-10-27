"""bag_labeler のデータセットから RNN を学習してチェックポイントを作る.

    python3 train.py <データセットのディレクトリ> [--task contact_state] [--val-ratio 0.2]

--val-ratio 0 で全データ学習 (検証なし、最終エポックを保存)。
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from dataset import DEFAULT_COLUMN, Bag, load_dataset
from estimator import StepRNN, compute_features, fit_feature_config, load_checkpoint, save_checkpoint
from evaluate import evaluate, format_metrics, save_reports
from tasks import IGNORE, TASKS, get_task

SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    task_help = '\n'.join(f'  {t.name}: {t.description}' for t in TASKS.values())
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('dataset_dir', type=Path)
    p.add_argument('--task', default='contact_state', choices=list(TASKS), help=f'推定対象\n{task_help}')
    p.add_argument('--column', default=DEFAULT_COLUMN, help='入力に使う CSV の列')
    p.add_argument('--exclude', action='append', default=[],
                   help="除外する bag の相対パス (fnmatch。複数可。例: 'sakamichi/*')")
    p.add_argument('--val-ratio', type=float, default=0.2,
                   help='シナリオごとに bag 単位で検証に回す割合 (0 で全データ学習)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--rnn', default='gru', choices=['gru', 'lstm'])
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--layers', type=int, default=2)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--baseline-n', type=int, default=10,
                   help='ベースラインに使う先頭のサンプル数 (約 0.092 s/サンプル)')
    p.add_argument('--no-crop', action='store_true',
                   help='学習時に系列の先頭をランダムに切り落とさない '
                        '(切り落とすと「起動から何秒後か」で当てる癖を防げる)')
    p.add_argument('--no-class-weight', action='store_true', help='クラス頻度の逆数で損失を重み付けしない')
    p.add_argument('--tolerance', type=float, default=0.5,
                   help='接触検出の評価で、正解より何秒前までの検出を許すか')
    p.add_argument('--onset-k', type=int, default=3,
                   help='接触検出の評価で、陽性が何行続いたら検出とみなすか')
    p.add_argument('--out', type=Path, default=SCRIPT_DIR / 'runs', help='出力先の親ディレクトリ')
    p.add_argument('--name', default=None, help='出力ディレクトリ名 (既定: 日時_タスク名)')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--log-every', type=int, default=10)
    p.add_argument('--no-plot', action='store_true', help='検証 bag ごとのプロットを出さない')
    return p.parse_args()


def split_bags(bags: list[Bag], val_ratio: float, rng: np.random.Generator) -> tuple[list[Bag], list[Bag]]:
    """シナリオごとに bag 単位で分割する (各シナリオから同じ割合を検証に回す)."""
    if val_ratio <= 0:
        return bags, []
    by_scenario: dict[str, list[Bag]] = defaultdict(list)
    for b in bags:
        by_scenario[b.scenario].append(b)
    train, val = [], []
    for scenario in sorted(by_scenario):
        group = by_scenario[scenario]
        order = rng.permutation(len(group))
        n_val = max(1, round(len(group) * val_ratio)) if len(group) >= 2 else 0
        val += [group[i] for i in order[:n_val]]
        train += [group[i] for i in order[n_val:]]
    return train, val


def crop_start(bag: Bag, baseline_n: int, rng: np.random.Generator) -> int:
    """学習時に切り落とす先頭の行数。ベースラインが接触前に収まる範囲で選ぶ."""
    onset = bag.onset_index()
    limit = (onset if onset is not None else len(bag) // 2) - baseline_n
    return int(rng.integers(0, limit + 1)) if limit > 0 else 0


def make_batch(bags: list[Bag], targets: dict, head_names: list[str], feat_cfg: dict,
               rng: np.random.Generator | None, device):
    """パディングした (特徴量 [B, T, F], {ヘッド名: 教師 [B, T]}) を作る。rng を渡すと先頭をランダムに切る."""
    xs, ys = [], {h: [] for h in head_names}
    for b in bags:
        s = crop_start(b, feat_cfg['baseline_n'], rng) if rng is not None else 0
        xs.append(compute_features(b.raw[s:], feat_cfg))
        for h in head_names:
            ys[h].append(targets[b.rel][h][s:])
    T = max(len(x) for x in xs)
    x = np.zeros((len(bags), T, xs[0].shape[1]), dtype=np.float32)
    y = {h: np.full((len(bags), T), IGNORE, dtype=np.int64) for h in head_names}
    for i, xi in enumerate(xs):
        x[i, :len(xi)] = xi
        for h in head_names:
            y[h][i, :len(xi)] = ys[h][i]
    return (torch.from_numpy(x).to(device),
            {h: torch.from_numpy(v).to(device) for h, v in y.items()})


def class_weights(bags: list[Bag], targets: dict, head: str, n_classes: int) -> torch.Tensor:
    y = np.concatenate([targets[b.rel][head] for b in bags])
    counts = np.bincount(y[y != IGNORE], minlength=n_classes).astype(np.float64)
    w = np.where(counts > 0, counts.sum() / (n_classes * np.maximum(counts, 1)), 1.0)
    return torch.tensor(w, dtype=torch.float32)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    task = get_task(args.task)

    bags = load_dataset(args.dataset_dir, args.column, args.exclude, min_len=args.baseline_n + 1)
    heads = task.build_heads(bags)
    head_names = [h.name for h in heads]
    train_bags, val_bags = split_bags(bags, args.val_ratio, rng)
    targets = {b.rel: task.make_targets(b, heads) for b in bags}
    feat_cfg = fit_feature_config([b.raw for b in train_bags], args.baseline_n)

    run_name = args.name or f'{time.strftime("%Y%m%d-%H%M%S")}_{task.name}'
    out_dir = args.out / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    (out_dir / 'config.json').write_text(json.dumps(config, indent=2, ensure_ascii=False))
    (out_dir / 'split.json').write_text(json.dumps(
        {'train': [b.rel for b in train_bags], 'val': [b.rel for b in val_bags]}, indent=2))

    print(f'task={task.name} ({task.description})')
    print(f'bags: train={len(train_bags)} val={len(val_bags)}  heads: '
          + ', '.join(f'{h.name}{list(h.classes)}' for h in heads))
    print(f'out: {out_dir}')

    device = torch.device(args.device)
    model_cfg = {'in_dim': len(feat_cfg['names']), 'rnn': args.rnn, 'hidden': args.hidden,
                 'layers': args.layers, 'dropout': args.dropout}
    model = StepRNN(heads={h.name: len(h.classes) for h in heads}, **model_cfg).to(device)
    losses = {}
    for h in heads:
        w = None if args.no_class_weight else class_weights(train_bags, targets, h.name, len(h.classes)).to(device)
        losses[h.name] = nn.CrossEntropyLoss(weight=w, ignore_index=IGNORE)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    def batch_loss(x, y):
        logits, _ = model(x)
        total = 0.0
        for name, fn in losses.items():
            if (y[name] != IGNORE).any():
                total = total + fn(logits[name].flatten(0, 1), y[name].flatten())
        return total

    ckpt_extra = {'train_bags': [b.rel for b in train_bags], 'val_bags': [b.rel for b in val_bags],
                  'args': config}
    history = []
    best_score, best_epoch, best_metrics = -1.0, None, None
    main_head = task.onset_head
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(len(train_bags))
        train_loss, n = 0.0, 0
        for i in range(0, len(order), args.batch_size):
            batch = [train_bags[j] for j in order[i:i + args.batch_size]]
            x, y = make_batch(batch, targets, head_names, feat_cfg,
                              None if args.no_crop else rng, device)
            loss = batch_loss(x, y)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += loss.item() * len(batch)
            n += len(batch)
        sched.step()
        row = {'epoch': epoch, 'train_loss': train_loss / n}

        if val_bags:
            model.eval()
            with torch.no_grad():
                x, y = make_batch(val_bags, targets, head_names, feat_cfg, None, device)
                row['val_loss'] = float(batch_loss(x, y))
            metrics, _ = evaluate(model, val_bags, task, heads, feat_cfg, device, args.tolerance, args.onset_k)
            for name, m in metrics['heads'].items():
                row[f'val_{name}_macro_f1'] = m['macro_f1']
            row.update({f'val_onset_{k}': v for k, v in metrics['onset']['counts'].items()})
            score = metrics['heads'][main_head]['macro_f1']
            if score > best_score:
                best_score, best_epoch, best_metrics = score, epoch, metrics
                save_checkpoint(out_dir / 'best.ckpt', model, model_cfg, task.name, heads, feat_cfg,
                                epoch=epoch, val_metrics=_json_safe(metrics), **ckpt_extra)
        history.append(row)
        if epoch % args.log_every == 0 or epoch == 1 or epoch == args.epochs:
            print(' '.join(f'{k}={v:.4f}' if isinstance(v, float) else f'{k}={v}' for k, v in row.items()))

    save_checkpoint(out_dir / 'last.ckpt', model, model_cfg, task.name, heads, feat_cfg,
                    epoch=args.epochs, **ckpt_extra)
    with open(out_dir / 'history.csv', 'w', newline='') as f:
        keys = list(dict.fromkeys(k for r in history for k in r))
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(history)

    if val_bags:
        # 検証の結果は best.ckpt の時点のもの
        best_model, _, _ = load_checkpoint(out_dir / 'best.ckpt', device)
        metrics, preds = evaluate(best_model, val_bags, task, heads, feat_cfg, device, args.tolerance, args.onset_k)
        (out_dir / 'metrics.json').write_text(json.dumps(
            {'best_epoch': best_epoch, **_json_safe(metrics)}, indent=2, ensure_ascii=False))
        save_reports(out_dir / 'val', val_bags, preds, metrics, task, heads, feat_cfg, plot=not args.no_plot)
        print(f'\n=== val (best epoch {best_epoch}, {main_head} macro_f1={best_score:.3f}) ===')
        print(format_metrics(metrics))
        print(f'\nckpt: {out_dir / "best.ckpt"} (最終エポック: {out_dir / "last.ckpt"})')
    else:
        print(f'\nckpt: {out_dir / "last.ckpt"}')


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


if __name__ == '__main__':
    main()
