"""学習済みチェックポイントでデータセットを推論・評価する.

    python3 predict.py <ckpt> <データセットのディレクトリ> [--out DIR]

bag ごとの確率の CSV とプロット、metrics.json を出す。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from dataset import load_dataset
from estimator import load_checkpoint
from evaluate import evaluate, format_metrics, save_reports
from tasks import get_task


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('ckpt', type=Path)
    p.add_argument('dataset_dir', type=Path)
    p.add_argument('--out', type=Path, default=None, help='既定: <ckpt のディレクトリ>/predict_<データセット名>')
    p.add_argument('--exclude', action='append', default=[])
    p.add_argument('--only-val', action='store_true', help='ckpt の学習時に検証に回した bag だけ評価する')
    p.add_argument('--tolerance', type=float, default=0.5)
    p.add_argument('--onset-k', type=int, default=3)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--no-plot', action='store_true')
    args = p.parse_args()

    model, ckpt, heads = load_checkpoint(args.ckpt, args.device)
    task = get_task(ckpt['task'])
    feat_cfg = ckpt['features']
    column = ckpt.get('args', {}).get('column')
    bags = load_dataset(args.dataset_dir, *([column] if column else []), exclude=args.exclude)
    if args.only_val:
        val = set(ckpt.get('val_bags', []))
        bags = [b for b in bags if b.rel in val]

    out_dir = args.out or args.ckpt.parent / f'predict_{args.dataset_dir.expanduser().resolve().name}'
    metrics, preds = evaluate(model, bags, task, heads, feat_cfg, args.device, args.tolerance, args.onset_k)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    save_reports(out_dir, bags, preds, metrics, task, heads, feat_cfg, plot=not args.no_plot)
    print(f'task={task.name} bags={len(bags)}')
    print(format_metrics(metrics))
    print(f'\nout: {out_dir}')


if __name__ == '__main__':
    main()
