# chain-based_step_estimator

[bag_labeler](../bag_labeler) が出力した `*_dataset.csv` (hx711 の生値 + ラベル) から、
チェーンと段差の接触を推定する RNN (単方向 GRU/LSTM) を学習してチェックポイントを作る。

## セットアップ

```bash
pip install -r requirements.txt
```

## 学習

```bash
python3 train.py ~/raspicat2/rosbags/chain/tsudanuma/tooi/dataset_circuit_v2
./train.bash                       # train.bash の TARGET_DIR を使う
./train.bash <ディレクトリ> --task step_multitask
```

指定ディレクトリ以下の `*_dataset.csv` を再帰的に読む。シナリオ (bag の親ディレクトリ、
例: `ijo/step_150_up_choku_02`) ごとに bag 単位で train/val に分ける。

| オプション | 既定 | 内容 |
|---|---|---|
| `--task` | `contact_state` | 推定対象 (下表) |
| `--val-ratio` | `0.2` | 検証に回す bag の割合。**`0` で全データ学習** (検証なし、最終エポックを保存) |
| `--exclude` | | 除外する bag の相対パス (fnmatch、複数可)。例: `--exclude 'sakamichi/*'` |
| `--epochs` / `--batch-size` / `--lr` | 200 / 16 / 1e-3 | |
| `--rnn` / `--hidden` / `--layers` / `--dropout` | gru / 64 / 2 / 0.1 | |
| `--baseline-n` | `10` | ベースラインに使う先頭のサンプル数 (約 0.92 s) |
| `--no-crop` | | 学習時に系列の先頭をランダムに切り落とすのをやめる |
| `--no-class-weight` | | クラス頻度の逆数による損失の重み付けをやめる |
| `--seed` | `0` | 分割と初期値の乱数 |

`python3 train.py -h` で全オプションを表示する。

### タスク

| `--task` | ヘッド | 教師 |
|---|---|---|
| `contact_state` | contact (2) | 0 vs 1+2 (接触しているか) |
| `contact_onset` | contact (2) | 0 vs 1。label 2 の行は損失・評価から外す |
| `three_class` | phase (3) | label 0 / 1 / 2 をそのまま |
| `step_multitask` | contact (2), height, direction (2) | contact は 0 vs 1+2。高さ・向きはディレクトリ名 (`step_<高さ>_<up\|down>`) から取り、接触中の行だけで学習する |

タスクを足すときは `tasks.py` に「ヘッドを作る関数」と「bag から教師を作る関数」を書いて
`register(Task(...))` する。教師を `IGNORE` にした行は消すのではなく損失から外すので、時系列は途切れない。

## 出力

`runs/<日時>_<タスク>/` (`--out`, `--name` で変更可) に保存する。

| ファイル | 内容 |
|---|---|
| `best.ckpt` | 検証で onset ヘッドの macro F1 が最高のエポック (検証ありのときだけ) |
| `last.ckpt` | 最終エポック |
| `config.json` / `split.json` | 引数 / train・val の bag 一覧 |
| `history.csv` | エポックごとの損失と検証の指標 |
| `metrics.json` | best.ckpt での検証の指標 |
| `val/predictions/` | 検証 bag ごとの確率 CSV とプロット |

ckpt には重みのほか、タスク名・ヘッド・特徴量の正規化パラメータ・モデル構成・分割が入っているので、
ckpt 単体で推論できる。

### 評価指標

- ヘッドごとの混同行列 (行: 正解, 列: 予測)、recall / precision、macro F1
- **onset** (接触検出): 先頭 `baseline_n` 行を除き、陽性が `--onset-k` 行 (既定 3) 続いた時点を検出とする
  - `ok`: 正解の接触時刻の `--tolerance` 秒 (既定 0.5) 前以降に検出。`delay` = 検出 - 正解
  - `early`: それより前に検出 (誤検出) / `miss`: 検出しなかった
  - 接触しない bag (平地・坂): `clean` / `false_alarm`

## 推論

```bash
python3 predict.py runs/xxx/best.ckpt <データセットのディレクトリ> [--only-val]
```

ckpt のディレクトリの `predict_<データセット名>/` に確率 CSV・プロット・`metrics.json` を出す。
全データで学習した ckpt を別のデータセットで評価するときに使う。

オンライン (1 サンプルずつ) では `StreamingEstimator` を使う。特徴量は因果的なので、
bag 全体を一度に推論したときと同じ値になる。

```python
from estimator import StreamingEstimator
est = StreamingEstimator('runs/xxx/best.ckpt')
probs = est.update(hx711_raw)   # {'contact': array([p_free, p_contact])}
```

## 特徴量

hx711 の生値は bag ごとにオフセットが数万違うので、そのままは使わない。

- `level`: 生値 - ベースライン (先頭 `baseline_n` サンプルの中央値。揃うまではそれまでの中央値)
- `diff`: 前サンプルとの差

どちらも学習データの標準偏差で割る。起動直後 (先頭 `baseline_n` サンプル) は
ロボットが止まっていて段差に接触していない前提。

## ファイル構成

| ファイル | 内容 |
|---|---|
| `train.py` | 学習 |
| `predict.py` | ckpt で推論・評価 |
| `tasks.py` | タスク (推定対象) の定義 |
| `dataset.py` | CSV の読み込み |
| `estimator.py` | 特徴量・モデル・ckpt・逐次推論 |
| `evaluate.py` | 評価指標・プロット |
| `train.bash` | 起動スクリプト |
