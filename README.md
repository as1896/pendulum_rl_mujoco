# pendulum_rl_mujoco

## 概要

本リポジトリは倒立振子タスクを通して強化学習を学ぶための個人用メモです.</br>
目的は以下を理解することです.</br>

- MuJoCoの操作方法を理解すること.</br>
- 強化学習用のカスタム環境の実装方法を理解すること.</br>

## タスク

- **タスク**</br>
    倒立振子
- **状態(State)**</br>
    - 台車の位置[m]
    - 台車の速度[m/s]
    - 振子の角度[rad]
    - 振子の角速度[rad/s]</br>

- **行動**</br>
    台車のトルク [-50 - 50]

- **目的**</br>
    振子を直立状態に保つこと

## 実行環境

以下のライブラリを使用しました.</br>

- Python >= 3.10
- MuJoCo
- gymnasium
- stable-baselines3
- numpy
- matplotlib

## リポジトリ構成

```
pendulum_rl_mujoco
├── env
│   └── pendulum_env.py
├── model
│   ├── floor.xml
│   └── pendulum.xml
├── README.md
├── eval.py
├── requirements.txt
├── show_model.py
└── train.py
```

- env</br>
    学習用フレームワーク
- model</br>
    倒立振子 モデルファイル
- train.py</br>
    学習用スクリプト
- eval.py</br>
    評価用
- show_model.py</br>
    MuJoCoモデルの可視化用
- requirements.txt</br>
    ライブラリインストール用

## 実行方法

### 1. リポジトリの取得
```bash
git clone https://github.com/as1896/pendulum_rl_mujoco.git
cd pendulum_rl_mujoco
```

### 2. ライブラリのインストール

```bash
pip install -r requirements.txt
```

### 3. MuJoCoモデルの可視化(動作確認)

```bash
python3 show_model.py
```

### 4. 学習

```bash
python3 train.py
```

### 5. 評価

```bash
python3 eval.py
```

## 補足・注意事項

- 学習中は描画を無効化
- 本リポジトリは学習・検証目的であるため, 制御性能の最適化はまだできていません.

## 参考

- MuJoCo: https://mujoco.org/
- Stable Baselines3: https://github.com/DLR-RM/stable-baselines3

## ライセンス