#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
硬约束 GNN 超参数批量试验：依次调用 train.py，将各实验的测试集 MAE/RMSE 汇总到 CSV。

用法（在项目根目录、已激活 GPU 环境）:
  python tune_hard_gnn.py --epochs 120
  python tune_hard_gnn.py --epochs 80 --dry-run   # 只打印将执行的命令

挑最优一行后，用 paper_experiments 只换 hard 权重评估多 horizon:
  python paper_experiments.py --checkpoint-hard checkpoints/t_hard_lr1e3/best_model_hard.pth
"""
from __future__ import annotations

import argparse
from typing import List, Tuple
import csv
import os
import pickle
import subprocess
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__)) or "."

# (实验子目录名, train.py 额外参数列表) — 可按需增删
DEFAULT_TRIALS: List[Tuple[str, List[str]]] = [
    ("t_hard_lr3e4", ["--lr", "0.0003"]),
    ("t_hard_lr1e3", ["--lr", "0.001"]),
    ("t_hard_dropout25", ["--dropout", "0.25"]),
    ("t_hard_dropout08", ["--dropout", "0.08"]),
    ("t_hard_wd1e4", ["--weight-decay", "0.0001"]),
    ("t_hard_batch256", ["--batch-size", "256"]),
    ("t_hard_lmae08", ["--lambda-mae", "0.8"]),
    ("t_hard_lout02", ["--lambda-outbreak", "0.2"]),
]


def read_test_metrics(exp_name: str) -> dict | None:
    pkl = os.path.join(ROOT, "checkpoints", exp_name, "test_results_hard.pkl")
    if not os.path.isfile(pkl):
        return None
    with open(pkl, "rb") as f:
        d = pickle.load(f)
    return {
        "test_mae": d.get("test_mae"),
        "test_rmse": d.get("test_rmse"),
        "test_loss": d.get("test_loss"),
    }


def main():
    parser = argparse.ArgumentParser(description="硬约束 GNN 超参试验")
    parser.add_argument("--epochs", type=int, default=120, help="每个试验的训练轮数（省时可 80）")
    parser.add_argument("--dry-run", action="store_true", help="不执行训练，只打印命令")
    parser.add_argument(
        "--log",
        type=str,
        default=os.path.join("paper_results", "tuning_hard_runs.csv"),
        help="汇总 CSV 路径",
    )
    args = parser.parse_args()

    if args.dry_run:
        for exp_name, extra in DEFAULT_TRIALS:
            cmd = [
                sys.executable,
                "train.py",
                "--constraint",
                "hard",
                "--exp-name",
                exp_name,
                "--num_epochs",
                str(args.epochs),
            ] + extra
            print(" ".join(cmd))
        print("\n(dry-run 结束，未写入 CSV)")
        return

    rows = []
    for exp_name, extra in DEFAULT_TRIALS:
        cmd = [
            sys.executable,
            "train.py",
            "--constraint",
            "hard",
            "--exp-name",
            exp_name,
            "--num_epochs",
            str(args.epochs),
        ] + extra
        print("\n" + "=" * 70)
        print("RUN:", " ".join(cmd))
        r = subprocess.run(cmd, cwd=ROOT)
        status = "ok" if r.returncode == 0 else f"fail_{r.returncode}"
        m = read_test_metrics(exp_name) if status == "ok" else None
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "exp_name": exp_name,
            "status": status,
            "epochs": args.epochs,
            "extra_args": " ".join(extra),
            "test_mae": m["test_mae"] if m else "",
            "test_rmse": m["test_rmse"] if m else "",
            "test_loss": m["test_loss"] if m else "",
            "checkpoint": os.path.join("checkpoints", exp_name, "best_model_hard.pth"),
        }
        rows.append(row)
        print(f"  -> {status}  test_mae={row['test_mae']}  test_rmse={row['test_rmse']}")

    os.makedirs(os.path.dirname(os.path.abspath(args.log)) or ".", exist_ok=True)
    fieldnames = [
        "timestamp",
        "exp_name",
        "status",
        "epochs",
        "extra_args",
        "test_mae",
        "test_rmse",
        "test_loss",
        "checkpoint",
    ]
    write_header = not os.path.isfile(args.log)
    with open(args.log, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            w.writeheader()
        for row in rows:
            w.writerow(row)

    print(f"\n已追加记录: {args.log}")
    print("按 test_mae / test_rmse 挑最优后，多 horizon 评估:")
    print("  python paper_experiments.py --checkpoint-hard checkpoints/<exp_name>/best_model_hard.pth")


if __name__ == "__main__":
    main()
