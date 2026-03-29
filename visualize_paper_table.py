#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
读取 paper_results/table1_metrics_by_horizon.csv，生成论文表风格图与多指标柱状/折线图。
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

CSV_PATH = os.path.join("paper_results", "table1_metrics_by_horizon.csv")
OUT_DIR = os.path.join("paper_results", "figures")

MODEL_ORDER = ["ARIMA", "LSTM", "Base", "Proposed", "Hard"]
LABELS = {
    "ARIMA": "ARIMA",
    "LSTM": "LSTM",
    "Base": "Base Model",
    "Proposed": "Proposed (soft)",
    "Hard": "Proposed (hard)",
}
COLORS = {
    "ARIMA": "#2ecc71",
    "LSTM": "#95a5a6",
    "Base": "#3498db",
    "Proposed": "#e74c3c",
    "Hard": "#9b59b6",
}


def load_df() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    df["model"] = df["model"].str.strip()
    return df


def model_order_present(df: pd.DataFrame) -> list[str]:
    """保持 MODEL_ORDER 顺序，只保留 CSV 里出现的模型（兼容旧 CSV 无 Hard）。"""
    have = set(df["model"].unique())
    return [m for m in MODEL_ORDER if m in have]


def plot_grouped_bars(df: pd.DataFrame, save_path: str) -> None:
    """四指标 × 四个 horizon：每个指标一子图，横轴为 horizon，多模型分组柱。"""
    horizons = sorted(df["horizon"].unique())
    metrics = [("RMSE", "RMSE"), ("MAE", "MAE"), ("MAPE", "MAPE (%)"), ("RAE", "RAE")]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.ravel()
    n_h = len(horizons)
    models_plot = model_order_present(df)
    n_m = len(models_plot)
    width = min(0.18, 0.8 / max(n_m, 1))
    x = np.arange(n_h)

    for ax, (col, title) in zip(axes, metrics):
        for mi, model in enumerate(models_plot):
            vals = []
            for h in horizons:
                row = df[(df["model"] == model) & (df["horizon"] == h)]
                v = row[col].values[0] if len(row) else np.nan
                if col == "MAPE" and (not np.isfinite(v) or v <= 0):
                    v = np.nan
                vals.append(v)
            plot_vals = [np.nan_to_num(v, nan=0.0) for v in vals]
            offset = (mi - (n_m - 1) / 2) * width
            ax.bar(x + offset, plot_vals, width, label=LABELS[model], color=COLORS[model], alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{int(h)}-day" for h in horizons])
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Forecast horizon")
        ax.grid(True, axis="y", alpha=0.3)
        if col == "MAPE":
            ax.set_yscale("log")
            ax.set_ylabel("MAPE (log scale, %)")
        else:
            ax.set_ylabel(col)

    handles = [mpatches.Patch(color=COLORS[m], label=LABELS[m], alpha=0.9) for m in models_plot]
    ncol = min(5, max(len(handles), 1))
    fig.legend(handles=handles, loc="upper center", ncol=ncol, bbox_to_anchor=(0.5, 1.04), fontsize=9)
    plt.suptitle("Test-set metrics by model and horizon (Eq. 27–30)", fontsize=14, y=1.03)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_line_rmse_mae(df: pd.DataFrame, save_path: str) -> None:
    """RMSE / MAE 随 horizon 变化折线图（便于看趋势）。"""
    horizons = sorted(df["horizon"].unique())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for model in model_order_present(df):
        sub = df[df["model"] == model]
        if sub.empty:
            continue
        sub = sub.set_index("horizon")
        rmse = [float(sub.loc[h, "RMSE"]) if h in sub.index and np.isfinite(sub.loc[h, "RMSE"]) else np.nan for h in horizons]
        mae = [float(sub.loc[h, "MAE"]) if h in sub.index and np.isfinite(sub.loc[h, "MAE"]) else np.nan for h in horizons]
        ax1.plot(horizons, rmse, "o-", label=LABELS[model], color=COLORS[model], linewidth=2, markersize=7)
        ax2.plot(horizons, mae, "o-", label=LABELS[model], color=COLORS[model], linewidth=2, markersize=7)
    ax1.set_xlabel("Horizon (days)")
    ax1.set_ylabel("RMSE")
    ax1.set_title("RMSE vs horizon")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax2.set_xlabel("Horizon (days)")
    ax2.set_ylabel("MAE")
    ax2.set_title("MAE vs horizon")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    plt.suptitle("Error scaling with forecast horizon", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_table_image(df: pd.DataFrame, save_path: str) -> None:
    """将数值表渲染为一张 PNG（便于插入幻灯片）。"""
    horizons = [3, 7, 14, 30]
    metrics = ["RMSE", "MAE", "MAPE", "RAE"]
    # 左：3&7 天各 4 列指标；右：14&30 天
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(16, 6))
    axL.axis("off")
    axR.axis("off")

    def build_block(h_list):
        rows = [[""] + [f"{h}d {m}" for h in h_list for m in metrics]]
        for model in model_order_present(df):
            line = [LABELS[model]]
            for h in h_list:
                r = df[(df["model"] == model) & (df["horizon"] == h)]
                if len(r) == 0:
                    line.extend(["—"] * 4)
                    continue
                row = r.iloc[0]
                if not np.isfinite(row["RMSE"]):
                    line.extend(["—"] * 4)
                    continue
                line.append(f"{row['RMSE']:.4f}")
                line.append(f"{row['MAE']:.4f}")
                line.append(f"{row['MAPE']:.2f}")
                line.append(f"{row['RAE']:.4f}")
            rows.append(line)
        return rows

    rows_left = build_block([3, 7])
    rows_right = build_block([14, 30])

    tblL = axL.table(
        cellText=rows_left,
        loc="center",
        cellLoc="center",
        colWidths=[0.12] + [0.09] * 8,
    )
    tblL.auto_set_font_size(False)
    tblL.set_fontsize(8)
    tblL.scale(1.0, 1.8)
    axL.set_title("Horizons 3 & 7 days", fontsize=12, fontweight="bold", pad=12)

    tblR = axR.table(
        cellText=rows_right,
        loc="center",
        cellLoc="center",
        colWidths=[0.12] + [0.09] * 8,
    )
    tblR.auto_set_font_size(False)
    tblR.set_fontsize(8)
    tblR.scale(1.0, 1.8)
    axR.set_title("Horizons 14 & 30 days", fontsize=12, fontweight="bold", pad=12)

    plt.suptitle("Table 1 — Test metrics (from paper_experiments.py)", fontsize=13, y=0.98)
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.isfile(CSV_PATH):
        raise FileNotFoundError(f"请先运行 paper_experiments.py 生成 {CSV_PATH}")

    df = load_df()
    plot_grouped_bars(df, os.path.join(OUT_DIR, "fig_metrics_grouped_by_horizon.png"))
    plot_line_rmse_mae(df, os.path.join(OUT_DIR, "fig_rmse_mae_lines.png"))
    plot_table_image(df, os.path.join(OUT_DIR, "fig_table1_render.png"))

    print("已保存:")
    for name in (
        "fig_metrics_grouped_by_horizon.png",
        "fig_rmse_mae_lines.png",
        "fig_table1_render.png",
    ):
        print(" ", os.path.join(OUT_DIR, name))


if __name__ == "__main__":
    main()
