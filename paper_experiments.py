#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
论文 5.3 指标（RMSE, MAE, MAPE, RAE）与 5.4 多模型对比：
ARIMA、LSTM、Base(none)、Proposed(soft)、Proposed(hard，残差硬约束)。
在测试集上对 horizon ∈ {3,7,14,30} 做多步自回归预测，与 visualize_results 评估协议一致。

MAE/MAPE 使用绝对值（论文式 (28)(29) 标准写法）。
RAE 分母为全体真值相对其整体均值 ¯Y 的绝对离差之和（式 30）。
"""
from __future__ import annotations

import os

# Windows 上 NumPy(MKL)/PyTorch 可能各带一份 OpenMP，导入顺序会触发 libiomp5 冲突；须在 import numpy/torch 之前设置
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import pickle
import subprocess
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from data_preprocessing import load_and_preprocess_data, create_sequences, split_train_test
from model import create_fully_connected_graph
from visualize_results import (
    DataTransformer,
    load_model,
    batch_predict_multiple_days,
)


HORIZONS = [3, 7, 14, 30]


def paper_metrics(pred: np.ndarray, target: np.ndarray, mape_eps: float = 1.0) -> dict:
    """
    pred, target: 同形状，原始病例尺度（非负）。
    在展平后的所有 (样本, 步长, 节点) 上计算式 (27)-(30)（MAE/MAPE 带 abs）。
    """
    p = pred.astype(np.float64).ravel()
    y = target.astype(np.float64).ravel()
    err = p - y
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    denom_mape = np.maximum(np.abs(y), mape_eps)
    mape = float(100.0 * np.mean(np.abs(err) / denom_mape))
    y_bar = float(np.mean(y))
    denom_rae = np.sum(np.abs(y - y_bar))
    if denom_rae < 1e-12:
        rae = float("nan")
    else:
        rae = float(np.sum(np.abs(err)) / denom_rae)
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape, "RAE": rae}


def _arima_one_sample_row(
    i: int,
    data_matrix: np.ndarray,
    n_train: int,
    n_val: int,
    window_size: int,
    horizon: int,
    n_cities: int,
    max_hist: int,
) -> tuple[int, np.ndarray]:
    """单测试窗口、所有城市的 ARIMA 多步预测；供线程池调用。"""
    from statsmodels.tsa.arima.model import ARIMA

    g = n_train + n_val + i
    t_end = g + window_size - 1
    row = np.zeros((horizon, n_cities), dtype=np.float32)
    for n in range(n_cities):
        hist = data_matrix[: t_end + 1, n].astype(np.float64)
        if max_hist and len(hist) > max_hist:
            hist = hist[-max_hist:]
        fc = np.zeros(horizon, dtype=np.float64)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if np.std(hist) < 1e-12:
                    fc[:] = hist[-1]
                else:
                    m = ARIMA(hist, order=(1, 1, 1)).fit()
                    fc = np.asarray(m.forecast(steps=horizon), dtype=np.float64)
        except Exception:
            fc[:] = hist[-1]
        fc = np.maximum(fc, 0.0)
        row[:, n] = fc.astype(np.float32)
    return i, row


def build_targets_raw(test_y: np.ndarray, n_samples: int, horizon: int, n_cities: int) -> np.ndarray:
    """与 visualize_results.evaluate_multi_step_by_city 一致。"""
    targets_raw = np.zeros((n_samples, horizon, n_cities), dtype=np.float32)
    for i in range(n_samples):
        targets_raw[i] = test_y[i : i + horizon]
    return targets_raw


def predict_gnn_multistep(
    checkpoint_path: str,
    device: str,
    test_X: np.ndarray,
    horizon: int,
    use_hard_constraint: bool,
    gnn_batch_size: int,
) -> np.ndarray:
    """每个 checkpoint 自带 scaler，须分别做 log 标准化。"""
    model, _, config, transformer = load_model(checkpoint_path, device)
    n_samples = len(test_X) - horizon + 1
    initial_windows_raw = test_X[:n_samples]
    initial_windows = transformer.transform(initial_windows_raw)
    edge_index = create_fully_connected_graph(model.num_cities).to(device)
    pred_t = batch_predict_multiple_days(
        model,
        initial_windows,
        horizon,
        edge_index,
        device,
        gnn_batch_size,
        use_hard_constraint=use_hard_constraint,
    )
    return transformer.inverse_transform(pred_t)


class LSTMBaseline(nn.Module):
    """多变量 LSTM：输入 (B, w, N)，预测下一步 N 维（与训练标签对齐）。"""

    def __init__(self, num_cities: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            num_cities,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, num_cities)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def fit_log_scaler(train_X: np.ndarray, train_y: np.ndarray) -> dict:
    X_log = np.log1p(train_X)
    y_log = np.log1p(train_y)
    log_mean = float(X_log.mean())
    log_std = float(X_log.std() + 1e-8)
    return {"log_mean": log_mean, "log_std": log_std}


def transform_arr(x: np.ndarray, scaler: dict) -> np.ndarray:
    xl = np.log1p(x)
    return (xl - scaler["log_mean"]) / scaler["log_std"]


def train_lstm_baseline(
    train_X: np.ndarray,
    train_y: np.ndarray,
    val_X: np.ndarray,
    val_y: np.ndarray,
    window_size: int,
    num_cities: int,
    device: str,
    epochs: int,
    save_path: str,
    batch_size: int = 256,
) -> dict:
    scaler = fit_log_scaler(train_X, train_y)
    transformer = DataTransformer(scaler)

    X_tr = torch.FloatTensor(transform_arr(train_X, scaler))
    y_tr = torch.FloatTensor(transform_arr(train_y, scaler))
    X_va = torch.FloatTensor(transform_arr(val_X, scaler))
    y_va = torch.FloatTensor(transform_arr(val_y, scaler))

    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_va, y_va), batch_size=batch_size, shuffle=False)

    model = LSTMBaseline(num_cities).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    best_state = None
    best_val = float("inf")

    for ep in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        se = 0.0
        n = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                se += (pred - yb).pow(2).sum().item()
                n += yb.numel()
        val_rmse = (se / n) ** 0.5
        if val_rmse < best_val:
            best_val = val_rmse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f"  [LSTM] epoch {ep+1}/{epochs} val_rmse(transformed)={val_rmse:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "scaler": scaler, "window_size": window_size}, save_path)
    print(f"LSTM 基线已保存: {save_path}")
    return scaler


def predict_lstm_multistep(
    state_path: str,
    device: str,
    test_X: np.ndarray,
    horizon: int,
    num_cities: int,
    window_size: int,
    batch_size: int = 512,
) -> np.ndarray:
    ckpt = torch.load(state_path, map_location=device, weights_only=False)
    transformer = DataTransformer(ckpt["scaler"])
    model = LSTMBaseline(num_cities).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    n_samples = len(test_X) - horizon + 1
    preds = np.zeros((n_samples, horizon, num_cities), dtype=np.float32)
    current = transform_arr(test_X[:n_samples].copy(), ckpt["scaler"])

    for day in range(horizon):
        day_pred = np.zeros((n_samples, num_cities), dtype=np.float32)
        with torch.no_grad():
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                xb = torch.FloatTensor(current[start:end]).to(device)
                pb = model(xb).cpu().numpy()
                day_pred[start:end] = pb
        preds[:, day, :] = transformer.inverse_transform(day_pred)
        current = np.concatenate([current[:, 1:, :], day_pred[:, np.newaxis, :]], axis=1)
    return preds


def predict_arima_multistep(
    data_matrix: np.ndarray,
    n_train: int,
    n_val: int,
    window_size: int,
    test_X: np.ndarray,
    test_y: np.ndarray,
    horizon: int,
    max_hist: int = 150,
    max_samples: int | None = None,
    num_workers: int = 8,
) -> np.ndarray:
    try:
        import statsmodels  # noqa: F401
    except ImportError:
        raise RuntimeError("请安装 statsmodels: pip install statsmodels")

    n_cities = data_matrix.shape[1]
    n_samples_full = len(test_X) - horizon + 1
    n_samples = n_samples_full if max_samples is None else min(n_samples_full, max_samples)
    preds = np.zeros((n_samples, horizon, n_cities), dtype=np.float32)

    worker = partial(
        _arima_one_sample_row,
        data_matrix=data_matrix,
        n_train=n_train,
        n_val=n_val,
        window_size=window_size,
        horizon=horizon,
        n_cities=n_cities,
        max_hist=max_hist,
    )
    workers = max(1, min(num_workers, os.cpu_count() or 8))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(worker, i) for i in range(n_samples)]
        for fut in tqdm(as_completed(futures), total=n_samples, desc=f"ARIMA h={horizon} ({workers} threads)"):
            i, row = fut.result()
            preds[i] = row
    return preds


def _flush_metrics_csv(rows: list, csv_path: str) -> None:
    import csv

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "horizon", "RMSE", "MAE", "MAPE", "RAE"])
        w.writeheader()
        for row in rows:
            w.writerow(row)


def write_paper_table_files(rows: list, out_dir: str) -> None:
    """
    生成与论文 Table 1 版式一致的 Markdown（含按指标最优的加粗）与 LaTeX 片段。
    指标：式 (27)-(30)，数值越小越好。
    """
    os.makedirs(out_dir, exist_ok=True)
    idx = {}
    for r in rows:
        idx[(r["model"], int(r["horizon"]))] = r

    model_order = [
        ("ARIMA", "ARIMA"),
        ("LSTM", "LSTM"),
        ("Base", "Base Model"),
        ("Proposed", "Proposed (soft)"),
        ("Hard", "Proposed (hard)"),
    ]
    metrics = ["RMSE", "MAE", "MAPE", "RAE"]

    def cell(m: str, h: int, met: str) -> str:
        rec = idx.get((m, h), {})
        v = rec.get(met, float("nan"))
        if np.isnan(v):
            return "—"
        if met == "MAPE":
            return f"{v:.2f}"
        return f"{v:.4f}"

    def best_for(h: int, met: str) -> str | None:
        vals = []
        for m, _ in model_order:
            rec = idx.get((m, h))
            if rec is None:
                continue
            vv = rec.get(met, np.nan)
            if not np.isnan(vv):
                vals.append((m, vv))
        if not vals:
            return None
        return min(vals, key=lambda x: x[1])[0]

    lines_md = [
        "# Table 1（由 `paper_experiments.py` 实测生成，对应式 27–30）",
        "",
        "## 3-day & 7-day",
        "",
        "| Model | RMSE | MAE | MAPE (%) | RAE | RMSE | MAE | MAPE (%) | RAE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for m, mlabel in model_order:
        parts = [mlabel]
        for h in (3, 7):
            for met in metrics:
                s = cell(m, h, met)
                if best_for(h, met) == m:
                    s = f"**{s}**"
                parts.append(s)
        lines_md.append("| " + " | ".join(parts) + " |")

    lines_md.extend(
        [
            "",
            "## 14-day & 30-day",
            "",
            "| Model | RMSE | MAE | MAPE (%) | RAE | RMSE | MAE | MAPE (%) | RAE |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for m, mlabel in model_order:
        parts = [mlabel]
        for h in (14, 30):
            for met in metrics:
                s = cell(m, h, met)
                if best_for(h, met) == m:
                    s = f"**{s}**"
                parts.append(s)
        lines_md.append("| " + " | ".join(parts) + " |")

    md_path = os.path.join(out_dir, "TABLE1_PAPER.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines_md) + "\n")

    # 紧凑 LaTeX 行（便于粘贴）
    tex_lines = ["% Table 1 numeric rows (paste into your tabular)", "% 3-day & 7-day"]
    for m, mlabel in model_order:
        vals = []
        for h in (3, 7):
            for met in metrics:
                rec = idx.get((m, h), {})
                v = rec.get(met, float("nan"))
                vals.append(f"{v:.4f}" if met != "MAPE" and not np.isnan(v) else (f"{v:.2f}" if not np.isnan(v) else "nan"))
        tex_lines.append(f"% {mlabel}: " + " & ".join(vals) + r" \\")
    tex_lines.append("% 14-day & 30-day")
    for m, mlabel in model_order:
        vals = []
        for h in (14, 30):
            for met in metrics:
                rec = idx.get((m, h), {})
                v = rec.get(met, float("nan"))
                vals.append(f"{v:.4f}" if met != "MAPE" and not np.isnan(v) else (f"{v:.2f}" if not np.isnan(v) else "nan"))
        tex_lines.append(f"% {mlabel}: " + " & ".join(vals) + r" \\")
    with open(os.path.join(out_dir, "table1_rows.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(tex_lines) + "\n")

    print(f"论文用表格已写入: {md_path}")


def ensure_processed_data():
    if not os.path.exists("processed_data.pkl"):
        print("未找到 processed_data.pkl，正在运行 data_preprocessing.py ...")
        subprocess.check_call([sys.executable, "data_preprocessing.py"], cwd=os.path.dirname(os.path.abspath(__file__)) or ".")


def ensure_gnn_checkpoints(epochs: int, train_gnn: bool):
    soft = "checkpoints/best_model_soft.pth"
    none = "checkpoints/best_model_none.pth"
    hard = "checkpoints/best_model_hard.pth"
    need = (
        train_gnn
        or not os.path.exists(soft)
        or not os.path.exists(none)
        or not os.path.exists(hard)
    )
    if not need:
        return
    os.makedirs("checkpoints", exist_ok=True)
    root = os.path.dirname(os.path.abspath(__file__)) or "."
    for constraint in ("none", "soft", "hard"):
        print(f"\n>>> 训练 GNN ({constraint})，{epochs} epochs ...")
        cmd = [sys.executable, "train.py", "--constraint", constraint, "--num_epochs", str(epochs)]
        subprocess.check_call(cmd, cwd=root)


def main():
    parser = argparse.ArgumentParser(description="论文 5.3–5.4：指标与基线对比表")
    parser.add_argument("--gnn-epochs", type=int, default=80, help="缺少或强制训练 GNN 时的 epoch 数")
    parser.add_argument(
        "--train-gnn",
        action="store_true",
        help="强制重新训练 Base(none)、Proposed(soft)、Proposed(hard)",
    )
    parser.add_argument(
        "--skip-hard",
        action="store_true",
        help="不评估硬约束模型（若未训练 hard 且无权重文件）",
    )
    parser.add_argument(
        "--checkpoint-hard",
        type=str,
        default=None,
        help="硬约束模型权重路径（默认 checkpoints/best_model_hard.pth，用于对比调参产物）",
    )
    parser.add_argument("--lstm-epochs", type=int, default=60, help="LSTM 基线训练轮数")
    parser.add_argument("--train-lstm", action="store_true", help="强制重新训练 LSTM")
    parser.add_argument("--skip-arima", action="store_true", help="跳过 ARIMA（较慢）")
    parser.add_argument(
        "--max-arima-samples",
        type=int,
        default=None,
        help="ARIMA 最多评估的测试窗口数（默认全部，数据大可设 500）",
    )
    parser.add_argument("--device", type=str, default=None, help="cuda 或 cpu")
    parser.add_argument(
        "--arima-workers",
        type=int,
        default=8,
        help="ARIMA 按测试窗口并行的线程数（statsmodels 在各自线程内独立 fit）",
    )
    parser.add_argument(
        "--gnn-batch-size",
        type=int,
        default=None,
        help="GNN 多步推理 batch（默认：CUDA 用 256，CPU 用 32）。"
        "模型内部会把 batch×城市数 展平进 LSTM，故不能按 visualize 里 2048 设那么大；8GB 若 OOM 再改为 128 或 64。",
    )
    args = parser.parse_args()

    cuda_ok = torch.cuda.is_available()
    if args.device is not None:
        device = args.device
        if device == "cuda" and not cuda_ok:
            print(
                "警告: 指定了 --device cuda 但 torch.cuda.is_available() 为 False，"
                "将回退到 CPU。请安装带 CUDA 的 PyTorch 与 NVIDIA 驱动（见下方说明）。"
            )
            device = "cpu"
    else:
        device = "cuda" if cuda_ok else "cpu"

    gnn_batch_size = args.gnn_batch_size
    if gnn_batch_size is None:
        # 时间编码器对 (batch × num_cities) 条序列做 LSTM，显存随 batch 近似线性放大；8GB 卡默认 256 较稳
        gnn_batch_size = 256 if device == "cuda" else 32

    print("=" * 60)
    print(f"PyTorch {torch.__version__} | torch.cuda.is_available() = {cuda_ok}")
    if cuda_ok:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(
            "当前为 CPU 推理（batch 默认 32）。要用 GPU：\n"
            "  1) 安装 NVIDIA 显卡驱动；\n"
            "  2) 安装与驱动匹配的 CUDA 版 PyTorch（勿用仅 CPU 的 wheel）：\n"
            "     https://pytorch.org/get-started/locally/\n"
            "  3) 在终端执行: python -c \"import torch; print(torch.cuda.is_available())\" 应输出 True。"
        )
    print(f"本脚本实际使用 device={device} | GNN batch_size={gnn_batch_size}")
    if not args.skip_hard:
        print(f"硬约束 checkpoint: {args.checkpoint_hard or 'checkpoints/best_model_hard.pth'}")
    print("=" * 60)

    ensure_processed_data()

    with open("processed_data.pkl", "rb") as f:
        data = pickle.load(f)
    test_X = data["test_X"]
    test_y = data["test_y"]
    cities = data["cities"]
    window_size = data["window_size"]
    n_train = len(data["train_X"])
    n_val = len(data["val_X"])
    num_cities = len(cities)

    ensure_gnn_checkpoints(args.gnn_epochs, args.train_gnn)

    lstm_path = "checkpoints/lstm_baseline.pth"
    if args.train_lstm or not os.path.exists(lstm_path):
        print("\n>>> 训练 LSTM 基线 ...")
        train_lstm_baseline(
            data["train_X"],
            data["train_y"],
            data["val_X"],
            data["val_y"],
            window_size,
            num_cities,
            device,
            args.lstm_epochs,
            lstm_path,
        )

    print("\n加载完整病例矩阵（用于 ARIMA）...")
    raw = load_and_preprocess_data("Dengue_Daily_EN.csv", use_notification_date=True)
    M = raw["data_matrix"].astype(np.float32)
    w = window_size
    X_full, y_full = create_sequences(M, window_size=w, forecast_horizon=1)
    n_train_s = len(data["train_X"])
    assert X_full.shape[0] >= n_train_s + len(data["test_X"])
    # 与预处理划分一致：按样本索引切分
    n_val_s = len(data["val_X"])
    n_test_s = len(data["test_X"])
    assert n_train_s + n_val_s + n_test_s == X_full.shape[0]

    rows = []
    out_dir = "paper_results"
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "table1_metrics_by_horizon.csv")

    for h in HORIZONS:
        n_samples = len(test_X) - h + 1
        targets = build_targets_raw(test_y, n_samples, h, num_cities)
        max_ar = args.max_arima_samples

        print(f"\n======== Horizon {h} days (n_samples={n_samples}) ========")
        print(f"  GNN 推理 batch_size={gnn_batch_size}  device={device}")

        # --- Proposed (soft + SIS loss path; 推理非 hard) ---
        pred_prop = predict_gnn_multistep(
            "checkpoints/best_model_soft.pth",
            device,
            test_X,
            h,
            use_hard_constraint=False,
            gnn_batch_size=gnn_batch_size,
        )
        if max_ar:
            pred_prop = pred_prop[:max_ar]
            targets_sub = targets[:max_ar]
        else:
            targets_sub = targets
        m = paper_metrics(pred_prop, targets_sub)
        print(f"  Proposed(soft): RMSE={m['RMSE']:.4f} MAE={m['MAE']:.4f} MAPE={m['MAPE']:.2f}% RAE={m['RAE']:.4f}")
        rows.append({"model": "Proposed", "horizon": h, **m})
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # --- Proposed (hard): 输出 = SIS + softplus(NN)，与 train.py --constraint hard 一致 ---
        hard_ckpt = args.checkpoint_hard or "checkpoints/best_model_hard.pth"
        if args.skip_hard:
            rows.append({"model": "Hard", "horizon": h, "RMSE": np.nan, "MAE": np.nan, "MAPE": np.nan, "RAE": np.nan})
            print("  Proposed(hard): (已 --skip-hard 跳过)")
        elif not os.path.isfile(hard_ckpt):
            print(f"  Proposed(hard): 跳过（未找到 {hard_ckpt}，请运行 python train.py --constraint hard）")
            rows.append({"model": "Hard", "horizon": h, "RMSE": np.nan, "MAE": np.nan, "MAPE": np.nan, "RAE": np.nan})
        else:
            pred_hard = predict_gnn_multistep(
                hard_ckpt,
                device,
                test_X,
                h,
                use_hard_constraint=True,
                gnn_batch_size=gnn_batch_size,
            )
            if max_ar:
                pred_hard = pred_hard[:max_ar]
            m = paper_metrics(pred_hard, targets_sub)
            print(f"  Proposed(hard): RMSE={m['RMSE']:.4f} MAE={m['MAE']:.4f} MAPE={m['MAPE']:.2f}% RAE={m['RAE']:.4f}")
            rows.append({"model": "Hard", "horizon": h, **m})
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # --- Base (no physics) ---
        pred_base = predict_gnn_multistep(
            "checkpoints/best_model_none.pth",
            device,
            test_X,
            h,
            use_hard_constraint=False,
            gnn_batch_size=gnn_batch_size,
        )
        if max_ar:
            pred_base = pred_base[:max_ar]
        m = paper_metrics(pred_base, targets_sub)
        print(f"  Base    : RMSE={m['RMSE']:.4f} MAE={m['MAE']:.4f} MAPE={m['MAPE']:.2f}% RAE={m['RAE']:.4f}")
        rows.append({"model": "Base", "horizon": h, **m})
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # --- LSTM ---
        pred_lstm = predict_lstm_multistep(
            lstm_path, device, test_X, h, num_cities, window_size
        )
        if max_ar:
            pred_lstm = pred_lstm[:max_ar]
        m = paper_metrics(pred_lstm, targets_sub)
        print(f"  LSTM    : RMSE={m['RMSE']:.4f} MAE={m['MAE']:.4f} MAPE={m['MAPE']:.2f}% RAE={m['RAE']:.4f}")
        rows.append({"model": "LSTM", "horizon": h, **m})

        # --- ARIMA ---
        if not args.skip_arima:
            pred_ar = predict_arima_multistep(
                M,
                n_train_s,
                n_val_s,
                w,
                test_X,
                test_y,
                h,
                max_samples=max_ar,
                num_workers=args.arima_workers,
            )
            m = paper_metrics(pred_ar, targets_sub[: len(pred_ar)])
            print(f"  ARIMA   : RMSE={m['RMSE']:.4f} MAE={m['MAE']:.4f} MAPE={m['MAPE']:.2f}% RAE={m['RAE']:.4f}")
            rows.append({"model": "ARIMA", "horizon": h, **m})
        else:
            rows.append({"model": "ARIMA", "horizon": h, "RMSE": np.nan, "MAE": np.nan, "MAPE": np.nan, "RAE": np.nan})

        _flush_metrics_csv(rows, csv_path)
        print(f"  (已增量写入 {csv_path})")

    write_paper_table_files(rows, out_dir)
    print(f"\n指标 CSV: {csv_path}")


if __name__ == "__main__":
    main()
