#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
软约束 vs 硬约束 vs 无约束 三路模型对比：同一测试集上评估三种权重，生成对比表格与可视化。
需先分别训练:
  python train.py --constraint soft
  python train.py --constraint hard
  python train.py --constraint none   # 无 SIS 基线
"""
import os
import sys
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import csv
from tqdm import tqdm

# 复用 visualize_results 的加载与预测逻辑
from visualize_results import (
    load_model,
    batch_predict_one_step,
    batch_predict_multiple_days,
    get_test_dates,
    evaluate_multi_step_by_city,
    BATCH_SIZE,
)

OUTPUT_DIR = 'visualization_results'
COMPARE_DIR = os.path.join(OUTPUT_DIR, 'soft_vs_hard_comparison')

# 三路对比配置：(checkpoint 路径, 推理时是否硬约束, 显示名, 颜色)
MODEL_CONFIGS = [
    ('checkpoints/best_model_soft.pth', False, 'Soft constraint', '#45B7D1'),
    ('checkpoints/best_model_hard.pth', True, 'Hard constraint', '#FFA07A'),
    ('checkpoints/best_model_none.pth', False, 'No constraint', '#95A5A6'),
]


def ensure_compare_dir():
    os.makedirs(COMPARE_DIR, exist_ok=True)
    return COMPARE_DIR


def run_evaluation_for_model(checkpoint_path, device, data, horizons, use_hard_constraint):
    """加载指定 checkpoint 并在测试集上评估，返回 city_metrics, all_predictions, all_targets"""
    from model import create_fully_connected_graph
    model, data_full, config, transformer = load_model(checkpoint_path, device)
    cities = data_full['cities']
    test_X = data['test_X']
    test_y = data['test_y']
    edge_index = create_fully_connected_graph(len(cities)).to(device)
    city_metrics, all_predictions, all_targets = evaluate_multi_step_by_city(
        model, test_X, test_y, edge_index, device, cities, transformer, horizons=horizons,
        use_hard_constraint=use_hard_constraint
    )
    return city_metrics, all_predictions, all_targets, cities


def main():
    ensure_compare_dir()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    horizons = [3, 7, 14, 30]
    model_keys = ['soft', 'hard', 'none']

    for path, _, label, _ in MODEL_CONFIGS:
        if not os.path.exists(path):
            print(f"未找到权重: {path}")
            print("请先运行: python train.py --constraint soft")
            print("           python train.py --constraint hard")
            print("           python train.py --constraint none")
            sys.exit(1)

    with open('processed_data.pkl', 'rb') as f:
        data = pickle.load(f)
    from model import create_fully_connected_graph
    cities = data['cities']
    edge_index = create_fully_connected_graph(len(cities)).to(device)

    # 加载三路模型并评估
    metrics_by_key = {}
    preds_by_key = {}
    targets_ref = None
    print("=" * 70)
    print("软约束 vs 硬约束 vs 无约束 三路模型对比")
    print("=" * 70)
    for path, use_hard, label, _ in MODEL_CONFIGS:
        key = 'soft' if 'soft' in path else ('hard' if 'hard' in path else 'none')
        print(f"\n评估 {label} 模型...")
        m, p, t, _ = run_evaluation_for_model(path, device, data, horizons, use_hard_constraint=use_hard)
        metrics_by_key[key] = m
        preds_by_key[key] = p
        if targets_ref is None:
            targets_ref = t

    # ----- 1. 对比表格 CSV（三列 MAE/RMSE + Best）-----
    table_path = os.path.join(COMPARE_DIR, 'comparison_table.csv')
    with open(table_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(
            ['City', 'Horizon', 'MAE_soft', 'MAE_hard', 'MAE_none', 'RMSE_soft', 'RMSE_hard', 'RMSE_none',
             'Best_MAE', 'Best_RMSE'])
        for city in cities:
            for h in horizons:
                mae_vals = {k: metrics_by_key[k][city][h]['MAE'] for k in model_keys}
                rmse_vals = {k: metrics_by_key[k][city][h]['RMSE'] for k in model_keys}
                best_mae = min(mae_vals, key=mae_vals.get)
                best_rmse = min(rmse_vals, key=rmse_vals.get)
                writer.writerow([
                    city, h,
                    f"{mae_vals['soft']:.4f}", f"{mae_vals['hard']:.4f}", f"{mae_vals['none']:.4f}",
                    f"{rmse_vals['soft']:.4f}", f"{rmse_vals['hard']:.4f}", f"{rmse_vals['none']:.4f}",
                    best_mae, best_rmse
                ])
    print(f"对比表已保存: {table_path}")

    # 汇总表：按 horizon 的平均 MAE/RMSE（三列 + Best）
    summary_path = os.path.join(COMPARE_DIR, 'comparison_summary_by_horizon.csv')
    with open(summary_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Horizon', 'MAE_soft_avg', 'MAE_hard_avg', 'MAE_none_avg',
            'RMSE_soft_avg', 'RMSE_hard_avg', 'RMSE_none_avg', 'Best_MAE', 'Best_RMSE'
        ])
        for h in horizons:
            mae_avg = {k: np.mean([metrics_by_key[k][c][h]['MAE'] for c in cities]) for k in model_keys}
            rmse_avg = {k: np.mean([metrics_by_key[k][c][h]['RMSE'] for c in cities]) for k in model_keys}
            best_mae = min(mae_avg, key=mae_avg.get)
            best_rmse = min(rmse_avg, key=rmse_avg.get)
            writer.writerow([
                h,
                f"{mae_avg['soft']:.4f}", f"{mae_avg['hard']:.4f}", f"{mae_avg['none']:.4f}",
                f"{rmse_avg['soft']:.4f}", f"{rmse_avg['hard']:.4f}", f"{rmse_avg['none']:.4f}",
                best_mae, best_rmse
            ])
    print(f"按 horizon 汇总表已保存: {summary_path}")

    # ----- 2. 对比图：MAE/RMSE 按城市（三柱）-----
    n_cities = len(cities)
    width = 0.25
    x = np.arange(n_cities)
    fig, axes = plt.subplots(2, 1, figsize=(max(14, n_cities * 0.5), 10))
    mae_avgs = {k: np.array([np.mean([metrics_by_key[k][c][h]['MAE'] for h in horizons]) for c in cities]) for k in model_keys}
    rmse_avgs = {k: np.array([np.mean([metrics_by_key[k][c][h]['RMSE'] for h in horizons]) for c in cities]) for k in model_keys}

    ax = axes[0]
    ax.bar(x - width, mae_avgs['soft'], width, label='Soft constraint', color='#45B7D1', alpha=0.9)
    ax.bar(x, mae_avgs['hard'], width, label='Hard constraint', color='#FFA07A', alpha=0.9)
    ax.bar(x + width, mae_avgs['none'], width, label='No constraint', color='#95A5A6', alpha=0.9)
    ax.set_ylabel('MAE (avg over horizons)', fontsize=12, fontweight='bold')
    ax.set_title('Soft vs Hard vs No Constraint: MAE by City (average over 3,7,14,30 days)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(cities, rotation=45, ha='right', fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1]
    ax.bar(x - width, rmse_avgs['soft'], width, label='Soft constraint', color='#45B7D1', alpha=0.9)
    ax.bar(x, rmse_avgs['hard'], width, label='Hard constraint', color='#FFA07A', alpha=0.9)
    ax.bar(x + width, rmse_avgs['none'], width, label='No constraint', color='#95A5A6', alpha=0.9)
    ax.set_ylabel('RMSE (avg over horizons)', fontsize=12, fontweight='bold')
    ax.set_xlabel('City', fontsize=12, fontweight='bold')
    ax.set_title('Soft vs Hard vs No Constraint: RMSE by City (average over 3,7,14,30 days)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(cities, rotation=45, ha='right', fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig_path = os.path.join(COMPARE_DIR, 'fig_comparison_mae_rmse_by_city.png')
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"对比图已保存: {fig_path}")

    # ----- 3. 按 horizon 的总体 MAE/RMSE 柱状图（三柱）-----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    h_mae = {k: [np.mean([metrics_by_key[k][c][h]['MAE'] for c in cities]) for h in horizons] for k in model_keys}
    h_rmse = {k: [np.mean([metrics_by_key[k][c][h]['RMSE'] for c in cities]) for h in horizons] for k in model_keys}

    xh = np.arange(len(horizons))
    w = 0.25
    axes[0].bar(xh - w, h_mae['soft'], w, label='Soft constraint', color='#45B7D1', alpha=0.9)
    axes[0].bar(xh, h_mae['hard'], w, label='Hard constraint', color='#FFA07A', alpha=0.9)
    axes[0].bar(xh + w, h_mae['none'], w, label='No constraint', color='#95A5A6', alpha=0.9)
    axes[0].set_xticks(xh)
    axes[0].set_xticklabels([f'{h}d' for h in horizons])
    axes[0].set_ylabel('MAE')
    axes[0].set_title('Average MAE by Forecast Horizon')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    axes[1].bar(xh - w, h_rmse['soft'], w, label='Soft constraint', color='#45B7D1', alpha=0.9)
    axes[1].bar(xh, h_rmse['hard'], w, label='Hard constraint', color='#FFA07A', alpha=0.9)
    axes[1].bar(xh + w, h_rmse['none'], w, label='No constraint', color='#95A5A6', alpha=0.9)
    axes[1].set_xticks(xh)
    axes[1].set_xticklabels([f'{h}d' for h in horizons])
    axes[1].set_ylabel('RMSE')
    axes[1].set_title('Average RMSE by Forecast Horizon')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig_path2 = os.path.join(COMPARE_DIR, 'fig_comparison_by_horizon.png')
    plt.savefig(fig_path2, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"按 horizon 对比图已保存: {fig_path2}")

    # ----- 4. 示例城市时间序列：Actual vs Soft vs Hard vs None -----
    test_dates = get_test_dates(data)
    sample_cities = cities[:4] if len(cities) >= 4 else cities
    max_h = max(horizons)
    n_samples = preds_by_key['soft'][max_h].shape[0]
    dates_full = test_dates[max_h - 1:max_h - 1 + n_samples]
    max_pts = 600
    if len(dates_full) > max_pts:
        step = len(dates_full) // max_pts
        indices = np.arange(0, len(dates_full), step)[:max_pts]
        dates_plot = dates_full[indices]
    else:
        indices = np.arange(len(dates_full))
        dates_plot = dates_full

    fig, axes = plt.subplots(len(sample_cities), 1, figsize=(14, 4 * len(sample_cities)))
    if len(sample_cities) == 1:
        axes = [axes]
    for i, city_name in enumerate(sample_cities):
        ax = axes[i]
        cidx = cities.index(city_name)
        actual = targets_ref[max_h][:, -1, cidx]
        a_plot = actual[indices]
        ax.plot(dates_plot, a_plot, 'k-', label='Actual', lw=1.5, alpha=0.9)
        for key, color in [('soft', '#45B7D1'), ('hard', '#FFA07A'), ('none', '#95A5A6')]:
            pred = preds_by_key[key][max_h][:, -1, cidx][indices]
            legend_label = {'soft': 'Soft constraint', 'hard': 'Hard constraint', 'none': 'No constraint'}[key]
            ax.plot(dates_plot, pred, '--', color=color, label=legend_label, lw=1.2, alpha=0.8)
        ax.set_ylabel(f'{city_name}\nCases', fontsize=10)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    axes[-1].set_xlabel('Date', fontsize=12, fontweight='bold')
    plt.suptitle('Soft vs Hard vs No Constraint: Example Cities Time Series (max horizon)', fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig_path3 = os.path.join(COMPARE_DIR, 'fig_comparison_timeseries_sample.png')
    plt.savefig(fig_path3, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"示例时间序列图已保存: {fig_path3}")

    print("\n" + "=" * 70)
    print("三路对比完成。输出目录:", COMPARE_DIR)
    print("  - comparison_table.csv: 按城市、按 horizon 的 MAE/RMSE（soft/hard/none）及 Best")
    print("  - comparison_summary_by_horizon.csv: 按 horizon 的平均指标（三列）及 Best")
    print("  - fig_comparison_mae_rmse_by_city.png: 按城市 MAE/RMSE 三柱对比")
    print("  - fig_comparison_by_horizon.png: 按预测天数 MAE/RMSE 三柱对比")
    print("  - fig_comparison_timeseries_sample.png: 示例城市时间序列（Actual + 三种模型）")
    print("=" * 70)


if __name__ == "__main__":
    main()
