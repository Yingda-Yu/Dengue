#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
软约束 vs 硬约束模型对比：同一测试集上评估两个权重，生成对比表格与可视化。
需先分别训练: python train.py --constraint soft 与 python train.py --constraint hard
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
    soft_path = 'checkpoints/best_model_soft.pth'
    hard_path = 'checkpoints/best_model_hard.pth'

    if not os.path.exists(soft_path):
        print(f"未找到软约束权重: {soft_path}，请先运行: python train.py --constraint soft")
        sys.exit(1)
    if not os.path.exists(hard_path):
        print(f"未找到硬约束权重: {hard_path}，请先运行: python train.py --constraint hard")
        sys.exit(1)

    with open('processed_data.pkl', 'rb') as f:
        data = pickle.load(f)
    from model import create_fully_connected_graph
    cities = data['cities']
    edge_index = create_fully_connected_graph(len(cities)).to(device)

    print("=" * 70)
    print("软约束 vs 硬约束 模型对比")
    print("=" * 70)
    print("评估软约束模型...")
    metrics_soft, preds_soft, targets_soft, _ = run_evaluation_for_model(
        soft_path, device, data, horizons, use_hard_constraint=False
    )
    print("\n评估硬约束模型...")
    metrics_hard, preds_hard, targets_hard, _ = run_evaluation_for_model(
        hard_path, device, data, horizons, use_hard_constraint=True
    )

    # ----- 1. 对比表格 CSV -----
    table_path = os.path.join(COMPARE_DIR, 'comparison_table.csv')
    with open(table_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['City', 'Horizon', 'MAE_soft', 'MAE_hard', 'RMSE_soft', 'RMSE_hard',
                         'MAE_diff (hard-soft)', 'RMSE_diff (hard-soft)', 'Better_MAE', 'Better_RMSE'])
        for city in cities:
            for h in horizons:
                mae_s = metrics_soft[city][h]['MAE']
                mae_h = metrics_hard[city][h]['MAE']
                rmse_s = metrics_soft[city][h]['RMSE']
                rmse_h = metrics_hard[city][h]['RMSE']
                mae_diff = mae_h - mae_s
                rmse_diff = rmse_h - rmse_s
                better_mae = 'hard' if mae_h < mae_s else 'soft'
                better_rmse = 'hard' if rmse_h < rmse_s else 'soft'
                writer.writerow([city, h, f'{mae_s:.4f}', f'{mae_h:.4f}', f'{rmse_s:.4f}', f'{rmse_h:.4f}',
                                 f'{mae_diff:.4f}', f'{rmse_diff:.4f}', better_mae, better_rmse])
    print(f"对比表已保存: {table_path}")

    # 汇总表：按 horizon 的平均 MAE/RMSE
    summary_path = os.path.join(COMPARE_DIR, 'comparison_summary_by_horizon.csv')
    with open(summary_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Horizon', 'MAE_soft_avg', 'MAE_hard_avg', 'RMSE_soft_avg', 'RMSE_hard_avg', 'Better_MAE', 'Better_RMSE'])
        for h in horizons:
            mae_s_avg = np.mean([metrics_soft[c][h]['MAE'] for c in cities])
            mae_h_avg = np.mean([metrics_hard[c][h]['MAE'] for c in cities])
            rmse_s_avg = np.mean([metrics_soft[c][h]['RMSE'] for c in cities])
            rmse_h_avg = np.mean([metrics_hard[c][h]['RMSE'] for c in cities])
            better_mae = 'hard' if mae_h_avg < mae_s_avg else 'soft'
            better_rmse = 'hard' if rmse_h_avg < rmse_s_avg else 'soft'
            writer.writerow([h, f'{mae_s_avg:.4f}', f'{mae_h_avg:.4f}', f'{rmse_s_avg:.4f}', f'{rmse_h_avg:.4f}', better_mae, better_rmse])
    print(f"按 horizon 汇总表已保存: {summary_path}")

    # ----- 2. 对比图：MAE 按城市（多 horizon 平均或分 horizon）-----
    n_cities = len(cities)
    mae_soft_avg = np.array([np.mean([metrics_soft[c][h]['MAE'] for h in horizons]) for c in cities])
    mae_hard_avg = np.array([np.mean([metrics_hard[c][h]['MAE'] for h in horizons]) for c in cities])
    rmse_soft_avg = np.array([np.mean([metrics_soft[c][h]['RMSE'] for h in horizons]) for c in cities])
    rmse_hard_avg = np.array([np.mean([metrics_hard[c][h]['RMSE'] for h in horizons]) for c in cities])

    x = np.arange(n_cities)
    width = 0.35
    fig, axes = plt.subplots(2, 1, figsize=(max(14, n_cities * 0.5), 10))
    ax = axes[0]
    ax.bar(x - width/2, mae_soft_avg, width, label='Soft constraint', color='#45B7D1', alpha=0.9)
    ax.bar(x + width/2, mae_hard_avg, width, label='Hard constraint', color='#FFA07A', alpha=0.9)
    ax.set_ylabel('MAE (avg over horizons)', fontsize=12, fontweight='bold')
    ax.set_title('Soft vs Hard Constraint: MAE by City (average over 3,7,14,30 days)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(cities, rotation=45, ha='right', fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1]
    ax.bar(x - width/2, rmse_soft_avg, width, label='Soft constraint', color='#45B7D1', alpha=0.9)
    ax.bar(x + width/2, rmse_hard_avg, width, label='Hard constraint', color='#FFA07A', alpha=0.9)
    ax.set_ylabel('RMSE (avg over horizons)', fontsize=12, fontweight='bold')
    ax.set_xlabel('City', fontsize=12, fontweight='bold')
    ax.set_title('Soft vs Hard Constraint: RMSE by City (average over 3,7,14,30 days)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(cities, rotation=45, ha='right', fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig_path = os.path.join(COMPARE_DIR, 'fig_comparison_mae_rmse_by_city.png')
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"对比图已保存: {fig_path}")

    # ----- 3. 按 horizon 的总体 MAE/RMSE 柱状图 -----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    h_mae_soft = [np.mean([metrics_soft[c][h]['MAE'] for c in cities]) for h in horizons]
    h_mae_hard = [np.mean([metrics_hard[c][h]['MAE'] for c in cities]) for h in horizons]
    h_rmse_soft = [np.mean([metrics_soft[c][h]['RMSE'] for c in cities]) for h in horizons]
    h_rmse_hard = [np.mean([metrics_hard[c][h]['RMSE'] for c in cities]) for h in horizons]

    xh = np.arange(len(horizons))
    w = 0.35
    axes[0].bar(xh - w/2, h_mae_soft, w, label='Soft', color='#45B7D1', alpha=0.9)
    axes[0].bar(xh + w/2, h_mae_hard, w, label='Hard', color='#FFA07A', alpha=0.9)
    axes[0].set_xticks(xh)
    axes[0].set_xticklabels([f'{h}d' for h in horizons])
    axes[0].set_ylabel('MAE')
    axes[0].set_title('Average MAE by Forecast Horizon')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    axes[1].bar(xh - w/2, h_rmse_soft, w, label='Soft', color='#45B7D1', alpha=0.9)
    axes[1].bar(xh + w/2, h_rmse_hard, w, label='Hard', color='#FFA07A', alpha=0.9)
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

    # ----- 4. 示例城市时间序列：Actual vs Soft vs Hard -----
    test_dates = get_test_dates(data)
    sample_cities = cities[:4] if len(cities) >= 4 else cities
    max_h = max(horizons)
    n_samples = preds_soft[max_h].shape[0]
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
        actual = targets_soft[max_h][:, -1, cidx]
        soft_pred = preds_soft[max_h][:, -1, cidx]
        hard_pred = preds_hard[max_h][:, -1, cidx]
        a_plot = actual[indices]
        s_plot = soft_pred[indices]
        h_plot = hard_pred[indices]
        ax.plot(dates_plot, a_plot, 'k-', label='Actual', lw=1.5, alpha=0.9)
        ax.plot(dates_plot, s_plot, '--', color='#45B7D1', label='Soft', lw=1.2, alpha=0.8)
        ax.plot(dates_plot, h_plot, '--', color='#FFA07A', label='Hard', lw=1.2, alpha=0.8)
        ax.set_ylabel(f'{city_name}\nCases', fontsize=10)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    axes[-1].set_xlabel('Date', fontsize=12, fontweight='bold')
    plt.suptitle('Soft vs Hard: Example Cities Time Series (max horizon)', fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig_path3 = os.path.join(COMPARE_DIR, 'fig_comparison_timeseries_sample.png')
    plt.savefig(fig_path3, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"示例时间序列图已保存: {fig_path3}")

    print("\n" + "=" * 70)
    print("对比完成。输出目录:", COMPARE_DIR)
    print("  - comparison_table.csv: 按城市、按 horizon 的 MAE/RMSE 及谁更优")
    print("  - comparison_summary_by_horizon.csv: 按 horizon 的平均指标")
    print("  - fig_comparison_mae_rmse_by_city.png: 按城市 MAE/RMSE 柱状对比")
    print("  - fig_comparison_by_horizon.png: 按预测天数 MAE/RMSE 对比")
    print("  - fig_comparison_timeseries_sample.png: 示例城市时间序列")
    print("=" * 70)


if __name__ == "__main__":
    main()
