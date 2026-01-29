#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多步预测评估：评估模型在不同预测步长（3, 7, 14, 30天）下的性能
"""
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from model import EpidemiologyGNN, create_fully_connected_graph
import pandas as pd

def load_model(checkpoint_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """加载训练好的模型"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    
    with open('processed_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    num_cities = len(data['cities'])
    
    model = EpidemiologyGNN(
        num_cities=num_cities,
        gcn_hidden_dim=config['gcn_hidden_dim'],
        gcn_num_layers=config['gcn_num_layers'],
        lstm_hidden_dim=config['lstm_hidden_dim'],
        lstm_num_layers=config['lstm_num_layers'],
        dropout=config['dropout'],
        use_sis=config['use_sis'],
        learnable_sis_params=config['learnable_sis_params']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, data['cities'], config

def predict_multiple_days(model, initial_data, num_days, edge_index, device):
    """预测未来多天的病例数（使用自回归方式）"""
    window_size = initial_data.shape[0]
    predictions = []
    current_window = initial_data.copy()
    
    for day in range(num_days):
        x = torch.FloatTensor(current_window).unsqueeze(0).to(device)
        with torch.no_grad():
            next_day = model(x, edge_index, return_sis=False)
            next_day = next_day.squeeze(0).cpu().numpy()
        predictions.append(next_day)
        # 更新窗口（滑动窗口）
        current_window = np.vstack([current_window[1:], next_day.reshape(1, -1)])
    
    return np.array(predictions)

def calculate_metrics(predictions, targets):
    """计算评估指标"""
    # 确保形状一致
    if predictions.shape != targets.shape:
        min_len = min(len(predictions), len(targets))
        predictions = predictions[:min_len]
        targets = targets[:min_len]
    
    # 计算各种指标
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    
    # MAPE (Mean Absolute Percentage Error)
    # 避免除零
    mask = targets != 0
    if np.sum(mask) > 0:
        mape = np.mean(np.abs((predictions[mask] - targets[mask]) / targets[mask])) * 100
    else:
        mape = np.nan
    
    # RAE (Relative Absolute Error)
    mean_target = np.mean(targets)
    if mean_target != 0:
        rae = np.sum(np.abs(predictions - targets)) / np.sum(np.abs(targets - mean_target))
    else:
        rae = np.nan
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'RAE': rae
    }

def evaluate_multi_step(model, test_X, test_y, edge_index, device, horizon_days):
    """评估特定步长的预测性能"""
    all_predictions = []
    all_targets = []
    
    # 对测试集中的每个样本进行预测
    for i in range(len(test_X) - horizon_days + 1):
        initial_data = test_X[i]
        true_values = test_y[i:i+horizon_days]  # 获取未来horizon_days天的真实值
        
        # 预测
        predictions = predict_multiple_days(model, initial_data, horizon_days, edge_index, device)
        
        all_predictions.append(predictions)
        all_targets.append(true_values)
    
    # 转换为numpy数组
    all_predictions = np.array(all_predictions)  # (n_samples, horizon_days, n_cities)
    all_targets = np.array(all_targets)
    
    # 计算每个时间步的指标
    step_metrics = []
    for step in range(horizon_days):
        step_pred = all_predictions[:, step, :].flatten()
        step_target = all_targets[:, step, :].flatten()
        metrics = calculate_metrics(step_pred, step_target)
        step_metrics.append(metrics)
    
    # 计算总体指标（所有时间步的平均）
    overall_metrics = {}
    for key in ['RMSE', 'MAE', 'MAPE', 'RAE']:
        values = [m[key] for m in step_metrics if not np.isnan(m[key])]
        if values:
            overall_metrics[key] = np.mean(values)
        else:
            overall_metrics[key] = np.nan
    
    return overall_metrics, all_predictions, all_targets

def plot_predictions(predictions, targets, horizon_days, save_path, num_cities_to_plot=5):
    """绘制预测图"""
    # 选择前几个城市进行可视化
    num_cities_to_plot = min(num_cities_to_plot, predictions.shape[2])
    
    fig, axes = plt.subplots(num_cities_to_plot, 1, figsize=(14, 3*num_cities_to_plot))
    if num_cities_to_plot == 1:
        axes = [axes]
    
    # 计算平均预测和真实值（跨样本）
    mean_predictions = np.mean(predictions, axis=0)  # (horizon_days, n_cities)
    mean_targets = np.mean(targets, axis=0)
    
    days = np.arange(1, horizon_days + 1)
    
    for i in range(num_cities_to_plot):
        ax = axes[i]
        city_pred = mean_predictions[:, i]
        city_target = mean_targets[:, i]
        
        ax.plot(days, city_target, 'b-o', label='True', linewidth=2, markersize=6)
        ax.plot(days, city_pred, 'r--s', label='Predicted', linewidth=2, markersize=6, alpha=0.7)
        
        ax.set_xlabel('Days Ahead', fontsize=12)
        ax.set_ylabel('Cases', fontsize=12)
        ax.set_title(f'City {i} - {horizon_days} Days Ahead Prediction', fontsize=13, weight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(days)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"预测图已保存到 {save_path}")
    plt.close()

def create_results_table(results_dict, save_path='multi_step_results.txt'):
    """创建结果表格"""
    horizons = sorted(results_dict.keys())
    
    # 创建表格
    table_lines = []
    table_lines.append("=" * 100)
    table_lines.append("Multi-Step Prediction Results")
    table_lines.append("=" * 100)
    table_lines.append("")
    
    # 表头
    header = "Model".ljust(15)
    for h in horizons:
        header += f"{h}DaysAhead".ljust(40)
    table_lines.append(header)
    
    # 指标行
    metrics = ['RMSE', 'MAE', 'MAPE', 'RAE']
    for metric in metrics:
        row = metric.ljust(15)
        for h in horizons:
            value = results_dict[h].get(metric, np.nan)
            if not np.isnan(value):
                if metric in ['MAPE']:
                    row += f"{value:.2f}".ljust(40)
                else:
                    row += f"{value:.2f}".ljust(40)
            else:
                row += "N/A".ljust(40)
        table_lines.append(row)
    
    table_lines.append("")
    table_lines.append("=" * 100)
    
    # 保存到文件
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(table_lines))
    
    # 打印到控制台
    print('\n'.join(table_lines))
    print(f"\n结果表格已保存到 {save_path}")

def create_latex_table(results_dict, save_path='multi_step_results.tex'):
    """创建LaTeX格式的表格"""
    horizons = sorted(results_dict.keys())
    metrics = ['RMSE', 'MAE', 'MAPE', 'RAE']
    
    lines = []
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\begin{tabular}{l" + "c" * len(horizons) + "}")
    lines.append("\\toprule")
    
    # 表头
    header = "Model"
    for h in horizons:
        header += f" & {h}DaysAhead"
    header += " \\\\"
    lines.append(header)
    lines.append("\\midrule")
    
    # 指标行
    for metric in metrics:
        row = metric
        for h in horizons:
            value = results_dict[h].get(metric, np.nan)
            if not np.isnan(value):
                if metric in ['MAPE']:
                    row += f" & {value:.2f}"
                else:
                    row += f" & {value:.2f}"
            else:
                row += " & N/A"
        row += " \\\\"
        lines.append(row)
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\caption{Multi-step prediction results for different horizons.}")
    lines.append("\\label{tab:multi_step_results}")
    lines.append("\\end{table}")
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"LaTeX表格已保存到 {save_path}")

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 加载模型
    print("\n正在加载模型...")
    model, cities, config = load_model('checkpoints/best_model.pth', device)
    print(f"模型加载完成！城市数量: {len(cities)}")
    
    # 创建图
    edge_index = create_fully_connected_graph(len(cities)).to(device)
    
    # 加载测试数据
    print("\n正在加载测试数据...")
    with open('processed_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    test_X = data['test_X']
    test_y = data['test_y']
    print(f"测试集大小: {len(test_X)}")
    
    # 评估不同步长
    horizons = [3, 7, 14, 30]
    results_dict = {}
    all_predictions_dict = {}
    all_targets_dict = {}
    
    print("\n" + "=" * 80)
    print("开始多步预测评估")
    print("=" * 80)
    
    for horizon in horizons:
        print(f"\n评估 {horizon} 天预测...")
        metrics, predictions, targets = evaluate_multi_step(
            model, test_X, test_y, edge_index, device, horizon
        )
        results_dict[horizon] = metrics
        all_predictions_dict[horizon] = predictions
        all_targets_dict[horizon] = targets
        
        print(f"  RMSE: {metrics['RMSE']:.2f}")
        print(f"  MAE: {metrics['MAE']:.2f}")
        print(f"  MAPE: {metrics['MAPE']:.2f}%")
        print(f"  RAE: {metrics['RAE']:.2f}")
    
    # 生成预测图
    print("\n" + "=" * 80)
    print("生成预测图...")
    print("=" * 80)
    
    for horizon in horizons:
        save_path = f'prediction_{horizon}days.png'
        plot_predictions(
            all_predictions_dict[horizon],
            all_targets_dict[horizon],
            horizon,
            save_path
        )
    
    # 生成结果表格
    print("\n" + "=" * 80)
    print("生成结果表格...")
    print("=" * 80)
    
    create_results_table(results_dict)
    create_latex_table(results_dict)
    
    print("\n" + "=" * 80)
    print("多步预测评估完成！")
    print("=" * 80)

if __name__ == "__main__":
    main()


