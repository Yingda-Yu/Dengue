#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
可视化结果脚本（高性能批量推理版本）- 支持v2模型
生成：
1. 每个城市在不同预测天数（3, 7, 14, 30天）下的MAE和RMSE柱状图
2. 每个城市的具体预测图（展示2020-2024年完整时间序列）

针对高显存GPU（如5090 30G+）优化，使用大batch批量推理
"""
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
from tqdm import tqdm
from model_v2 import EpidemiologyGNNv2, create_fully_connected_graph
import pandas as pd

# 设置matplotlib样式
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 输出文件夹
OUTPUT_DIR = 'visualization_results_v2'

# 批量推理配置 - 针对高显存GPU优化
BATCH_SIZE = 2048  # 5090有30G+显存，可以用很大的batch


def ensure_output_dir():
    """确保输出文件夹存在"""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"创建输出文件夹: {OUTPUT_DIR}")
    return OUTPUT_DIR


class DataTransformer:
    """数据变换器（与训练时保持一致）"""
    def __init__(self, scaler):
        self.log_mean = scaler['log_mean']
        self.log_std = scaler['log_std']
    
    def transform(self, data):
        """log1p变换 + 标准化"""
        data_log = np.log1p(data)
        return (data_log - self.log_mean) / self.log_std
    
    def inverse_transform(self, data):
        """反变换"""
        if isinstance(data, torch.Tensor):
            data = data.numpy()
        # 反标准化
        data = data * self.log_std + self.log_mean
        # 反log变换（确保非负）
        return np.maximum(np.expm1(data), 0)


def load_model(checkpoint_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """加载训练好的v2模型"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    scaler = checkpoint['scaler']
    
    with open('processed_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    num_cities = len(data['cities'])
    window_size = data['window_size']
    
    model = EpidemiologyGNNv2(
        num_cities=num_cities,
        window_size=window_size,
        spatial_hidden_dim=config['spatial_hidden_dim'],
        temporal_hidden_dim=config['temporal_hidden_dim'],
        num_spatial_layers=config['num_spatial_layers'],
        num_temporal_layers=config['num_temporal_layers'],
        dropout=config['dropout'],
        use_sis=config['use_sis']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 创建数据变换器
    transformer = DataTransformer(scaler)
    
    return model, data, config, transformer


def batch_predict_one_step(model, batch_windows, edge_index, device):
    """
    批量预测下一天（单步）
    """
    x = torch.FloatTensor(batch_windows).to(device)
    with torch.no_grad():
        predictions = model(x, edge_index, return_sis=False)
    return predictions.cpu().numpy()


def batch_predict_multiple_days(model, initial_windows, num_days, edge_index, device, batch_size=BATCH_SIZE):
    """
    批量预测未来多天（自回归方式，高效批量版本）
    """
    n_samples = len(initial_windows)
    window_size = initial_windows.shape[1]
    n_cities = initial_windows.shape[2]
    
    all_predictions = np.zeros((n_samples, num_days, n_cities), dtype=np.float32)
    current_windows = initial_windows.copy()
    
    for day in range(num_days):
        day_predictions = np.zeros((n_samples, n_cities), dtype=np.float32)
        
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_windows = current_windows[start_idx:end_idx]
            batch_pred = batch_predict_one_step(model, batch_windows, edge_index, device)
            day_predictions[start_idx:end_idx] = batch_pred
        
        all_predictions[:, day, :] = day_predictions
        current_windows = np.concatenate([
            current_windows[:, 1:, :],
            day_predictions[:, np.newaxis, :]
        ], axis=1)
    
    return all_predictions


def get_test_dates(data):
    """
    获取测试集对应的日期范围
    """
    dates = data['dates']
    window_size = data['window_size']
    
    n_train = len(data['train_X'])
    n_val = len(data['val_X'])
    n_test = len(data['test_X'])
    
    # 测试集开始的日期索引
    test_start_idx = n_train + n_val + window_size
    test_end_idx = test_start_idx + n_test
    
    # 确保不超出范围
    test_end_idx = min(test_end_idx, len(dates))
    
    test_dates = dates[test_start_idx:test_end_idx]
    return test_dates


def evaluate_multi_step_by_city(model, test_X, test_y, edge_index, device, cities, transformer, horizons=[3, 7, 14, 30]):
    """
    按城市评估不同预测步长的性能（高效批量版本）
    返回每个时间点的预测值，用于绘制完整时间序列
    
    注意：test_X和test_y是原始数据，需要先变换再预测，预测后再反变换
    """
    city_metrics = {city: {} for city in cities}
    all_predictions = {}
    all_targets = {}
    
    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"批次大小: {BATCH_SIZE}")
    
    for horizon in horizons:
        print(f"\n{'='*60}")
        print(f"评估 {horizon} 天预测...")
        print(f"{'='*60}")
        
        n_samples = len(test_X) - horizon + 1
        print(f"样本数量: {n_samples}")
        
        # 对输入数据进行变换
        initial_windows_raw = test_X[:n_samples]
        initial_windows = transformer.transform(initial_windows_raw)
        
        # 构建目标数组（原始尺度）
        targets_raw = np.zeros((n_samples, horizon, len(cities)), dtype=np.float32)
        for i in range(n_samples):
            targets_raw[i] = test_y[i:i+horizon]
        
        print(f"开始批量预测（batch_size={BATCH_SIZE}）...")
        # 预测（在变换后的空间中）
        predictions_transformed = batch_predict_multiple_days(
            model, initial_windows, horizon, edge_index, device, BATCH_SIZE
        )
        
        # 反变换预测结果到原始尺度
        predictions_raw = transformer.inverse_transform(predictions_transformed)
        
        if torch.cuda.is_available():
            print(f"峰值显存使用: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
        
        all_predictions[horizon] = predictions_raw
        all_targets[horizon] = targets_raw
        
        # 计算每个城市的MAE和RMSE（原始尺度）
        for city_idx, city_name in enumerate(cities):
            city_pred = predictions_raw[:, :, city_idx].flatten()
            city_tgt = targets_raw[:, :, city_idx].flatten()
            
            mae = np.mean(np.abs(city_pred - city_tgt))
            rmse = np.sqrt(np.mean((city_pred - city_tgt) ** 2))
            
            city_metrics[city_name][horizon] = {
                'MAE': mae,
                'RMSE': rmse
            }
        
        all_mae = [city_metrics[city][horizon]['MAE'] for city in cities]
        all_rmse = [city_metrics[city][horizon]['RMSE'] for city in cities]
        print(f"✓ {horizon}天预测完成 - 平均MAE: {np.mean(all_mae):.4f}, 平均RMSE: {np.mean(all_rmse):.4f}")
    
    return city_metrics, all_predictions, all_targets


def plot_city_metrics_bar(city_metrics, cities, horizons, output_dir):
    """
    图1: 每个城市在不同预测天数下的MAE和RMSE柱状图
    """
    n_cities = len(cities)
    n_horizons = len(horizons)
    
    mae_data = np.zeros((n_cities, n_horizons))
    rmse_data = np.zeros((n_cities, n_horizons))
    
    for i, city in enumerate(cities):
        for j, horizon in enumerate(horizons):
            if horizon in city_metrics[city]:
                mae_data[i, j] = city_metrics[city][horizon]['MAE']
                rmse_data[i, j] = city_metrics[city][horizon]['RMSE']
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    
    # MAE 图
    fig, ax = plt.subplots(figsize=(max(14, n_cities * 0.8), 8))
    x = np.arange(n_cities)
    width = 0.8 / n_horizons
    
    for j, horizon in enumerate(horizons):
        offset = (j - n_horizons / 2 + 0.5) * width
        ax.bar(x + offset, mae_data[:, j], width, 
               label=f'{horizon} Days Ahead', color=colors[j], alpha=0.8)
    
    ax.set_xlabel('City', fontsize=12, fontweight='bold')
    ax.set_ylabel('MAE (Mean Absolute Error)', fontsize=12, fontweight='bold')
    ax.set_title('MAE by City and Forecast Horizon', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(cities, rotation=45, ha='right', fontsize=9)
    ax.legend(title='Forecast Horizon', fontsize=10, title_fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'fig1_mae_by_city_horizon.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"MAE柱状图已保存到: {save_path}")
    plt.close()
    
    # RMSE 图
    fig, ax = plt.subplots(figsize=(max(14, n_cities * 0.8), 8))
    
    for j, horizon in enumerate(horizons):
        offset = (j - n_horizons / 2 + 0.5) * width
        ax.bar(x + offset, rmse_data[:, j], width, 
               label=f'{horizon} Days Ahead', color=colors[j], alpha=0.8)
    
    ax.set_xlabel('City', fontsize=12, fontweight='bold')
    ax.set_ylabel('RMSE (Root Mean Square Error)', fontsize=12, fontweight='bold')
    ax.set_title('RMSE by City and Forecast Horizon', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(cities, rotation=45, ha='right', fontsize=9)
    ax.legend(title='Forecast Horizon', fontsize=10, title_fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'fig1_rmse_by_city_horizon.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"RMSE柱状图已保存到: {save_path}")
    plt.close()
    
    # MAE 和 RMSE 组合图
    fig, axes = plt.subplots(2, 1, figsize=(max(16, n_cities * 0.9), 14))
    
    ax = axes[0]
    for j, horizon in enumerate(horizons):
        offset = (j - n_horizons / 2 + 0.5) * width
        ax.bar(x + offset, mae_data[:, j], width, 
               label=f'{horizon} Days', color=colors[j], alpha=0.8)
    
    ax.set_xlabel('City', fontsize=12, fontweight='bold')
    ax.set_ylabel('MAE', fontsize=12, fontweight='bold')
    ax.set_title('MAE by City and Forecast Horizon', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(cities, rotation=45, ha='right', fontsize=9)
    ax.legend(title='Horizon', fontsize=9, title_fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    ax = axes[1]
    for j, horizon in enumerate(horizons):
        offset = (j - n_horizons / 2 + 0.5) * width
        ax.bar(x + offset, rmse_data[:, j], width, 
               label=f'{horizon} Days', color=colors[j], alpha=0.8)
    
    ax.set_xlabel('City', fontsize=12, fontweight='bold')
    ax.set_ylabel('RMSE', fontsize=12, fontweight='bold')
    ax.set_title('RMSE by City and Forecast Horizon', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(cities, rotation=45, ha='right', fontsize=9)
    ax.legend(title='Horizon', fontsize=9, title_fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'fig1_mae_rmse_combined.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"MAE和RMSE组合图已保存到: {save_path}")
    plt.close()


def plot_city_time_series(all_predictions, all_targets, cities, horizons, test_dates, output_dir):
    """
    图2: 每个城市的完整时间序列预测图（2020-2024年）
    展示真实值和不同提前天数的预测值
    """
    city_dir = os.path.join(output_dir, 'city_predictions')
    if not os.path.exists(city_dir):
        os.makedirs(city_dir)
    
    colors = {
        3: '#FF6B6B',   # 红色 - 3天预测
        7: '#4ECDC4',   # 青色 - 7天预测
        14: '#45B7D1',  # 蓝色 - 14天预测
        30: '#FFA07A'   # 橙色 - 30天预测
    }
    
    # 对于每个horizon，提取第horizon天的预测（即提前horizon天的预测）
    # predictions[horizon] 形状: (n_samples, horizon_days, n_cities)
    # 我们要取最后一天的预测，即 predictions[horizon][:, -1, :]
    
    for city_idx, city_name in enumerate(tqdm(cities, desc="生成城市时间序列图")):
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # 获取真实值（使用最长horizon的targets来获取完整序列）
        max_horizon = max(horizons)
        n_samples = all_targets[max_horizon].shape[0]
        
        # 真实值：取每个样本的最后一天目标值
        actual_values = all_targets[max_horizon][:, -1, city_idx]
        
        # 对应的日期
        plot_dates = test_dates[max_horizon-1:max_horizon-1+n_samples]
        
        # 确保日期和值的长度一致
        min_len = min(len(plot_dates), len(actual_values))
        plot_dates = plot_dates[:min_len]
        actual_values = actual_values[:min_len]
        
        # 绘制真实值
        ax.plot(plot_dates, actual_values, 'k-', label='Actual', linewidth=2, alpha=0.8)
        
        # 绘制不同提前天数的预测
        for horizon in horizons:
            # 取提前horizon天的预测（即预测序列的最后一天）
            pred_values = all_predictions[horizon][:, -1, city_idx]
            
            # 对应的日期需要对齐
            pred_n_samples = len(pred_values)
            pred_dates = test_dates[horizon-1:horizon-1+pred_n_samples]
            
            min_pred_len = min(len(pred_dates), len(pred_values))
            pred_dates = pred_dates[:min_pred_len]
            pred_values = pred_values[:min_pred_len]
            
            ax.plot(pred_dates, pred_values, '--', color=colors[horizon], 
                   label=f'{horizon}-Day Ahead Forecast', linewidth=1.5, alpha=0.7)
        
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Dengue Cases', fontsize=12, fontweight='bold')
        ax.set_title(f'{city_name} - Dengue Forecast (2020-2024)\nActual vs Predicted at Different Horizons', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 格式化x轴日期
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        safe_city_name = city_name.replace('/', '_').replace(' ', '_')
        save_path = os.path.join(city_dir, f'fig2_{safe_city_name}_timeseries.png')
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
    
    print(f"所有城市时间序列图已保存到: {city_dir}")
    
    # 生成汇总图（选取部分城市）
    num_cities_to_show = min(6, len(cities))
    fig, axes = plt.subplots(num_cities_to_show, 1, figsize=(16, 4*num_cities_to_show))
    if num_cities_to_show == 1:
        axes = [axes]
    
    for city_idx in range(num_cities_to_show):
        city_name = cities[city_idx]
        ax = axes[city_idx]
        
        max_horizon = max(horizons)
        n_samples = all_targets[max_horizon].shape[0]
        actual_values = all_targets[max_horizon][:, -1, city_idx]
        plot_dates = test_dates[max_horizon-1:max_horizon-1+n_samples]
        
        min_len = min(len(plot_dates), len(actual_values))
        plot_dates = plot_dates[:min_len]
        actual_values = actual_values[:min_len]
        
        ax.plot(plot_dates, actual_values, 'k-', label='Actual', linewidth=2, alpha=0.8)
        
        for horizon in horizons:
            pred_values = all_predictions[horizon][:, -1, city_idx]
            pred_n_samples = len(pred_values)
            pred_dates = test_dates[horizon-1:horizon-1+pred_n_samples]
            
            min_pred_len = min(len(pred_dates), len(pred_values))
            pred_dates = pred_dates[:min_pred_len]
            pred_values = pred_values[:min_pred_len]
            
            ax.plot(pred_dates, pred_values, '--', color=colors[horizon], 
                   label=f'{horizon}-Day', linewidth=1.2, alpha=0.7)
        
        ax.set_ylabel(f'{city_name}\nCases', fontsize=10)
        ax.legend(loc='upper right', fontsize=8, ncol=5)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        
        if city_idx == num_cities_to_show - 1:
            ax.set_xlabel('Date', fontsize=12, fontweight='bold')
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        else:
            ax.set_xticklabels([])
    
    plt.suptitle('Dengue Forecast Time Series (2020-2024) - Sample Cities', 
                fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'fig2_city_timeseries_summary.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"城市时间序列汇总图已保存到: {save_path}")
    plt.close()


def plot_horizon_comparison(all_predictions, all_targets, cities, horizons, test_dates, output_dir):
    """
    图3: 每个城市分别展示不同预测天数的子图（2x2布局）
    """
    city_dir = os.path.join(output_dir, 'city_predictions')
    
    colors = {
        3: '#FF6B6B',
        7: '#4ECDC4',
        14: '#45B7D1',
        30: '#FFA07A'
    }
    
    for city_idx, city_name in enumerate(tqdm(cities, desc="生成城市多horizon对比图")):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for h_idx, horizon in enumerate(horizons):
            ax = axes[h_idx]
            
            # 获取该horizon的预测和真实值
            pred = all_predictions[horizon][:, -1, city_idx]  # 取最后一天预测
            tgt = all_targets[horizon][:, -1, city_idx]  # 取最后一天目标
            
            n_samples = len(pred)
            dates = test_dates[horizon-1:horizon-1+n_samples]
            
            min_len = min(len(dates), len(pred), len(tgt))
            dates = dates[:min_len]
            pred = pred[:min_len]
            tgt = tgt[:min_len]
            
            ax.plot(dates, tgt, 'k-', label='Actual', linewidth=1.5, alpha=0.8)
            ax.plot(dates, pred, '--', color=colors[horizon], 
                   label=f'{horizon}-Day Forecast', linewidth=1.5, alpha=0.7)
            
            # 计算该horizon的指标
            mae = np.mean(np.abs(pred - tgt))
            rmse = np.sqrt(np.mean((pred - tgt) ** 2))
            
            ax.set_title(f'{horizon}-Day Ahead Forecast\nMAE: {mae:.2f}, RMSE: {rmse:.2f}', 
                        fontsize=12, fontweight='bold')
            ax.set_xlabel('Date', fontsize=10)
            ax.set_ylabel('Cases', fontsize=10)
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.suptitle(f'{city_name} - Multi-Horizon Forecast Comparison (2020-2024)', 
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        safe_city_name = city_name.replace('/', '_').replace(' ', '_')
        save_path = os.path.join(city_dir, f'fig3_{safe_city_name}_horizon_comparison.png')
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
    
    print(f"所有城市多horizon对比图已保存到: {city_dir}")


def save_metrics_to_csv(city_metrics, cities, horizons, output_dir):
    """保存指标到CSV文件"""
    import csv
    
    mae_path = os.path.join(output_dir, 'mae_by_city_horizon.csv')
    with open(mae_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        header = ['City'] + [f'{h}_Days_Ahead' for h in horizons]
        writer.writerow(header)
        for city in cities:
            row = [city] + [f"{city_metrics[city][h]['MAE']:.4f}" for h in horizons]
            writer.writerow(row)
    print(f"MAE数据已保存到: {mae_path}")
    
    rmse_path = os.path.join(output_dir, 'rmse_by_city_horizon.csv')
    with open(rmse_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        header = ['City'] + [f'{h}_Days_Ahead' for h in horizons]
        writer.writerow(header)
        for city in cities:
            row = [city] + [f"{city_metrics[city][h]['RMSE']:.4f}" for h in horizons]
            writer.writerow(row)
    print(f"RMSE数据已保存到: {rmse_path}")


def main():
    output_dir = ensure_output_dir()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    print("\n正在加载模型（v2版本）...")
    model, data, config, transformer = load_model('checkpoints_v2/best_model.pth', device)
    
    cities = data['cities']
    test_X = data['test_X']
    test_y = data['test_y']
    
    # 获取测试集的日期
    test_dates = get_test_dates(data)
    
    print(f"模型加载完成！")
    print(f"城市数量: {len(cities)}")
    print(f"测试集大小: {len(test_X)}")
    print(f"测试集日期范围: {test_dates[0]} 到 {test_dates[-1]}")
    
    edge_index = create_fully_connected_graph(len(cities)).to(device)
    
    horizons = [3, 7, 14, 30]
    
    print("\n" + "=" * 80)
    print("开始多步预测评估（2020-2024年测试集）...")
    print("=" * 80)
    
    city_metrics, all_predictions, all_targets = evaluate_multi_step_by_city(
        model, test_X, test_y, edge_index, device, cities, transformer, horizons
    )
    
    print("\n" + "=" * 80)
    print("评估结果汇总")
    print("=" * 80)
    
    for horizon in horizons:
        all_mae = [city_metrics[city][horizon]['MAE'] for city in cities]
        all_rmse = [city_metrics[city][horizon]['RMSE'] for city in cities]
        print(f"\n{horizon}天预测:")
        print(f"  平均MAE: {np.mean(all_mae):.4f} (std: {np.std(all_mae):.4f})")
        print(f"  平均RMSE: {np.mean(all_rmse):.4f} (std: {np.std(all_rmse):.4f})")
    
    print("\n" + "=" * 80)
    print("生成可视化图表...")
    print("=" * 80)
    
    # 图1: MAE和RMSE柱状图
    print("\n生成图1: MAE和RMSE柱状图...")
    plot_city_metrics_bar(city_metrics, cities, horizons, output_dir)
    
    # 图2: 完整时间序列预测图（2020-2024）
    print("\n生成图2: 城市时间序列预测图（2020-2024）...")
    plot_city_time_series(all_predictions, all_targets, cities, horizons, test_dates, output_dir)
    
    # 图3: 每个城市的多horizon对比图
    print("\n生成图3: 城市多horizon对比图...")
    plot_horizon_comparison(all_predictions, all_targets, cities, horizons, test_dates, output_dir)
    
    # 保存指标到CSV
    print("\n保存指标数据...")
    save_metrics_to_csv(city_metrics, cities, horizons, output_dir)
    
    if torch.cuda.is_available():
        print(f"\n最终显存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"峰值显存使用: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    
    print("\n" + "=" * 80)
    print("可视化完成！")
    print("=" * 80)
    print(f"\n所有结果已保存到文件夹: {output_dir}")
    print(f"  - fig1_mae_by_city_horizon.png: MAE柱状图")
    print(f"  - fig1_rmse_by_city_horizon.png: RMSE柱状图")
    print(f"  - fig1_mae_rmse_combined.png: MAE和RMSE组合图")
    print(f"  - fig2_city_timeseries_summary.png: 城市时间序列汇总图（2020-2024）")
    print(f"  - city_predictions/fig2_*_timeseries.png: 每个城市的完整时间序列图")
    print(f"  - city_predictions/fig3_*_horizon_comparison.png: 每个城市的多horizon对比图")
    print(f"  - mae_by_city_horizon.csv: MAE数据表")
    print(f"  - rmse_by_city_horizon.csv: RMSE数据表")


if __name__ == "__main__":
    main()
