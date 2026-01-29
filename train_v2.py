#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
改进版训练脚本
优化点：
1. 数据标准化
2. 组合损失函数（MSE + MAE + 爆发期加权）
3. 梯度裁剪
4. 早停机制
5. 学习率预热 + 余弦退火
6. 更好的评估指标
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
import numpy as np
import pickle
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from model_v2 import EpidemiologyGNNv2, create_fully_connected_graph
import time


class DengueDataset(Dataset):
    """登革热数据集（简化版：只用log1p变换）"""
    def __init__(self, X, y, scaler=None, fit_scaler=False):
        self.X_raw = X.copy()
        self.y_raw = y.copy()
        
        # 使用log1p变换处理长尾分布（简单有效）
        if fit_scaler:
            # 计算log变换后的统计量
            X_log = np.log1p(X)
            self.log_mean = X_log.mean()
            self.log_std = X_log.std() + 1e-8
        elif scaler is not None:
            self.log_mean = scaler['log_mean']
            self.log_std = scaler['log_std']
        else:
            self.log_mean = 0.0
            self.log_std = 1.0
        
        # 变换数据
        self.X = self._transform(X)
        self.y = self._transform(y)
        
        self.X = torch.FloatTensor(self.X)
        self.y = torch.FloatTensor(self.y)
    
    def _transform(self, data):
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
    
    def get_scaler(self):
        return {'log_mean': self.log_mean, 'log_std': self.log_std}
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class CombinedLoss(nn.Module):
    """
    组合损失函数：
    1. MSE损失（整体精度）
    2. MAE损失（鲁棒性）
    3. 爆发期加权损失（关注高病例数时期）
    4. SIS一致性损失
    """
    def __init__(self, lambda_mae=0.3, lambda_outbreak=0.2, lambda_sis=0.1, outbreak_threshold=0.5):
        super().__init__()
        self.lambda_mae = lambda_mae
        self.lambda_outbreak = lambda_outbreak
        self.lambda_sis = lambda_sis
        self.outbreak_threshold = outbreak_threshold
        
        self.mse = nn.MSELoss(reduction='none')
        self.mae = nn.L1Loss(reduction='none')
        self.huber = nn.SmoothL1Loss(reduction='none')
    
    def forward(self, predictions, targets, sis_predictions=None):
        """
        计算组合损失
        """
        # 基础MSE损失
        mse_loss = self.mse(predictions, targets).mean()
        
        # MAE损失（对异常值更鲁棒）
        mae_loss = self.mae(predictions, targets).mean()
        
        # 爆发期加权损失（高病例数时期给予更高权重）
        weights = torch.ones_like(targets)
        outbreak_mask = targets > self.outbreak_threshold
        weights[outbreak_mask] = 2.0  # 爆发期权重加倍
        outbreak_loss = (self.huber(predictions, targets) * weights).mean()
        
        # SIS一致性损失
        sis_loss = torch.tensor(0.0, device=predictions.device)
        if sis_predictions is not None:
            sis_loss = self.mae(predictions, sis_predictions).mean()
        
        # 总损失
        total_loss = (
            mse_loss + 
            self.lambda_mae * mae_loss + 
            self.lambda_outbreak * outbreak_loss +
            self.lambda_sis * sis_loss
        )
        
        return total_loss, mse_loss, mae_loss, sis_loss


class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=15, min_delta=1e-4, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


def train_epoch(model, dataloader, criterion, optimizer, device, edge_index, 
                scaler_amp=None, grad_clip=1.0):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    total_mse = 0.0
    total_mae = 0.0
    total_sis = 0.0
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for X, y in pbar:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        if scaler_amp is not None:
            with torch.amp.autocast('cuda'):
                predictions, sis_predictions = model(X, edge_index, return_sis=True)
                loss, mse, mae, sis = criterion(predictions, y, sis_predictions)
            
            scaler_amp.scale(loss).backward()
            
            # 梯度裁剪
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            scaler_amp.step(optimizer)
            scaler_amp.update()
        else:
            predictions, sis_predictions = model(X, edge_index, return_sis=True)
            loss, mse, mae, sis = criterion(predictions, y, sis_predictions)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            optimizer.step()
        
        total_loss += loss.item()
        total_mse += mse.item()
        total_mae += mae.item()
        total_sis += sis.item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    n_batches = len(dataloader)
    return {
        'loss': total_loss / n_batches,
        'mse': total_mse / n_batches,
        'mae': total_mae / n_batches,
        'sis': total_sis / n_batches
    }


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, edge_index, dataset=None):
    """评估模型"""
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    total_mae = 0.0
    total_sis = 0.0
    all_predictions = []
    all_targets = []
    
    for X, y in tqdm(dataloader, desc="Evaluating", leave=False):
        X = X.to(device)
        y = y.to(device)
        
        predictions, sis_predictions = model(X, edge_index, return_sis=True)
        loss, mse, mae, sis = criterion(predictions, y, sis_predictions)
        
        total_loss += loss.item()
        total_mse += mse.item()
        total_mae += mae.item()
        total_sis += sis.item()
        
        all_predictions.append(predictions.cpu().numpy())
        all_targets.append(y.cpu().numpy())
    
    n_batches = len(dataloader)
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # 反变换到原始尺度计算真实指标
    if dataset is not None:
        all_predictions_orig = dataset.inverse_transform(all_predictions)
        all_targets_orig = dataset.inverse_transform(all_targets)
    else:
        all_predictions_orig = all_predictions
        all_targets_orig = all_targets
    
    # 计算原始尺度的MAE和RMSE
    mae_orig = np.mean(np.abs(all_predictions_orig - all_targets_orig))
    rmse_orig = np.sqrt(np.mean((all_predictions_orig - all_targets_orig) ** 2))
    
    return {
        'loss': total_loss / n_batches,
        'mse': total_mse / n_batches,
        'mae': total_mae / n_batches,
        'sis': total_sis / n_batches,
        'mae_orig': mae_orig,
        'rmse_orig': rmse_orig,
        'predictions': all_predictions,
        'targets': all_targets,
        'predictions_orig': all_predictions_orig,
        'targets_orig': all_targets_orig
    }


def plot_training_curves(history, save_path='training_curves_v2.png'):
    """绘制训练曲线"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # MAE (原始尺度)
    axes[0, 1].plot(epochs, history['val_mae_orig'], 'g-', label='Val MAE')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].set_title('Validation MAE (Original Scale)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # RMSE (原始尺度)
    axes[1, 0].plot(epochs, history['val_rmse_orig'], 'm-', label='Val RMSE')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('RMSE')
    axes[1, 0].set_title('Validation RMSE (Original Scale)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Learning Rate
    axes[1, 1].plot(epochs, history['lr'], 'c-', label='Learning Rate')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('LR')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"训练曲线已保存到 {save_path}")


def main():
    # 配置
    config = {
        'data_path': 'processed_data.pkl',
        'batch_size': 512,  # 大batch加速训练（5090显存充足）
        'learning_rate': 5e-4,  # 稍微降低初始学习率
        'min_lr': 1e-6,  # 最小学习率
        'weight_decay': 1e-5,  # 减小权重衰减
        'num_epochs': 300,  # 更多epoch
        'patience': 50,  # 增加早停耐心
        'grad_clip': 1.0,
        
        # 损失函数权重
        'lambda_mae': 0.5,  # 增加MAE权重
        'lambda_outbreak': 0.3,  # 增加爆发期权重
        'lambda_sis': 0.05,  # 减小SIS权重
        
        # 模型配置
        'spatial_hidden_dim': 128,  # 增加隐藏维度
        'temporal_hidden_dim': 256,  # 增加隐藏维度
        'num_spatial_layers': 3,  # 增加层数
        'num_temporal_layers': 3,  # 增加层数
        'dropout': 0.15,  # 稍微减小dropout
        'use_sis': True,
        
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': 'checkpoints_v2',
        'use_amp': True,
    }
    
    os.makedirs(config['save_dir'], exist_ok=True)
    
    print("=" * 70)
    print("Improved Epidemiology-Informed Spatio-Temporal GNN Training")
    print("=" * 70)
    print(f"设备: {config['device']}")
    print(f"配置: {config}")
    print("=" * 70)
    
    # 加载数据
    print("\n正在加载数据...")
    with open(config['data_path'], 'rb') as f:
        data = pickle.load(f)
    
    train_X, train_y = data['train_X'], data['train_y']
    val_X, val_y = data['val_X'], data['val_y']
    test_X, test_y = data['test_X'], data['test_y']
    cities = data['cities']
    window_size = data['window_size']
    
    num_cities = len(cities)
    print(f"城市数量: {num_cities}")
    print(f"窗口大小: {window_size}")
    print(f"训练集大小: {train_X.shape}")
    print(f"验证集大小: {val_X.shape}")
    print(f"测试集大小: {test_X.shape}")
    
    # 数据统计
    print(f"\n数据统计:")
    print(f"  训练集均值: {train_X.mean():.4f}, 标准差: {train_X.std():.4f}")
    print(f"  训练集最大值: {train_X.max():.4f}")
    
    # 创建数据集（带标准化）
    train_dataset = DengueDataset(train_X, train_y, fit_scaler=True)
    scaler = train_dataset.get_scaler()
    val_dataset = DengueDataset(val_X, val_y, scaler=scaler)
    test_dataset = DengueDataset(test_X, test_y, scaler=scaler)
    
    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'], shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config['batch_size'], shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    
    # 创建图
    print("\n正在创建图结构...")
    edge_index = create_fully_connected_graph(num_cities).to(config['device'])
    print(f"图边数: {edge_index.shape[1]}")
    
    # 创建模型
    print("\n正在初始化模型...")
    model = EpidemiologyGNNv2(
        num_cities=num_cities,
        window_size=window_size,
        spatial_hidden_dim=config['spatial_hidden_dim'],
        temporal_hidden_dim=config['temporal_hidden_dim'],
        num_spatial_layers=config['num_spatial_layers'],
        num_temporal_layers=config['num_temporal_layers'],
        dropout=config['dropout'],
        use_sis=config['use_sis']
    ).to(config['device'])
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数数量: {num_params:,}")
    
    # 损失函数
    criterion = CombinedLoss(
        lambda_mae=config['lambda_mae'],
        lambda_outbreak=config['lambda_outbreak'],
        lambda_sis=config['lambda_sis']
    )
    
    # 优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # 学习率调度器：CosineAnnealingWarmRestarts（周期性重启）
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=30,  # 每30个epoch重启一次
        T_mult=2,  # 每次重启后周期翻倍
        eta_min=config['min_lr']
    )
    
    # 早停（基于原始尺度的MAE，而不是损失）
    early_stopping = EarlyStopping(patience=config['patience'], mode='min')
    
    # 混合精度
    scaler_amp = torch.amp.GradScaler('cuda') if config['use_amp'] else None
    
    # 训练历史
    history = {
        'train_loss': [], 'val_loss': [],
        'val_mae_orig': [], 'val_rmse_orig': [],
        'lr': []
    }
    
    best_val_loss = float('inf')
    best_val_mae = float('inf')
    
    print("\n开始训练...")
    start_time = time.time()
    
    for epoch in range(1, config['num_epochs'] + 1):
        epoch_start = time.time()
        
        # 训练
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, 
            config['device'], edge_index, scaler_amp, config['grad_clip']
        )
        
        # 更新学习率（每个batch后已更新，这里记录当前lr）
        current_lr = optimizer.param_groups[0]['lr']
        
        # 验证
        val_metrics = evaluate(model, val_loader, criterion, 
                              config['device'], edge_index, val_dataset)
        
        # 记录历史
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_mae_orig'].append(val_metrics['mae_orig'])
        history['val_rmse_orig'].append(val_metrics['rmse_orig'])
        history['lr'].append(current_lr)
        
        epoch_time = time.time() - epoch_start
        
        # 打印进度
        print(f"\nEpoch {epoch}/{config['num_epochs']} ({epoch_time:.1f}s)")
        print(f"  Train Loss: {train_metrics['loss']:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}")
        print(f"  Val MAE: {val_metrics['mae_orig']:.4f}, Val RMSE: {val_metrics['rmse_orig']:.4f}")
        print(f"  LR: {current_lr:.6f}")
        
        # 保存最佳模型（基于验证损失）
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'val_mae': val_metrics['mae_orig'],
                'val_rmse': val_metrics['rmse_orig'],
                'config': config,
                'scaler': scaler
            }, os.path.join(config['save_dir'], 'best_model.pth'))
            print(f"  ✓ 保存最佳模型 (Val Loss: {val_metrics['loss']:.4f})")
        
        # 保存最佳MAE模型
        if val_metrics['mae_orig'] < best_val_mae:
            best_val_mae = val_metrics['mae_orig']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_mae': val_metrics['mae_orig'],
                'config': config,
                'scaler': scaler
            }, os.path.join(config['save_dir'], 'best_mae_model.pth'))
            print(f"  ✓ 保存最佳MAE模型 (Val MAE: {val_metrics['mae_orig']:.4f})")
        
        # 更新学习率调度器
        scheduler.step()
        
        # 早停检查（基于原始尺度MAE）
        if early_stopping(val_metrics['mae_orig']):
            print(f"\n早停触发！在 epoch {epoch}")
            break
    
    total_time = time.time() - start_time
    print(f"\n训练完成！总用时: {total_time/60:.1f} 分钟")
    
    # 绘制训练曲线
    plot_training_curves(history, os.path.join(config['save_dir'], 'training_curves.png'))
    
    # 测试
    print("\n" + "=" * 70)
    print("在测试集上评估...")
    print("=" * 70)
    
    # 加载最佳模型
    checkpoint = torch.load(os.path.join(config['save_dir'], 'best_model.pth'), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = evaluate(model, test_loader, criterion, 
                           config['device'], edge_index, test_dataset)
    
    print(f"测试损失: {test_metrics['loss']:.4f}")
    print(f"测试 MAE: {test_metrics['mae_orig']:.4f}")
    print(f"测试 RMSE: {test_metrics['rmse_orig']:.4f}")
    
    # 保存测试结果
    results = {
        'test_loss': test_metrics['loss'],
        'test_mae': test_metrics['mae_orig'],
        'test_rmse': test_metrics['rmse_orig'],
        'test_predictions': test_metrics['predictions_orig'],
        'test_targets': test_metrics['targets_orig'],
        'history': history,
        'config': config,
        'scaler': scaler
    }
    
    with open(os.path.join(config['save_dir'], 'test_results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n测试结果已保存到 {config['save_dir']}/test_results.pkl")
    print("\n训练完成！")


if __name__ == "__main__":
    main()
