"""
训练脚本：Epidemiology-Informed Spatio-Temporal GNN with SIS Regularization
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import EpidemiologyGNN, create_fully_connected_graph

class DengueDataset(Dataset):
    """登革热数据集"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LossFunction(nn.Module):
    """损失函数：数据保真度 + SIS正则化"""
    def __init__(self, lambda_sis=0.1):
        super(LossFunction, self).__init__()
        self.lambda_sis = lambda_sis
        self.mse_loss = nn.MSELoss()
    
    def forward(self, predictions, targets, sis_predictions=None):
        """
        计算总损失
        
        Args:
            predictions: 模型预测，形状 (batch, num_cities)
            targets: 真实值，形状 (batch, num_cities)
            sis_predictions: SIS模型预测，形状 (batch, num_cities)
        
        Returns:
            total_loss: 总损失
            data_loss: 数据保真度损失
            sis_loss: SIS正则化损失
        """
        # 数据保真度损失
        data_loss = self.mse_loss(predictions, targets)
        
        # SIS一致性损失
        sis_loss = torch.tensor(0.0, device=predictions.device)
        if sis_predictions is not None:
            sis_loss = self.mse_loss(predictions, sis_predictions)
        
        # 总损失
        total_loss = data_loss + self.lambda_sis * sis_loss
        
        return total_loss, data_loss, sis_loss

def train_epoch(model, dataloader, criterion, optimizer, device, edge_index, use_amp=False):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    total_data_loss = 0.0
    total_sis_loss = 0.0
    
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    
    for X, y in tqdm(dataloader, desc="Training"):
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        if use_amp:
            # 混合精度训练
            with torch.amp.autocast('cuda'):
                predictions, sis_predictions = model(X, edge_index, return_sis=True)
                loss, data_loss, sis_loss = criterion(predictions, y, sis_predictions)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # 标准训练
            predictions, sis_predictions = model(X, edge_index, return_sis=True)
            loss, data_loss, sis_loss = criterion(predictions, y, sis_predictions)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        total_data_loss += data_loss.item()
        total_sis_loss += sis_loss.item()
    
    n_batches = len(dataloader)
    return (
        total_loss / n_batches,
        total_data_loss / n_batches,
        total_sis_loss / n_batches
    )

def evaluate(model, dataloader, criterion, device, edge_index):
    """评估模型"""
    model.eval()
    total_loss = 0.0
    total_data_loss = 0.0
    total_sis_loss = 0.0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for X, y in tqdm(dataloader, desc="Evaluating"):
            X = X.to(device)
            y = y.to(device)
            
            predictions, sis_predictions = model(X, edge_index, return_sis=True)
            
            loss, data_loss, sis_loss = criterion(predictions, y, sis_predictions)
            
            total_loss += loss.item()
            total_data_loss += data_loss.item()
            total_sis_loss += sis_loss.item()
            
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    
    n_batches = len(dataloader)
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # 计算MAE和RMSE
    mae = np.mean(np.abs(all_predictions - all_targets))
    rmse = np.sqrt(np.mean((all_predictions - all_targets) ** 2))
    
    return (
        total_loss / n_batches,
        total_data_loss / n_batches,
        total_sis_loss / n_batches,
        mae,
        rmse,
        all_predictions,
        all_targets
    )

def plot_training_curves(train_losses, val_losses, save_path='training_curves.png'):
    """绘制训练曲线"""
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"训练曲线已保存到 {save_path}")

def main():
    # 配置
    config = {
        'data_path': 'processed_data.pkl',
        'batch_size': 128,  # 增大批次大小以加速训练
        'learning_rate': 0.001,
        'num_epochs': 50,
        'lambda_sis': 0.1,  # SIS正则化权重
        'gcn_hidden_dim': 64,
        'gcn_num_layers': 2,
        'lstm_hidden_dim': 128,
        'lstm_num_layers': 2,
        'dropout': 0.1,
        'use_sis': True,
        'learnable_sis_params': True,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': 'checkpoints',
        'use_amp': True,  # 启用混合精度训练
        'pin_memory': True  # 优化数据加载
    }
    
    # 创建保存目录
    os.makedirs(config['save_dir'], exist_ok=True)
    
    print("=" * 60)
    print("Epidemiology-Informed Spatio-Temporal GNN Training")
    print("=" * 60)
    print(f"设备: {config['device']}")
    print(f"配置: {config}")
    print("=" * 60)
    
    # 加载数据
    print("\n正在加载数据...")
    with open(config['data_path'], 'rb') as f:
        data = pickle.load(f)
    
    train_X, train_y = data['train_X'], data['train_y']
    val_X, val_y = data['val_X'], data['val_y']
    test_X, test_y = data['test_X'], data['test_y']
    cities = data['cities']
    
    num_cities = len(cities)
    print(f"城市数量: {num_cities}")
    print(f"训练集大小: {train_X.shape[0]}")
    print(f"验证集大小: {val_X.shape[0]}")
    print(f"测试集大小: {test_X.shape[0]}")
    
    # 创建数据集和数据加载器
    train_dataset = DengueDataset(train_X, train_y)
    val_dataset = DengueDataset(val_X, val_y)
    test_dataset = DengueDataset(test_X, test_y)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=config.get('pin_memory', False) if config['device'] == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=config.get('pin_memory', False) if config['device'] == 'cuda' else False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=config.get('pin_memory', False) if config['device'] == 'cuda' else False
    )
    
    # 创建图
    print("\n正在创建图结构...")
    edge_index = create_fully_connected_graph(num_cities)
    edge_index = edge_index.to(config['device'])
    print(f"图边数: {edge_index.shape[1]}")
    
    # 创建模型
    print("\n正在初始化模型...")
    model = EpidemiologyGNN(
        num_cities=num_cities,
        gcn_hidden_dim=config['gcn_hidden_dim'],
        gcn_num_layers=config['gcn_num_layers'],
        lstm_hidden_dim=config['lstm_hidden_dim'],
        lstm_num_layers=config['lstm_num_layers'],
        dropout=config['dropout'],
        use_sis=config['use_sis'],
        learnable_sis_params=config['learnable_sis_params']
    ).to(config['device'])
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 损失函数和优化器
    criterion = LossFunction(lambda_sis=config['lambda_sis'])
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # 训练循环
    print("\n开始训练...")
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(1, config['num_epochs'] + 1):
        print(f"\nEpoch {epoch}/{config['num_epochs']}")
        
        # 训练
        train_loss, train_data_loss, train_sis_loss = train_epoch(
            model, train_loader, criterion, optimizer, config['device'], edge_index,
            use_amp=config.get('use_amp', False)
        )
        
        # 验证
        val_loss, val_data_loss, val_sis_loss, val_mae, val_rmse, _, _ = evaluate(
            model, val_loader, criterion, config['device'], edge_index
        )
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"训练损失: {train_loss:.4f} (数据: {train_data_loss:.4f}, SIS: {train_sis_loss:.4f})")
        print(f"验证损失: {val_loss:.4f} (数据: {val_data_loss:.4f}, SIS: {val_sis_loss:.4f})")
        print(f"验证 MAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}")
        
        # 学习率调度
        scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config
            }, os.path.join(config['save_dir'], 'best_model.pth'))
            print(f"✓ 保存最佳模型 (验证损失: {val_loss:.4f})")
    
    # 绘制训练曲线
    plot_training_curves(train_losses, val_losses)
    
    # 测试
    print("\n" + "=" * 60)
    print("在测试集上评估...")
    print("=" * 60)
    
    # 加载最佳模型
    checkpoint = torch.load(os.path.join(config['save_dir'], 'best_model.pth'), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_data_loss, test_sis_loss, test_mae, test_rmse, test_pred, test_true = evaluate(
        model, test_loader, criterion, config['device'], edge_index
    )
    
    print(f"测试损失: {test_loss:.4f} (数据: {test_data_loss:.4f}, SIS: {test_sis_loss:.4f})")
    print(f"测试 MAE: {test_mae:.4f}")
    print(f"测试 RMSE: {test_rmse:.4f}")
    
    # 保存测试结果
    results = {
        'test_loss': test_loss,
        'test_data_loss': test_data_loss,
        'test_sis_loss': test_sis_loss,
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'test_predictions': test_pred,
        'test_targets': test_true
    }
    
    with open(os.path.join(config['save_dir'], 'test_results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n测试结果已保存到 {config['save_dir']}/test_results.pkl")
    print("\n训练完成！")

if __name__ == "__main__":
    main()


