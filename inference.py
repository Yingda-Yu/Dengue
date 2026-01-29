"""
使用训练好的模型进行预测
"""
import torch
import pickle
import numpy as np
from model import EpidemiologyGNN, create_fully_connected_graph

def load_model(checkpoint_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """加载训练好的模型"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    
    # 加载数据以获取城市信息
    with open('processed_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    num_cities = len(data['cities'])
    
    # 创建模型
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
    
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, data['cities'], config

def predict_next_day(model, historical_data, edge_index, device):
    """
    预测下一天的病例数
    
    Args:
        model: 训练好的模型
        historical_data: 历史数据，形状 (window_size, num_cities)
        edge_index: 图边索引
        device: 设备
    
    Returns:
        predictions: 预测值，形状 (num_cities,)
    """
    # 添加批次维度
    x = torch.FloatTensor(historical_data).unsqueeze(0).to(device)  # (1, window_size, num_cities)
    
    with torch.no_grad():
        predictions = model(x, edge_index, return_sis=False)
        predictions = predictions.squeeze(0).cpu().numpy()  # (num_cities,)
    
    return predictions

def predict_multiple_days(model, initial_data, num_days, edge_index, device):
    """
    预测未来多天的病例数（使用自回归方式）
    
    Args:
        model: 训练好的模型
        initial_data: 初始数据，形状 (window_size, num_cities)
        num_days: 要预测的天数
        edge_index: 图边索引
        device: 设备
    
    Returns:
        predictions: 预测值，形状 (num_days, num_cities)
    """
    window_size = initial_data.shape[0]
    predictions = []
    current_window = initial_data.copy()
    
    for day in range(num_days):
        # 预测下一天
        next_day = predict_next_day(model, current_window, edge_index, device)
        predictions.append(next_day)
        
        # 更新窗口（滑动窗口）
        current_window = np.vstack([current_window[1:], next_day.reshape(1, -1)])
    
    return np.array(predictions)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='使用训练好的模型进行预测')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth',
                        help='模型检查点路径')
    parser.add_argument('--days', type=int, default=7,
                        help='要预测的天数')
    parser.add_argument('--use_test_data', action='store_true',
                        help='使用测试集数据进行预测')
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 加载模型
    print("正在加载模型...")
    model, cities, config = load_model(args.checkpoint, device)
    print(f"模型加载完成！城市数量: {len(cities)}")
    
    # 创建图
    edge_index = create_fully_connected_graph(len(cities)).to(device)
    
    # 加载数据
    with open('processed_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    if args.use_test_data:
        # 使用测试集的最后一个窗口
        test_X = data['test_X']
        initial_data = test_X[-1]  # 最后一个样本
        true_values = data['test_y'][-1]
        print("\n使用测试集数据进行预测...")
    else:
        # 使用验证集的最后一个窗口
        val_X = data['val_X']
        initial_data = val_X[-1]
        print("\n使用验证集数据进行预测...")
    
    print(f"初始数据形状: {initial_data.shape}")
    print(f"窗口大小: {data['window_size']}")
    
    # 进行预测
    print(f"\n正在预测未来 {args.days} 天...")
    predictions = predict_multiple_days(
        model, initial_data, args.days, edge_index, device
    )
    
    # 打印结果
    print("\n" + "=" * 60)
    print("预测结果")
    print("=" * 60)
    
    for day in range(args.days):
        print(f"\n第 {day + 1} 天预测:")
        print("-" * 60)
        # 显示前10个城市的预测
        top_cities_idx = np.argsort(predictions[day])[-10:][::-1]
        for idx in top_cities_idx:
            city_name = cities[idx]
            pred_value = predictions[day, idx]
            print(f"  {city_name:20s}: {pred_value:8.2f} 例")
    
    # 如果有真实值，计算误差
    if args.use_test_data:
        print("\n" + "=" * 60)
        print("与真实值对比（第1天）:")
        print("=" * 60)
        mae = np.mean(np.abs(predictions[0] - true_values))
        rmse = np.sqrt(np.mean((predictions[0] - true_values) ** 2))
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")

if __name__ == "__main__":
    main()


