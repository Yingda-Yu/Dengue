"""
使用训练好的模型进行预测（支持软约束/硬约束模型选择）
"""
import torch
import pickle
import numpy as np
from model import EpidemiologyGNNv2, create_fully_connected_graph


class DataTransformer:
    """与训练时一致的数据变换"""
    def __init__(self, scaler):
        self.log_mean = scaler['log_mean']
        self.log_std = scaler['log_std']

    def transform(self, data):
        data_log = np.log1p(data)
        return (data_log - self.log_mean) / self.log_std

    def inverse_transform(self, data):
        if isinstance(data, torch.Tensor):
            data = data.numpy()
        data = data * self.log_std + self.log_mean
        return np.maximum(np.expm1(data), 0)


def load_model(checkpoint_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """加载 v2 模型，支持软/硬约束 checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    scaler = checkpoint['scaler']
    config['use_hard_constraint'] = config.get('use_hard_constraint', False)

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
    transformer = DataTransformer(scaler)
    return model, data['cities'], config, transformer


def predict_next_day(model, historical_data, edge_index, device, use_hard_constraint=False):
    """
    预测下一天（historical_data 应为已变换后的数据）
    返回变换空间预测值（用于自回归下一步），调用方需自行 inverse 得到原始尺度
    """
    x = torch.FloatTensor(historical_data).unsqueeze(0).to(device)
    with torch.no_grad():
        predictions = model(x, edge_index, return_sis=False, use_hard_constraint=use_hard_constraint)
    return predictions.squeeze(0).cpu().numpy()


def predict_multiple_days(model, initial_data_transformed, num_days, edge_index, device, transformer, use_hard_constraint=False):
    """
    预测未来多天（自回归）。initial_data_transformed 为已变换的 (window_size, num_cities）
    返回 (num_days, num_cities) 原始尺度的预测
    """
    current_window = np.array(initial_data_transformed, dtype=np.float32)
    predictions_orig = []
    for day in range(num_days):
        pred_t = predict_next_day(model, current_window, edge_index, device, use_hard_constraint=use_hard_constraint)
        pred_orig = transformer.inverse_transform(pred_t)
        predictions_orig.append(pred_orig)
        current_window = np.vstack([current_window[1:], pred_t])
    return np.array(predictions_orig)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='使用训练好的模型进行预测')
    parser.add_argument('--checkpoint', type=str, default=None, help='模型检查点路径（若指定则覆盖 --model）')
    parser.add_argument('--model', type=str, choices=['soft', 'hard'], default='soft', help='软约束或硬约束权重')
    parser.add_argument('--days', type=int, default=7, help='要预测的天数')
    parser.add_argument('--use_test_data', action='store_true', help='使用测试集数据进行预测')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    checkpoint_path = args.checkpoint or f'checkpoints/best_model_{args.model}.pth'
    print(f"正在加载模型: {checkpoint_path}")
    model, cities, config, transformer = load_model(checkpoint_path, device)
    use_hard_constraint = config.get('use_hard_constraint', False)
    print(f"模型加载完成！城市数量: {len(cities)}, 硬约束: {use_hard_constraint}")

    edge_index = create_fully_connected_graph(len(cities)).to(device)

    with open('processed_data.pkl', 'rb') as f:
        data = pickle.load(f)

    if args.use_test_data:
        test_X = data['test_X']
        initial_data_raw = test_X[-1]
        true_values = data['test_y'][-1]
        print("\n使用测试集数据进行预测...")
    else:
        val_X = data['val_X']
        initial_data_raw = val_X[-1]
        true_values = None
        print("\n使用验证集数据进行预测...")

    initial_data = transformer.transform(initial_data_raw)
    if initial_data.ndim == 3:
        initial_data = initial_data[0]
    print(f"初始数据形状: {initial_data.shape}, 窗口大小: {data['window_size']}")

    print(f"\n正在预测未来 {args.days} 天...")
    predictions = predict_multiple_days(
        model, initial_data, args.days, edge_index, device, transformer, use_hard_constraint=use_hard_constraint
    )

    print("\n" + "=" * 60)
    print("预测结果")
    print("=" * 60)

    for day in range(args.days):
        print(f"\n第 {day + 1} 天预测:")
        print("-" * 60)
        top_cities_idx = np.argsort(predictions[day])[-10:][::-1]
        for idx in top_cities_idx:
            city_name = cities[idx]
            pred_value = predictions[day, idx]
            print(f"  {city_name:20s}: {pred_value:8.2f} 例")

    if args.use_test_data and true_values is not None:
        print("\n" + "=" * 60)
        print("与真实值对比（第1天）:")
        print("=" * 60)
        mae = np.mean(np.abs(predictions[0] - true_values))
        rmse = np.sqrt(np.mean((predictions[0] - true_values) ** 2))
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")


if __name__ == "__main__":
    main()
