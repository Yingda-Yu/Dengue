"""
Epidemiology-Informed Spatio-Temporal Graph Neural Network with SIS Regularization
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from torch_geometric.nn import GCNConv
except ImportError:
    # 如果torch-geometric不可用，使用简单的线性层替代
    print("警告: torch-geometric未安装，将使用简化的图卷积实现")
    class GCNConv(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.linear = nn.Linear(in_channels, out_channels)
        def forward(self, x, edge_index):
            return self.linear(x)
import numpy as np

class GCNLayer(nn.Module):
    """图卷积层（兼容版本）"""
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        try:
            self.conv = GCNConv(in_features, out_features)
            self.use_pyg = True
        except:
            # 如果PyG不可用，使用简单的线性层
            self.linear = nn.Linear(in_features, out_features)
            self.use_pyg = False
    
    def forward(self, x, edge_index):
        if self.use_pyg:
            return F.relu(self.conv(x, edge_index))
        else:
            # 简单的图聚合：平均邻居特征
            return F.relu(self.linear(x))

class SpatialGCN(nn.Module):
    """空间建模：图卷积网络"""
    def __init__(self, num_nodes, hidden_dim=64, num_layers=2):
        super(SpatialGCN, self).__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        
        layers = []
        layers.append(GCNLayer(1, hidden_dim))
        for _ in range(num_layers - 1):
            layers.append(GCNLayer(hidden_dim, hidden_dim))
        
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x, edge_index):
        """
        Args:
            x: 节点特征，形状 (batch_size, num_nodes, 1) 或 (num_nodes, 1)
            edge_index: 边索引，形状 (2, num_edges)
        
        Returns:
            h: 节点嵌入，形状 (batch_size, num_nodes, hidden_dim) 或 (num_nodes, hidden_dim)
        """
        # 处理批次维度
        if x.dim() == 3:
            batch_size, num_nodes, _ = x.shape
            # 展平批次和节点维度（使用reshape确保兼容性）
            x = x.contiguous().view(batch_size * num_nodes, -1)
            
            # 为每个批次样本创建边索引
            # PyTorch Geometric需要为批次中的每个图创建边索引
            num_edges = edge_index.shape[1]
            batch_edge_index = []
            for b in range(batch_size):
                offset = b * num_nodes
                batch_edges = edge_index + offset
                batch_edge_index.append(batch_edges)
            batch_edge_index = torch.cat(batch_edge_index, dim=1)
            
            h = x
            for layer in self.layers:
                h = layer(h, batch_edge_index)
            
            # 恢复批次维度
            h = h.view(batch_size, num_nodes, self.hidden_dim)
        else:
            h = x
            for layer in self.layers:
                h = layer(h, edge_index)
        
        return h

class TemporalLSTM(nn.Module):
    """时间建模：LSTM"""
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.1):
        super(TemporalLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
    
    def forward(self, x):
        """
        Args:
            x: 输入序列，形状 (batch_size, seq_len, input_dim)
        
        Returns:
            output: 最后一个时间步的输出，形状 (batch_size, hidden_dim)
        """
        lstm_out, (h_n, c_n) = self.lstm(x)
        # 返回最后一个时间步的隐藏状态
        return lstm_out[:, -1, :]

class SISModel(nn.Module):
    """SIS动力学模型"""
    def __init__(self, num_cities, learnable_params=True):
        super(SISModel, self).__init__()
        self.num_cities = num_cities
        self.learnable_params = learnable_params
        
        if learnable_params:
            # 可学习的参数（每个城市可以有不同的参数）
            self.beta = nn.Parameter(torch.ones(num_cities) * 0.1)  # 感染率
            self.gamma = nn.Parameter(torch.ones(num_cities) * 0.05)  # 恢复率
        else:
            # 固定参数
            self.register_buffer('beta', torch.ones(num_cities) * 0.1)
            self.register_buffer('gamma', torch.ones(num_cities) * 0.05)
    
    def forward(self, I_t, N=1.0):
        """
        SIS模型前向传播
        
        Args:
            I_t: 当前感染比例，形状 (batch_size, num_cities) 或 (num_cities,)
            N: 总人口（归一化为1）
        
        Returns:
            I_t1: 下一时间步的感染比例，形状与I_t相同
        """
        if I_t.dim() == 1:
            I_t = I_t.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        
        batch_size, num_cities = I_t.shape
        
        # 扩展参数维度
        beta = self.beta.unsqueeze(0).expand(batch_size, -1)  # (batch, num_cities)
        gamma = self.gamma.unsqueeze(0).expand(batch_size, -1)  # (batch, num_cities)
        
        # SIS动力学方程
        # I_{t+1} = I_t + beta * (N - I_t) * I_t / N - gamma * I_t
        S_t = N - I_t  # 易感者比例
        new_infections = beta * S_t * I_t / N
        recoveries = gamma * I_t
        
        I_t1 = I_t + new_infections - recoveries
        I_t1 = torch.clamp(I_t1, min=0.0, max=N)  # 确保在合理范围内
        
        if squeeze:
            I_t1 = I_t1.squeeze(0)
        
        return I_t1

class EpidemiologyGNN(nn.Module):
    """完整的流行病学知识增强的时空图神经网络"""
    def __init__(
        self,
        num_cities,
        gcn_hidden_dim=64,
        gcn_num_layers=2,
        lstm_hidden_dim=128,
        lstm_num_layers=2,
        dropout=0.1,
        use_sis=True,
        learnable_sis_params=True
    ):
        super(EpidemiologyGNN, self).__init__()
        
        self.num_cities = num_cities
        self.use_sis = use_sis
        
        # 空间建模：GCN
        self.spatial_gcn = SpatialGCN(
            num_nodes=num_cities,
            hidden_dim=gcn_hidden_dim,
            num_layers=gcn_num_layers
        )
        
        # 时间建模：LSTM
        self.temporal_lstm = TemporalLSTM(
            input_dim=gcn_hidden_dim,
            hidden_dim=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            dropout=dropout
        )
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(lstm_hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
        # SIS模型（用于正则化）
        if use_sis:
            self.sis_model = SISModel(
                num_cities=num_cities,
                learnable_params=learnable_sis_params
            )
    
    def forward(self, x, edge_index, return_sis=False):
        """
        前向传播
        
        Args:
            x: 输入序列，形状 (batch_size, window_size, num_cities)
            edge_index: 边索引，形状 (2, num_edges)
            return_sis: 是否返回SIS预测
        
        Returns:
            predictions: 预测值，形状 (batch_size, num_cities)
            sis_predictions: SIS预测（如果return_sis=True）
        """
        batch_size, window_size, num_cities = x.shape
        
        # 优化：对每个时间步进行空间建模（批量处理）
        spatial_embeddings = []
        for t in range(window_size):
            x_t = x[:, t, :].unsqueeze(-1)  # (batch, num_cities, 1)
            # 直接对整个批次处理GCN（已优化SpatialGCN支持批次处理）
            h_t = self.spatial_gcn(x_t, edge_index)  # (batch, num_cities, gcn_hidden_dim)
            spatial_embeddings.append(h_t)
        
        # 堆叠为序列 (batch, window_size, num_cities, gcn_hidden_dim)
        spatial_embeddings = torch.stack(spatial_embeddings, dim=1)
        
        # 优化：批量处理所有城市的时间建模
        # 重塑为 (batch * num_cities, window_size, gcn_hidden_dim)
        spatial_reshaped = spatial_embeddings.view(batch_size * num_cities, window_size, -1)
        # 批量处理所有城市
        temporal_all = self.temporal_lstm(spatial_reshaped)  # (batch * num_cities, lstm_hidden_dim)
        # 重塑回 (batch, num_cities, lstm_hidden_dim)
        temporal_embeddings = temporal_all.view(batch_size, num_cities, -1)
        
        # 输出层
        predictions = self.output_layer(temporal_embeddings)  # (batch, num_cities, 1)
        predictions = predictions.squeeze(-1)  # (batch, num_cities)
        
        # SIS预测（用于正则化）
        sis_predictions = None
        if self.use_sis and return_sis:
            # 使用最后一个时间步的输入作为当前状态
            I_t = x[:, -1, :]  # (batch, num_cities)
            # 将病例数转换为比例（使用滑动平均作为人口估计）
            # 使用过去窗口的平均值作为基准，避免除零
            I_t_mean = x.mean(dim=1)  # (batch, num_cities) 过去窗口的平均值
            # 归一化：使用当前值相对于历史平均值的比例
            # 添加小的平滑项避免极端值
            normalization_factor = torch.clamp(I_t_mean, min=1.0)
            I_t_normalized = I_t / normalization_factor
            # 限制在合理范围内 [0, 10]（允许短期爆发）
            I_t_normalized = torch.clamp(I_t_normalized, min=0.0, max=10.0)
            
            # SIS模型预测
            sis_predictions_normalized = self.sis_model(I_t_normalized)
            
            # 反归一化回病例数
            sis_predictions = sis_predictions_normalized * normalization_factor
        
        if return_sis:
            return predictions, sis_predictions
        else:
            return predictions

def create_fully_connected_graph(num_nodes):
    """
    创建全连接图的边索引
    
    Args:
        num_nodes: 节点数量
    
    Returns:
        edge_index: 形状 (2, num_edges) 的边索引
    """
    edges = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:  # 不包括自环
                edges.append([i, j])
    
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index

