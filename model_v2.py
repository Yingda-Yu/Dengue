#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
改进版：Epidemiology-Informed Spatio-Temporal Graph Neural Network
优化点：
1. 添加时间特征编码（周期性、趋势）
2. 多尺度时间建模（CNN + LSTM + Attention）
3. 残差连接和LayerNorm
4. 改进的空间建模（GAT替代GCN）
5. 输出非负约束
6. 更好的特征工程
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

try:
    from torch_geometric.nn import GATConv, GCNConv
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    print("警告: torch-geometric未安装，将使用简化实现")


class PositionalEncoding(nn.Module):
    """位置编码，捕捉时间序列中的位置信息"""
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[:d_model//2])
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pe[:x.size(1), :].unsqueeze(0)


class TemporalConvBlock(nn.Module):
    """时间卷积块，捕捉局部时间模式"""
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                             padding=padding, dilation=dilation)
        self.norm = nn.LayerNorm(out_channels)
        self.activation = nn.GELU()
        
    def forward(self, x):
        # x: (batch, seq_len, channels)
        x = x.transpose(1, 2)  # (batch, channels, seq_len)
        x = self.conv(x)
        x = x.transpose(1, 2)  # (batch, seq_len, channels)
        x = self.norm(x)
        x = self.activation(x)
        return x


class MultiScaleTemporalEncoder(nn.Module):
    """多尺度时间编码器"""
    def __init__(self, input_dim, hidden_dim=64, num_scales=3):
        super().__init__()
        self.num_scales = num_scales
        
        # 不同膨胀率的时间卷积
        self.conv_layers = nn.ModuleList([
            TemporalConvBlock(input_dim if i == 0 else hidden_dim, 
                            hidden_dim, kernel_size=3, dilation=2**i)
            for i in range(num_scales)
        ])
        
        # 融合层
        self.fusion = nn.Linear(hidden_dim * num_scales, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        outputs = []
        h = x
        for conv in self.conv_layers:
            h = conv(h)
            outputs.append(h)
        
        # 拼接不同尺度的输出
        multi_scale = torch.cat(outputs, dim=-1)
        out = self.fusion(multi_scale)
        out = self.norm(out)
        return out


class SpatialAttention(nn.Module):
    """空间注意力模块（简化版GAT）"""
    def __init__(self, in_dim, out_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        
        self.query = nn.Linear(in_dim, out_dim)
        self.key = nn.Linear(in_dim, out_dim)
        self.value = nn.Linear(in_dim, out_dim)
        self.out_proj = nn.Linear(out_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        
    def forward(self, x, adj_mask=None):
        # x: (batch, num_nodes, in_dim)
        batch_size, num_nodes, _ = x.shape
        
        Q = self.query(x).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        K = self.key(x).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        V = self.value(x).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        
        # (batch, heads, nodes, head_dim)
        Q = Q.permute(0, 2, 1, 3)
        K = K.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)
        
        # 注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 应用邻接掩码（如果有）
        if adj_mask is not None:
            scores = scores.masked_fill(adj_mask == 0, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        attn = F.dropout(attn, p=0.1, training=self.training)
        
        out = torch.matmul(attn, V)
        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, num_nodes, -1)
        out = self.out_proj(out)
        out = self.norm(out + x[:, :, :out.size(-1)] if x.size(-1) == out.size(-1) else out)
        
        return out


class ImprovedSpatialGCN(nn.Module):
    """改进的空间图卷积模块"""
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.1):
        super().__init__()
        
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        
        # 使用多层空间注意力
        self.layers = nn.ModuleList([
            SpatialAttention(hidden_dim, hidden_dim, num_heads=4)
            for _ in range(num_layers)
        ])
        
        self.output_proj = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_dim)
        
    def forward(self, x, edge_index=None):
        # x: (batch, num_nodes, in_dim)
        h = self.input_proj(x)
        h = F.gelu(h)
        
        for layer in self.layers:
            h = layer(h)
            h = self.dropout(h)
        
        out = self.output_proj(h)
        out = self.norm(out)
        return out


class TemporalAttention(nn.Module):
    """时间注意力，关注重要的历史时间步"""
    def __init__(self, hidden_dim, num_heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        # x: (batch, seq_len, hidden_dim)
        attn_out, _ = self.attention(x, x, x)
        out = self.norm(x + attn_out)
        return out


class ImprovedTemporalEncoder(nn.Module):
    """改进的时间编码器：CNN + LSTM + Attention"""
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.1):
        super().__init__()
        
        # 多尺度卷积
        self.multi_scale_conv = MultiScaleTemporalEncoder(input_dim, hidden_dim // 2, num_scales=3)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(hidden_dim // 2)
        
        # LSTM
        self.lstm = nn.LSTM(
            hidden_dim // 2, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # 时间注意力
        self.temporal_attn = TemporalAttention(hidden_dim, num_heads=4)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        
        # 多尺度卷积
        h = self.multi_scale_conv(x)
        
        # 位置编码
        h = self.pos_encoding(h)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(h)
        
        # 时间注意力
        attn_out = self.temporal_attn(lstm_out)
        
        # 返回最后时间步
        out = attn_out[:, -1, :]
        out = self.dropout(out)
        
        return out


class SISRegularizer(nn.Module):
    """改进的SIS正则化模块"""
    def __init__(self, num_cities, learnable=True):
        super().__init__()
        self.num_cities = num_cities
        
        if learnable:
            # 使用更合理的初始值和约束
            self.log_beta = nn.Parameter(torch.zeros(num_cities))  # log空间确保正值
            self.log_gamma = nn.Parameter(torch.zeros(num_cities) - 1)  # gamma通常较小
        else:
            self.register_buffer('log_beta', torch.zeros(num_cities))
            self.register_buffer('log_gamma', torch.zeros(num_cities) - 1)
    
    @property
    def beta(self):
        return torch.sigmoid(self.log_beta) * 0.5  # 限制在[0, 0.5]
    
    @property
    def gamma(self):
        return torch.sigmoid(self.log_gamma) * 0.3  # 限制在[0, 0.3]
    
    def forward(self, I_t, population_scale=100.0):
        """
        SIS动力学预测
        I_t: (batch, num_cities) 当前感染数
        """
        # 归一化到[0, 1]范围
        I_normalized = I_t / population_scale
        I_normalized = torch.clamp(I_normalized, 0, 1)
        
        S_normalized = 1 - I_normalized
        
        # SIS方程
        new_infections = self.beta * S_normalized * I_normalized
        recoveries = self.gamma * I_normalized
        
        I_next_normalized = I_normalized + new_infections - recoveries
        I_next_normalized = torch.clamp(I_next_normalized, 0, 1)
        
        # 反归一化
        I_next = I_next_normalized * population_scale
        
        return I_next


class EpidemiologyGNNv2(nn.Module):
    """
    改进版流行病学知识增强的时空图神经网络
    
    改进点：
    1. 多尺度时间建模
    2. 空间注意力机制
    3. 残差连接
    4. 更好的特征工程
    5. 输出非负约束
    """
    def __init__(
        self,
        num_cities,
        window_size=14,
        spatial_hidden_dim=64,
        temporal_hidden_dim=128,
        num_spatial_layers=2,
        num_temporal_layers=2,
        dropout=0.2,
        use_sis=True
    ):
        super().__init__()
        
        self.num_cities = num_cities
        self.window_size = window_size
        self.use_sis = use_sis
        
        # 输入特征工程：将单一病例数扩展为多维特征
        self.feature_dim = 8  # 扩展的特征维度
        self.feature_encoder = nn.Sequential(
            nn.Linear(1, 32),
            nn.GELU(),
            nn.Linear(32, self.feature_dim),
            nn.LayerNorm(self.feature_dim)
        )
        
        # 空间编码器
        self.spatial_encoder = ImprovedSpatialGCN(
            in_dim=self.feature_dim,
            hidden_dim=spatial_hidden_dim,
            out_dim=spatial_hidden_dim,
            num_layers=num_spatial_layers,
            dropout=dropout
        )
        
        # 时间编码器（每个城市共享）
        self.temporal_encoder = ImprovedTemporalEncoder(
            input_dim=spatial_hidden_dim,
            hidden_dim=temporal_hidden_dim,
            num_layers=num_temporal_layers,
            dropout=dropout
        )
        
        # 输出头
        self.output_head = nn.Sequential(
            nn.Linear(temporal_hidden_dim, temporal_hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(temporal_hidden_dim // 2, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )
        
        # 残差连接：从输入直接连到输出
        self.residual_weight = nn.Parameter(torch.tensor(0.1))
        
        # SIS正则化
        if use_sis:
            self.sis_model = SISRegularizer(num_cities, learnable=True)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    
    def forward(self, x, edge_index=None, return_sis=False):
        """
        前向传播
        
        Args:
            x: (batch, window_size, num_cities) 输入序列
            edge_index: 图边索引（可选，当前使用全连接注意力）
            return_sis: 是否返回SIS预测
        
        Returns:
            predictions: (batch, num_cities) 预测值
            sis_predictions: SIS模型预测（如果return_sis=True）
        """
        batch_size, window_size, num_cities = x.shape
        
        # 1. 特征工程：扩展输入特征
        # (batch, window_size, num_cities) -> (batch, window_size, num_cities, feature_dim)
        x_expanded = x.unsqueeze(-1)  # (batch, window_size, num_cities, 1)
        x_features = self.feature_encoder(x_expanded)  # (batch, window_size, num_cities, feature_dim)
        
        # 2. 时空建模
        spatial_temporal_features = []
        for t in range(window_size):
            # 空间建模：(batch, num_cities, feature_dim)
            spatial_out = self.spatial_encoder(x_features[:, t, :, :])
            spatial_temporal_features.append(spatial_out)
        
        # (batch, window_size, num_cities, spatial_hidden_dim)
        spatial_temporal_features = torch.stack(spatial_temporal_features, dim=1)
        
        # 3. 时间建模（每个城市）
        # 重塑为 (batch * num_cities, window_size, spatial_hidden_dim)
        temporal_input = spatial_temporal_features.permute(0, 2, 1, 3)  # (batch, num_cities, window_size, dim)
        temporal_input = temporal_input.reshape(batch_size * num_cities, window_size, -1)
        
        temporal_out = self.temporal_encoder(temporal_input)  # (batch * num_cities, temporal_hidden_dim)
        temporal_out = temporal_out.view(batch_size, num_cities, -1)  # (batch, num_cities, temporal_hidden_dim)
        
        # 4. 输出预测
        predictions = self.output_head(temporal_out).squeeze(-1)  # (batch, num_cities)
        
        # 5. 残差连接：加上最后一个时间步的输入
        last_input = x[:, -1, :]  # (batch, num_cities)
        predictions = predictions + self.residual_weight * last_input
        
        # 6. 确保非负输出
        predictions = F.softplus(predictions)  # 使用softplus确保正值且平滑
        
        # 7. SIS预测
        sis_predictions = None
        if self.use_sis and return_sis:
            sis_predictions = self.sis_model(last_input)
        
        if return_sis:
            return predictions, sis_predictions
        return predictions


def create_fully_connected_graph(num_nodes):
    """创建全连接图的边索引"""
    edges = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                edges.append([i, j])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index


# 测试代码
if __name__ == "__main__":
    # 测试模型
    batch_size = 32
    window_size = 14
    num_cities = 22
    
    model = EpidemiologyGNNv2(
        num_cities=num_cities,
        window_size=window_size,
        spatial_hidden_dim=64,
        temporal_hidden_dim=128,
        num_spatial_layers=2,
        num_temporal_layers=2,
        dropout=0.2,
        use_sis=True
    )
    
    # 测试输入
    x = torch.randn(batch_size, window_size, num_cities).abs()  # 确保非负
    edge_index = create_fully_connected_graph(num_cities)
    
    # 前向传播
    predictions, sis_pred = model(x, edge_index, return_sis=True)
    
    print(f"输入形状: {x.shape}")
    print(f"预测形状: {predictions.shape}")
    print(f"SIS预测形状: {sis_pred.shape}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"预测值范围: [{predictions.min().item():.4f}, {predictions.max().item():.4f}]")
