#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
可视化模型结构
"""
import torch
import pickle
from model import EpidemiologyGNN, create_fully_connected_graph
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def print_model_summary(model, input_shape):
    """打印模型摘要"""
    print("=" * 80)
    print("模型结构详细摘要")
    print("=" * 80)
    
    total_params = 0
    trainable_params = 0
    
    print(f"\n输入形状: {input_shape}")
    print(f"\n{'层名称':<30} {'输出形状':<25} {'参数数量':<15}")
    print("-" * 80)
    
    # 遍历所有模块
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        total_params += params
        trainable_params += trainable
        
        print(f"{name:<30} {'-':<25} {params:,}")
        
        # 如果是Sequential或ModuleList，打印子模块
        if isinstance(module, (torch.nn.Sequential, torch.nn.ModuleList)):
            for sub_name, sub_module in module.named_children():
                sub_params = sum(p.numel() for p in sub_module.parameters())
                print(f"  └─ {sub_name:<28} {'-':<25} {sub_params:,}")
    
    print("-" * 80)
    print(f"{'总计':<30} {'-':<25} {total_params:,}")
    print(f"\n总参数数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print(f"模型大小: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")
    print("=" * 80)

def create_model_architecture_diagram(model, save_path='model_architecture.png'):
    """创建模型架构图"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # 定义颜色
    colors = {
        'input': '#E8F4F8',
        'gcn': '#FFE5B4',
        'lstm': '#D4EDDA',
        'output': '#F8D7DA',
        'sis': '#E2E3E5'
    }
    
    y_positions = [10, 7.5, 5, 2.5, 0.5]
    
    # 输入层
    input_box = FancyBboxPatch((0.5, y_positions[0]-0.5), 2, 1, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['input'], 
                               edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(1.5, y_positions[0], 'Input\n(batch, 14, 22)', 
            ha='center', va='center', fontsize=10, weight='bold')
    
    # GCN层
    gcn_box = FancyBboxPatch((3.5, y_positions[1]-0.5), 2.5, 1, 
                             boxstyle="round,pad=0.1", 
                             facecolor=colors['gcn'], 
                             edgecolor='black', linewidth=2)
    ax.add_patch(gcn_box)
    ax.text(5.25, y_positions[1], 'Spatial GCN\n2 layers, hidden=64', 
            ha='center', va='center', fontsize=10, weight='bold')
    
    # LSTM层
    lstm_box = FancyBboxPatch((3.5, y_positions[2]-0.5), 2.5, 1, 
                              boxstyle="round,pad=0.1", 
                              facecolor=colors['lstm'], 
                              edgecolor='black', linewidth=2)
    ax.add_patch(lstm_box)
    ax.text(5.25, y_positions[2], 'Temporal LSTM\n2 layers, hidden=128', 
            ha='center', va='center', fontsize=10, weight='bold')
    
    # 输出层
    output_box = FancyBboxPatch((3.5, y_positions[3]-0.5), 2.5, 1, 
                                boxstyle="round,pad=0.1", 
                                facecolor=colors['output'], 
                                edgecolor='black', linewidth=2)
    ax.add_patch(output_box)
    ax.text(5.25, y_positions[3], 'Output Layer\nLinear(128->64->1)', 
            ha='center', va='center', fontsize=10, weight='bold')
    
    # SIS模型
    sis_box = FancyBboxPatch((7, y_positions[2]-0.5), 2, 1, 
                             boxstyle="round,pad=0.1", 
                             facecolor=colors['sis'], 
                             edgecolor='black', linewidth=2)
    ax.add_patch(sis_box)
    ax.text(8, y_positions[2], 'SIS Model\nRegularization', 
            ha='center', va='center', fontsize=10, weight='bold')
    
    # 输出
    final_box = FancyBboxPatch((3.5, y_positions[4]-0.5), 2.5, 1, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['output'], 
                               edgecolor='black', linewidth=2)
    ax.add_patch(final_box)
    ax.text(5.25, y_positions[4], 'Prediction\n(batch, 22)', 
            ha='center', va='center', fontsize=10, weight='bold')
    
    # 箭头
    arrows = [
        ((2.5, y_positions[0]), (3.5, y_positions[1])),  # 输入 -> GCN
        ((5.25, y_positions[1]-0.5), (5.25, y_positions[2]+0.5)),  # GCN -> LSTM
        ((5.25, y_positions[2]-0.5), (5.25, y_positions[3]+0.5)),  # LSTM -> 输出层
        ((5.25, y_positions[3]-0.5), (5.25, y_positions[4]+0.5)),  # 输出层 -> 最终输出
        ((5.25, y_positions[2]), (7, y_positions[2])),  # LSTM -> SIS
        ((8, y_positions[2]), (5.25, y_positions[3])),  # SIS -> 输出层
    ]
    
    for start, end in arrows:
        arrow = FancyArrowPatch(start, end, 
                               arrowstyle='->', 
                               mutation_scale=20, 
                               linewidth=2, 
                               color='black',
                               zorder=1)
        ax.add_patch(arrow)
    
    # 标题
    ax.text(5, 11.5, 'Epidemiology-Informed Spatio-Temporal GNN 架构', 
            ha='center', va='center', fontsize=16, weight='bold')
    
    # 图例
    legend_elements = [
        mpatches.Patch(facecolor=colors['input'], label='Input Layer', edgecolor='black'),
        mpatches.Patch(facecolor=colors['gcn'], label='Spatial GCN', edgecolor='black'),
        mpatches.Patch(facecolor=colors['lstm'], label='Temporal LSTM', edgecolor='black'),
        mpatches.Patch(facecolor=colors['output'], label='Output Layer', edgecolor='black'),
        mpatches.Patch(facecolor=colors['sis'], label='SIS Regularization', edgecolor='black'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"模型架构图已保存到 {save_path}")
    plt.close()

def create_detailed_model_info(model, save_path='model_info.txt'):
    """创建详细的模型信息文本文件"""
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Epidemiology-Informed Spatio-Temporal GNN 模型详细信息\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("模型架构:\n")
        f.write("-" * 80 + "\n")
        f.write(str(model))
        f.write("\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("各模块详细信息\n")
        f.write("=" * 80 + "\n\n")
        
        for name, module in model.named_children():
            f.write(f"模块: {name}\n")
            f.write(f"类型: {type(module).__name__}\n")
            params = sum(p.numel() for p in module.parameters())
            f.write(f"参数数量: {params:,}\n")
            f.write(f"结构:\n{module}\n")
            f.write("-" * 80 + "\n\n")
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        f.write("=" * 80 + "\n")
        f.write("参数统计\n")
        f.write("=" * 80 + "\n")
        f.write(f"总参数数量: {total_params:,}\n")
        f.write(f"可训练参数: {trainable_params:,}\n")
        f.write(f"模型大小: {total_params * 4 / 1024 / 1024:.2f} MB (float32)\n")
        f.write("=" * 80 + "\n")
    
    print(f"模型详细信息已保存到 {save_path}")

def main():
    # 加载数据
    data = pickle.load(open('processed_data.pkl', 'rb'))
    num_cities = len(data['cities'])
    
    # 创建模型
    model = EpidemiologyGNN(
        num_cities=num_cities,
        gcn_hidden_dim=64,
        gcn_num_layers=2,
        lstm_hidden_dim=128,
        lstm_num_layers=2,
        dropout=0.1,
        use_sis=True,
        learnable_sis_params=True
    )
    
    # 打印模型摘要
    print_model_summary(model, (128, 14, 22))
    
    # 创建架构图
    print("\n正在生成模型架构图...")
    create_model_architecture_diagram(model)
    
    # 创建详细信息文件
    print("\n正在生成模型详细信息...")
    create_detailed_model_info(model)
    
    print("\n模型可视化完成！")

if __name__ == "__main__":
    main()

