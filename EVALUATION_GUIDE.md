# 综合评估图表生成指南

本指南说明如何使用 `comprehensive_evaluation.py` 生成论文风格的比较图表。

## 概述

`comprehensive_evaluation.py` 可以生成以下6种图表，对应论文中的 Fig.4-Fig.9：

1. **Fig.4: Temporal Performance** - 按预测步长（1-4周）的R²统计柱状图
2. **Fig.5: Spatial Performance** - 按县市的R²统计柱状图
3. **Fig.6: Model Comparison** - 模型对比柱状图（我们的模型 vs Baseline）
4. **Fig.7: Difference Heatmap** - RRMSE差值热力图
5. **Fig.8: Forecast Curves** - 真实vs预测时间序列对比
6. **Fig.9: Ablation Study** - 消融实验散点图

## 使用方法

### 基本使用（仅使用我们的模型）

```bash
python comprehensive_evaluation.py
```

这将：
1. 加载训练好的模型（`checkpoints/best_model.pth`）
2. 在测试集上评估1-4周的预测性能
3. 计算R²和RRMSE指标
4. 生成所有图表（对于需要baseline的图表，会使用模拟数据）

### 输出文件

- `comprehensive_evaluation_results.pkl` - 详细的评估结果（包含所有预测值和指标）
- `fig4_temporal_performance.png` - Fig.4
- `fig5_spatial_performance.png` - Fig.5
- `fig6_model_comparison.png` - Fig.6
- `fig7_difference_heatmap.png` - Fig.7
- `fig8_forecast_curves.png` - Fig.8
- `fig9_ablation_scatter.png` - Fig.9

## 高级使用

### 添加Baseline模型结果

要生成真实的模型对比图（Fig.6, Fig.7），需要提供baseline模型的结果。修改代码：

```python
# 在main()函数中，加载baseline模型并评估
baseline_results = evaluate_by_weeks(baseline_model, test_X, test_y, edge_index, device, cities)

# 然后传入图表生成函数
fig6_model_comparison(results, results_baseline=baseline_results, cities=cities)
fig7_difference_heatmap(results, results_baseline=baseline_results, cities=cities)
```

### 添加消融实验结果

要生成真实的消融实验图（Fig.9），需要训练一个不使用SIS正则化的模型：

```python
# 训练without SIS的模型
model_without_sis = EpidemiologyGNN(..., use_sis=False)
# ... 训练模型 ...

# 评估
results_without_sis = evaluate_by_weeks(model_without_sis, test_X, test_y, edge_index, device, cities)

# 生成图表
fig9_ablation_scatter(results, results_without_feature=results_without_sis, cities=cities)
```

## 指标说明

### R² (Coefficient of Determination)

R²衡量模型解释数据变异性的能力：
- R² = 1：完美预测
- R² = 0：模型预测等于平均值
- R² < 0：模型预测比平均值还差

### RRMSE (Relative Root Mean Square Error)

RRMSE = RMSE / mean(真实值)
- 值越小越好
- 无量纲，便于跨不同量级的数据比较

## 图表说明

### Fig.4: Temporal Performance
展示模型在不同预测步长下的整体表现。通常短期（W1）R²最高，随预测期延长R²下降。

### Fig.5: Spatial Performance
展示模型在不同县市的预测难度差异。某些县市可能更容易预测（平均R²高），某些更困难。

### Fig.6: Model Comparison
直接对比两个模型的平均R²。可以看出哪个模型在哪些县市表现更好。

### Fig.7: Difference Heatmap
热力图展示RRMSE差值：
- 蓝色（负值）：我们的模型更好
- 红色（正值）：Baseline更好

### Fig.8: Forecast Curves
最直观地展示模型对真实疫情波动的跟踪能力。每个县市两张子图：
- 上图：1周预测
- 下图：4周预测

### Fig.9: Ablation Study
散点图对比加入/不加入特定特征的效果：
- 左列：有特征
- 右列：无特征
- 点云越贴近对角线，预测越准确

## 注意事项

1. **数据要求**：需要先运行 `data_preprocessing.py` 和 `train.py` 生成模型和数据
2. **计算时间**：评估1-4周预测可能需要较长时间，特别是测试集较大时
3. **内存需求**：所有预测结果会保存在内存中，确保有足够内存
4. **模拟数据**：如果没有baseline或消融实验结果，代码会使用模拟数据用于演示

## 自定义

### 修改预测步长

在 `evaluate_by_weeks()` 函数中修改：

```python
weeks = [1, 2, 3, 4]  # 可以改为其他周数
week_days = [7, 14, 21, 28]  # 对应的天数
```

### 修改图表样式

所有图表函数都接受 `save_path` 参数，可以自定义保存路径。图表样式（颜色、字体等）可以在函数内部修改。

## 故障排除

### 问题：R²为负值或NaN
- 检查预测值是否合理（非负）
- 检查是否有足够的有效数据点
- 某些县市可能数据量太少

### 问题：图表太挤
- 减少显示的县市数量（修改 `num_cities_to_plot`）
- 调整图表大小（修改 `figsize`）

### 问题：内存不足
- 减少测试集大小
- 分批处理数据

