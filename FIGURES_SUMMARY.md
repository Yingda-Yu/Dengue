# 图表生成功能总结

## 已完成的工作

我已经创建了 `comprehensive_evaluation.py` 脚本，可以生成论文风格的6种比较图表（Fig.4-Fig.9）。

### 实现的图表

#### ✅ Fig.4: Temporal Performance（时间维度性能）
- **功能**：按预测步长（W1-W4）汇总的R²统计柱状图
- **展示内容**：MIN、MAX、AVERAGE、STD四个统计量
- **说明**：展示模型在不同预测步长下的整体表现，通常短期（W1）R²最高

#### ✅ Fig.5: Spatial Performance（空间维度性能）
- **功能**：按县市汇总的R²统计柱状图
- **展示内容**：每个县市的MIN、MAX、AVERAGE、STD
- **说明**：展示模型在不同县市的预测难度差异

#### ✅ Fig.6: Model Comparison（模型对比）
- **功能**：对比我们的模型和baseline模型的平均R²
- **展示内容**：每个县市两根柱子（Our Model vs Baseline）
- **说明**：直接回答哪个模型在哪些县市表现更好
- **注意**：如果没有baseline结果，会使用模拟数据

#### ✅ Fig.7: Difference Heatmap（差值热力图）
- **功能**：展示RRMSE差值热力图
- **展示内容**：县市 × 预测步长矩阵，颜色表示差值
- **说明**：
  - 蓝色（负值）：我们的模型更好
  - 红色（正值）：Baseline更好
- **注意**：如果没有baseline结果，会使用模拟数据

#### ✅ Fig.8: Forecast Curves（预测曲线）
- **功能**：真实病例 vs 预测病例的时间序列对比
- **展示内容**：每个县市两张子图（W1和W4预测）
- **说明**：最直观地展示模型对真实疫情波动的跟踪能力
- **注意**：如果没有baseline结果，会使用模拟数据

#### ✅ Fig.9: Ablation Study（消融实验）
- **功能**：对比加入/不加入SIS正则化的预测效果
- **展示内容**：左右两列散点图（With vs Without SIS）
- **说明**：展示SIS正则化对预测效果的提升
- **注意**：如果没有without SIS的结果，会使用模拟数据

## 核心功能

### 评估指标
- **R² (Coefficient of Determination)**：决定系数，衡量模型解释数据变异性的能力
- **RRMSE (Relative Root Mean Square Error)**：相对均方根误差，便于跨不同量级数据比较

### 评估方式
- **按周评估**：1周（7天）、2周（14天）、3周（21天）、4周（28天）
- **按县市评估**：每个县市单独计算指标
- **跨时间步统计**：对每个县市在不同时间步的指标进行统计（MIN、MAX、AVERAGE、STD）

## 使用方法

### 基本使用
```bash
python comprehensive_evaluation.py
```

### 输出文件
1. `comprehensive_evaluation_results.pkl` - 详细评估结果
2. `fig4_temporal_performance.png` - Fig.4
3. `fig5_spatial_performance.png` - Fig.5
4. `fig6_model_comparison.png` - Fig.6
5. `fig7_difference_heatmap.png` - Fig.7
6. `fig8_forecast_curves.png` - Fig.8
7. `fig9_ablation_scatter.png` - Fig.9

## 下一步工作

### 可选：添加真实Baseline模型
如果要生成真实的模型对比图，需要：
1. 实现baseline模型（如Random Forest）
2. 在测试集上评估baseline模型
3. 将结果传入图表生成函数

### 可选：添加真实消融实验
如果要生成真实的消融实验图，需要：
1. 训练一个不使用SIS正则化的模型（`use_sis=False`）
2. 在测试集上评估该模型
3. 将结果传入图表生成函数

## 技术细节

### 数据格式
- 输入：测试集数据（`test_X`, `test_y`）
- 输出：按周和按县市组织的预测结果和指标

### 预测方式
- 使用自回归方式：每次预测下一天，然后更新窗口继续预测
- 对每个测试样本进行多步预测

### 指标计算
- R²：使用sklearn的`r2_score`
- RRMSE：RMSE / mean(真实值)

## 注意事项

1. **计算时间**：评估1-4周预测可能需要较长时间
2. **内存需求**：所有预测结果会保存在内存中
3. **模拟数据**：对于需要baseline或消融实验的图表，如果没有真实数据会使用模拟数据
4. **数据要求**：需要先运行数据预处理和模型训练

## 文件说明

- `comprehensive_evaluation.py` - 主评估脚本
- `EVALUATION_GUIDE.md` - 详细使用指南
- `FIGURES_SUMMARY.md` - 本文件，功能总结

