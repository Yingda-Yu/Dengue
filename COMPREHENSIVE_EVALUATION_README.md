# 综合评估脚本使用说明

## 概述

`comprehensive_evaluation.py` 脚本用于生成论文风格的比较图表，支持按预测步长（1-4周）和按县市（county）的全面评估。

## 功能

脚本会生成以下6个图表，对应论文中的Fig.4-Fig.9：

### Fig.4: Temporal Performance（时间维度性能）
- **功能**：按预测步长（W1-W4）汇总的R²统计柱状图
- **展示内容**：MIN、MAX、AVERAGE、STD四个统计量
- **说明**：展示模型在不同预测步长下的整体表现，通常短期（W1）R²最高，随预测期延长R²下降

### Fig.5: Spatial Performance（空间维度性能）
- **功能**：按县市汇总的R²统计柱状图
- **展示内容**：每个县市的MIN、MAX、AVERAGE、STD
- **说明**：展示模型在不同县市的预测难度差异，某些县市可能更容易预测（平均R²高），某些更困难

### Fig.6: Model Comparison（模型对比）
- **功能**：对比我们的模型（EpidemiologyGNN）和baseline模型（如RF）的平均R²
- **展示内容**：每个县市两个柱子，对比两个模型的平均R²
- **说明**：直接对比两个模型的平均R²，可以看出哪个模型在哪些县市表现更好
- **注意**：如果没有提供baseline结果，会使用模拟数据用于演示

### Fig.7: Difference Heatmap（差值热力图）
- **功能**：展示RRMSE差值热力图
- **展示内容**：纵轴是县市，横轴是预测步长（W1-W4），颜色表示RRMSE差值
- **说明**：
  - 负值（蓝色）：我们的模型更好（误差更小）
  - 正值（红色）：baseline更好
  - 可以看出短期（W1）大量为负值 → 我们的模型在短期更占优

### Fig.8: Forecast Curves（预测曲线）
- **功能**：真实病例 vs 预测病例的时间序列对比
- **展示内容**：每个县市两张子图（1周预测和4周预测），显示测试集的实际时间序列
- **说明**：最直观地展示模型对真实疫情波动的跟踪能力，W1通常更贴近真实，W4可能出现滞后、过冲、峰值低估等

### Fig.9: Ablation Study（消融实验）
- **功能**：对比加入和不加入SIS正则化的预测效果
- **展示内容**：左右两列对比，每列多个县市的散点图（预测值 vs 真实值）
- **说明**：
  - 左列（有SIS正则化）：点云更贴近y=x对角线，离散更小 → 预测更稳
  - 右列（无SIS正则化）：点云更散 → 波动更大、误差更不稳定
- **注意**：如果没有提供without SIS的结果，会使用模拟数据用于演示

## 使用方法

### 基本使用

```bash
python comprehensive_evaluation.py
```

### 前置条件

1. **已训练模型**：需要 `checkpoints/best_model.pth` 文件
2. **预处理数据**：需要 `processed_data.pkl` 文件
3. **依赖库**：
   - torch
   - numpy
   - matplotlib
   - seaborn
   - sklearn
   - tqdm

### 输出文件

运行后会生成以下文件：

1. **comprehensive_evaluation_results.pkl**：详细的评估结果（包含所有预测值和指标）
2. **fig4_temporal_performance.png**：Fig.4 时间维度性能图
3. **fig5_spatial_performance.png**：Fig.5 空间维度性能图
4. **fig6_model_comparison.png**：Fig.6 模型对比图
5. **fig7_difference_heatmap.png**：Fig.7 差值热力图
6. **fig8_forecast_curves.png**：Fig.8 预测曲线图
7. **fig9_ablation_scatter.png**：Fig.9 消融实验图

## 评估指标

### R² (Coefficient of Determination)
- 衡量模型解释数据变异性的能力
- R² = 1：完美预测
- R² = 0：模型预测等于平均值
- R² < 0：模型预测比平均值还差

### RRMSE (Relative Root Mean Square Error)
- 相对均方根误差
- RRMSE = RMSE / mean(target)
- 值越小越好

## 自定义使用

### 提供Baseline结果

如果需要对比真实的baseline模型（如RF），需要：

1. 使用baseline模型在相同测试集上进行预测
2. 按照相同的格式组织结果（参考 `evaluate_by_weeks` 函数的返回格式）
3. 修改 `main()` 函数，传入baseline结果：

```python
# 加载baseline结果
with open('baseline_results.pkl', 'rb') as f:
    baseline_results = pickle.load(f)

# 生成对比图
fig6_model_comparison(results, results_baseline=baseline_results, cities=cities)
fig7_difference_heatmap(results, results_baseline=baseline_results, cities=cities)
fig8_forecast_curves(results, results_baseline=baseline_results, cities=cities)
```

### 提供消融实验结果

如果需要真实的消融实验结果（without SIS），需要：

1. 训练一个不使用SIS正则化的模型版本
2. 使用该模型在测试集上进行评估
3. 传入结果：

```python
# 加载without SIS的结果
with open('results_without_sis.pkl', 'rb') as f:
    results_without_sis = pickle.load(f)

# 生成消融实验图
fig9_ablation_scatter(results, results_without_feature=results_without_sis, cities=cities)
```

## 注意事项

1. **计算时间**：评估过程可能需要较长时间，特别是测试集较大时
2. **内存使用**：所有预测结果会保存在内存中，确保有足够内存
3. **GPU加速**：如果使用CUDA，评估会更快
4. **模拟数据**：如果没有提供baseline或消融实验结果，脚本会使用模拟数据用于演示，这些数据仅用于展示图表格式，不代表真实性能

## 结果解读

### 时间维度（Fig.4）
- 观察R²随预测步长如何变化
- 通常W1最高，W4最低
- STD反映不同县市间的波动

### 空间维度（Fig.5）
- 识别哪些县市更容易预测（平均R²高）
- 识别哪些县市更难预测（平均R²低且STD大）

### 模型对比（Fig.6, Fig.7）
- 对比两个模型的整体性能
- 识别在哪些县市、哪些预测步长下我们的模型更有优势

### 预测曲线（Fig.8）
- 观察模型对疫情峰值的捕捉能力
- 观察短期vs长期预测的差异

### 消融实验（Fig.9）
- 验证SIS正则化的有效性
- 观察加入SIS后预测稳定性的提升

## 故障排除

### 问题：R²为负值或NaN
- **原因**：预测效果很差，或者数据有问题
- **解决**：检查模型训练是否正常，数据预处理是否正确

### 问题：图表显示不完整
- **原因**：县市名称太长或数量太多
- **解决**：可以修改代码，只选择部分县市进行可视化

### 问题：内存不足
- **原因**：测试集太大，所有预测结果保存在内存中
- **解决**：可以分批处理，或者只评估部分测试样本

