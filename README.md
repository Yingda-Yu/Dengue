# Epidemiology-Informed Spatio-Temporal Graph Neural Network for Dengue Prediction

基于流行病学先验的时空图神经网络，用于台湾各县市登革热病例预测。支持**软约束**与**硬约束**两种训练方式，并可对比两者效果。命令行用法与最终结果说明见下文「命令行使用说明」与「最终结果说明」。

## Model Architecture

### Core Components

1. **Spatial Modeling**: Graph-based spatial attention mechanism to capture inter-city spatial correlations
2. **Temporal Modeling**: Multi-scale temporal encoder with positional encoding and LSTM for temporal pattern learning
3. **SIS Regularization**: SIS (Susceptible-Infected-Susceptible) dynamics model as a soft constraint

### Key Features

- **Hybrid Modeling**: Combines data-driven learning with epidemiological priors
- **Multi-Scale Temporal Processing**: Captures patterns at different time scales (3, 7, 14, 30 days)
- **Attention Mechanisms**: Both spatial and temporal attention for improved feature extraction
- **Soft Constraints**: Epidemiological dynamics guide learning without hard constraints
- **Interpretability**: Explicit disease dynamics modeling
- **Extensibility**: Supports learnable SIS parameters, city-specific parameters, and mobility information

## Project Structure

```
.
├── Dengue_Daily_EN.csv          # 原始数据
├── data_preprocessing.py        # 数据预处理
├── model.py                     # 模型定义 (EpidemiologyGNNv2)
├── train.py                     # 训练（--constraint soft/hard）
├── inference.py                 # 单步/多步预测
├── visualize_results.py         # 可视化（--model soft/hard/both）
├── compare_models.py            # 软 vs 硬 对比表格与图
├── run_pipeline.py              # 一键流程
└── README.md
```

## Quick Start

**推荐完整流程（得到软、硬两个模型及对比）：**  
1) `python data_preprocessing.py`  
2) `python train.py --constraint soft` → `python train.py --constraint hard`  
3) `python visualize_results.py --model both`（生成每城市 真实值+软+硬 图）  
4) `python compare_models.py`（生成对比表与图）

### Option 1: One-Click Pipeline

```bash
python run_pipeline.py
```

自动执行：数据预处理 → 训练软约束模型 → 可视化软约束结果（若测试结果存在）。

### Option 2: Step-by-Step Execution

#### Step 1: Data Preprocessing

```bash
python data_preprocessing.py
```

- 读取 `Dengue_Daily_EN.csv`，按通报日期与县市聚合
- 构造 14 天滑动窗口序列，按比例划分训练/验证/测试集（7 : 1.5 : 1.5）
- 保存为 `processed_data.pkl`

#### Step 2: Train Model (Soft or Hard Constraint)

```bash
# 软约束（SIS 仅作损失项）
python train.py --constraint soft

# 硬约束（输出 = SIS + NN 残差）
python train.py --constraint hard
```

- Saves: `checkpoints/best_model_soft.pth` / `best_model_hard.pth`, and `best_mae_model_soft.pth` / `best_mae_model_hard.pth`.
- Training uses the same pipeline (AMP, early stopping, etc.); hard constraint sets λ_SIS=0 in the loss.

#### Step 3: Visualize Results

```bash
python visualize_results.py --model soft   # 仅软约束
python visualize_results.py --model hard   # 仅硬约束
python visualize_results.py --model both   # 每张城市图含：真实值 + 软约束 + 硬约束（推荐）
```

- `soft`/`hard`：单模型，生成 MAE/RMSE 柱状图、时间序列、多 horizon 对比。
- `both`：双模型，`city_predictions/` 下每张图为真实值 + 软 + 硬（fig2 为 7 天预测，fig3 为 2×2 各 horizon）。

#### Step 4: Run Inference

```bash
# 使用软约束模型预测 7 天
python inference.py --model soft --days 7 --use_test_data

# 使用硬约束模型
python inference.py --model hard --days 7 --use_test_data

# 或指定 checkpoint 路径
python inference.py --checkpoint checkpoints/best_model_soft.pth --days 7 --use_test_data
```

#### Step 5: Compare Soft vs Hard Constraint

After training both models:

```bash
python compare_models.py
```

This produces under `visualization_results/soft_vs_hard_comparison/`:
- `comparison_table.csv`: MAE/RMSE per city and horizon, with which model is better.
- `comparison_summary_by_horizon.csv`: Average MAE/RMSE by forecast horizon.
- `fig_comparison_mae_rmse_by_city.png`: Bar charts of MAE and RMSE by city.
- `fig_comparison_by_horizon.png`: MAE/RMSE by horizon.
- `fig_comparison_timeseries_sample.png`: Example time series (Actual vs Soft vs Hard).

---

## 命令行使用说明（详细）

| 脚本 | 命令 | 说明 |
|------|------|------|
| 预处理 | `python data_preprocessing.py` | 无参数。生成 `processed_data.pkl`。 |
| 训练 | `python train.py --constraint soft` | 训练软约束，保存 `best_model_soft.pth` 等。 |
| 训练 | `python train.py --constraint hard` | 训练硬约束，保存 `best_model_hard.pth` 等。 |
| 可视化 | `python visualize_results.py --model soft` | 使用软约束权重生成图表。 |
| 可视化 | `python visualize_results.py --model hard` | 使用硬约束权重生成图表。 |
| 可视化 | `python visualize_results.py --model both` | 双模型：city_predictions 每张图为真实值+软+硬。 |
| 推理 | `python inference.py --model soft --days 7 --use_test_data` | 软约束预测 7 天；`--model` 可换为 `hard`。 |
| 推理 | `python inference.py --checkpoint <path> --days N` | 指定 checkpoint 预测 N 天。 |
| 对比 | `python compare_models.py` | 无参数。需先有 soft 与 hard 两个权重。 |
| 一键 | `python run_pipeline.py` | 预处理 → 训练软约束 → 可视化软约束。 |

**推理参数**：`--checkpoint` 可选，指定则覆盖 `--model`；`--days` 默认 7；`--use_test_data` 表示用测试集最后一窗口并输出 MAE/RMSE。

---

## 最终结果说明

### 检查点与训练结果（`checkpoints/`）

| 文件 | 说明 |
|------|------|
| `best_model_soft.pth` / `best_model_hard.pth` | 验证 loss 最优权重，含 config、scaler。 |
| `best_mae_model_soft.pth` / `best_mae_model_hard.pth` | 验证 MAE 最优权重。 |
| `test_results_soft.pkl` / `test_results_hard.pkl` | 测试集预测、真实值、指标。 |
| `training_curves_soft.png` / `training_curves_hard.png` | 训练/验证 loss、MAE、RMSE、学习率曲线。 |

### 可视化（`visualization_results/`）

- **fig1_***：MAE、RMSE 按城市与 horizon（3/7/14/30 天）的柱状图；`mae_by_city_horizon.csv`、`rmse_by_city_horizon.csv` 为对应数值表。
- **fig2_city_timeseries_summary.png**：若干城市时间序列汇总。
- **city_predictions/**  
  - 单模型（`--model soft` 或 `hard`）：`fig2_*_timeseries.png` 为每城市真实值 + 多 horizon 预测；`fig3_*_horizon_comparison.png` 为每城市 2×2 各 horizon 对比。  
  - 双模型（`--model both`）：`fig2_*_timeseries.png` 为每城市**真实值 + 软约束 + 硬约束**（7 天预测）；`fig3_*_horizon_comparison.png` 为每城市 2×2，每子图**Actual、Soft、Hard** 三条线及对应 MAE。

### 软 vs 硬对比（`visualization_results/soft_vs_hard_comparison/`）

| 文件 | 说明 |
|------|------|
| `comparison_table.csv` | 按城市、按 horizon 的 MAE/RMSE（软、硬）及谁更优。 |
| `comparison_summary_by_horizon.csv` | 按 horizon 的平均 MAE、RMSE 及 Better。 |
| `fig_comparison_mae_rmse_by_city.png` | 按城市 MAE、RMSE 柱状对比。 |
| `fig_comparison_by_horizon.png` | 按预测天数平均 MAE、RMSE 对比。 |
| `fig_comparison_timeseries_sample.png` | 示例城市时间序列：真实值 vs 软 vs 硬。 |

## Model Configuration

You can modify the following configurations in `train.py`:

```python
config = {
    'batch_size': 512,
    'learning_rate': 5e-4,
    'min_lr': 1e-6,
    'weight_decay': 1e-5,
    'num_epochs': 300,
    'patience': 50,              # Early stopping patience
    'grad_clip': 1.0,
    
    # Loss weights
    'lambda_mae': 0.5,
    'lambda_outbreak': 0.3,
    'lambda_sis': 0.05,
    
    # Model architecture
    'spatial_hidden_dim': 128,
    'temporal_hidden_dim': 256,
    'num_spatial_layers': 3,
    'num_temporal_layers': 3,
    'dropout': 0.15,
    'use_sis': True,
    'use_amp': True,             # Mixed precision training
}
```

## Model Architecture Details

### Input/Output

- **Input**: `X ∈ R^(B×w×N)` where B is batch size, w is window size (14 days default), N is number of cities
- **Output**: `Î_{t+1} ∈ R^N`, predicted case counts for the next time step

### Loss Function

Total Loss = MSE Loss + λ_mae × MAE Loss + λ_outbreak × Outbreak Loss + λ_sis × SIS Consistency Loss

```
L = L_MSE + λ_mae × L_MAE + λ_outbreak × L_outbreak + λ_sis × L_SIS
```

Where:
- `L_MSE`: Mean Squared Error on log-transformed predictions
- `L_MAE`: Mean Absolute Error for robustness
- `L_outbreak`: Weighted loss emphasizing outbreak periods
- `L_SIS`: Consistency with SIS dynamics model

### SIS Dynamics Model

```
I_{t+1}^{SIS} = I_t + β(N - I_t)I_t/N - γI_t
```

Where:
- `I_t`: Current infection ratio
- `β`: Infection rate (learnable)
- `γ`: Recovery rate (learnable)
- `N`: Total population (normalized to 1)

## Output Files（与上文「最终结果说明」对应）

- `processed_data.pkl`：预处理后的训练/验证/测试数据。
- `checkpoints/`：`best_model_soft.pth`、`best_model_hard.pth` 及对应 `best_mae_model_*.pth`、`test_results_*.pkl`、`training_curves_*.png`。
- `visualization_results/`：fig1 与 CSV；`city_predictions/` 下为每城市/县图表（单模型为多 horizon，双模型为真实值+软+硬）。
- `visualization_results/soft_vs_hard_comparison/`：由 `compare_models.py` 生成的对比表与对比图。

## Evaluation Metrics

The model is evaluated using:
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Square Error)
- **MAPE** (Mean Absolute Percentage Error)
- **R²** (Coefficient of Determination)

## Future Directions

1. **Distance-Weighted Graphs**: Use geographic distance for weighted graph construction
2. **Mobility Graphs**: Incorporate population flow data
3. **Multi-Compartment Models**: Extend to SEIR and other complex models
4. **City-Specific Parameters**: Learn different SIS parameters per city
5. **External Features**: Include weather, temperature, and other environmental factors
6. **Uncertainty Quantification**: Add prediction intervals

## Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- tqdm

## License

MIT License
