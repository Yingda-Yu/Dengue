# 硬约束模型调参指南（与当前结果简析）

## 你最近一次表里的现象（`table1_metrics_by_horizon.csv`）

- **ARIMA** 在 3/7/14/30 天 **RMSE 均为最低**（约 3.9→6.7），一元滚动 + 低密度计数序列上很常见。
- **Proposed (hard)** 在 **30 天** 上明显优于 **Base** 与 **Proposed (soft)**（RMSE ≈10.7 vs 11.9 / 17.6），MAPE 也更稳定，说明 **长 horizon 硬约束结构先验** 有价值。
- **短 horizon（3/7 天）** 仍是 **Base** 略优于 **Hard**：硬约束把输出绑在 SIS 形状附近，若短期更偏「纯拟合噪声/局部趋势」，无 SIS 的 Base 有时更灵活。

目标不必强求全面压过 ARIMA，可先 **拉近 RMSE** 或 **在 14/30 天上超过 ARIMA**（需运气与特征）；更现实的是 **写清对比协议** 并 **系统调 GNN 超参**。

---

## 已接入的工具

### 1. `train.py` 命令行超参（且不覆盖默认权重）

- `--exp-name <子目录名>`：权重写到 `checkpoints/<子目录>/best_model_hard.pth`，默认 `checkpoints/best_model_hard.pth` 不变。
- 可选：`--lr`、`--weight-decay`、`--dropout`、`--lambda-mae`、`--lambda-outbreak`、`--lambda-sis`（仅 soft）、`--batch-size`、`--spatial-hidden`、`--temporal-hidden`、`--num-spatial-layers`、`--num-temporal-layers`。

示例：

```bash
python train.py --constraint hard --exp-name my_try --lr 0.0003 --dropout 0.2 --num_epochs 150
```

### 2. `tune_hard_gnn.py` 批量试 8 组预设

```bash
python tune_hard_gnn.py --dry-run          # 先看会跑哪些命令
python tune_hard_gnn.py --epochs 120       # 逐组训练（耗时久）
```

结果追加到 **`paper_results/tuning_hard_runs.csv`**（含一步测试 MAE/RMSE）。可按 `test_mae` 排序选最优。

### 3. 用选中的权重跑完整 Table1（多 horizon）

```bash
python paper_experiments.py --checkpoint-hard checkpoints/t_hard_lr1e3/best_model_hard.pth
```

---

## 调参优先级建议

1. **学习率** `lr`：3e-4 ~ 1e-3 优先扫。
2. **dropout**：0.08 ~ 0.25，减轻过拟合或增强长 horizon 泛化。
3. **weight_decay**：1e-5 ~ 1e-4。
4. **batch_size**：显存允许时 512；略小有时更稳。
5. **lambda_mae / lambda_outbreak**：影响训练目标与爆发期权重，对 hard 仍生效（无 λ_sis）。

进一步再想 **改结构 / SIS 输入尺度 / 多步监督**，见 `IMPROVEMENT_NOTES_ZH.md`。
