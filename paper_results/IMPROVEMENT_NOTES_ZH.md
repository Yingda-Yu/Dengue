# 实验结果与可改进方向（简要）

基于 `paper_experiments.py` 在测试集上的多步预测（式 27–30）及 **Base / Proposed(soft) / Proposed(hard)** 对比，若 **ARIMA 或 Base 仍优于两种 Proposed**，可考虑以下方向（需改代码或重训，非本文件自动执行）。

## 1. 硬约束 Proposed (hard)

- **含义**：推理时 \(\hat{Y} = I^{SIS} + \mathrm{softplus}(\mathrm{NN})\)，训练时通常 \(\lambda_{SIS}=0\)，结构已嵌入流行病学一步。
- **若仍差**：SIS 分支的输入若在 **log 标准化空间**而非真实病例尺度，动力学与数据不对齐，硬约束会把误差 **锁进 SIS 形状**；宜把 SIS 输入改为 **反变换后的病例数**或单独可学习标度，再重训 `train.py --constraint hard`。

## 2. 软约束 Proposed (soft)

- **\(\lambda_{SIS}\)**：网格搜索或减小，避免为迁就 SIS 而损害对真值的拟合。
- **早停指标**：可改为仅验证集 **MAE/RMSE（反变换后）**，避免验证 loss 中含过强的 SIS 项导致选到「SIS 友好但预测差」的 epoch。

## 3. 多步自回归

- 长 horizon 误差累积：可试 **scheduled sampling**、**多步直接预测头**（一次预测 \(H\) 天而非纯自回归）、或在损失中对远 horizon 加权。

## 4. 基线公平性与报告

- **ARIMA** 在低密度日序列上常很强；可补充 **分县 / 仅高发期** 子集表，或 **同期 VAR** 作为多元统计基线。
- **MAPE** 在真值近 0 时会爆炸；正文宜并列 **RMSE/MAE**，并说明 MAPE 计算方式（如 \(\epsilon\) 下限）。

## 5. 训练资源

- 在 **GPU** 上满 epoch 训练三种 GNN，保证 **soft / hard / none** 收敛程度相当后再比表。

---

重新生成带 **Hard** 的 CSV 与图：

```bash
python paper_experiments.py
python visualize_paper_table.py
```

若暂无 `best_model_hard.pth`，请先：`python train.py --constraint hard`。
