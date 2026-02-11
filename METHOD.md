# Method: Epidemiology-Informed Spatio-Temporal Graph Neural Network for Dengue Prediction

本文档以论文 **Method** 的格式详述本项目的算法核心，包括问题形式化、数据预处理、图构建、模型架构、损失函数与训练流程。公式与实现与仓库中 `model.py`、`train.py`、`data_preprocessing.py` 一致。

**默认超参与维度（与代码一致）**：\(w=14\)，\(N=22\)，\(d_f=8\)，\(d_s=128\)，\(d_t=256\)，空间/时间层数各 3，多头数 4，\(\rho=100\)（SIS 归一化尺度），\(\lambda_{\mathrm{MAE}}=0.5\)，\(\lambda_{\mathrm{outbreak}}=0.3\)，\(\lambda_{\mathrm{SIS}}=0.05\)（软约束）或 0（硬约束）。

---

## 1. Problem Formulation

### 1.1 Notation

- **空间维度**：设共有 \(N\) 个县/市（节点），记为 \(\mathcal{V} = \{v_1, \ldots, v_N\}\)。
- **时间维度**：日级病例序列，时间步 \(t \in \{1,\ldots,T\}\)，\(T\) 为总天数。
- **观测**：\(I_{t}^{(n)} \in \mathbb{R}_{\geq 0}\) 表示在时刻 \(t\)、节点（县市）\(n\) 的当日确诊登革热病例数。
- **矩阵形式**：\(\mathbf{I}_t = [I_{t}^{(1)}, \ldots, I_{t}^{(N)}]^\top \in \mathbb{R}^N\) 为 \(t\) 时刻所有节点的病例向量；整体观测矩阵为 \(\mathbf{I} \in \mathbb{R}^{T \times N}\)。

### 1.2 预测任务

给定过去 \(w\) 天的病例序列（滑动窗口）：

\[
\mathbf{X} = [\mathbf{I}_{t-w+1}, \ldots, \mathbf{I}_t] \in \mathbb{R}^{w \times N},
\]

预测下一时刻（或未来第 \(h\) 步）的病例向量 \(\mathbf{I}_{t+1} \in \mathbb{R}^N\)（本实现中 \(h=1\)，即单步预测；多步通过自回归实现）。  
即学习映射 \(f_\theta: \mathbb{R}^{w \times N} \to \mathbb{R}^N\)，使得 \(\hat{\mathbf{I}}_{t+1} = f_\theta(\mathbf{X})\) 在适当损失下逼近真实 \(\mathbf{I}_{t+1}\)。

---

## 2. Data Preprocessing

### 2.1 原始数据与聚合

- **输入**：原始 CSV 记录，每条包含通报日期（Date_Notification 或发病日期 Date_Onset）、县市（MOI_County_living）、确诊人数（Number_of_confirmed_cases）等。
- **聚合**：按 (日期, 县市) 对确诊人数求和，得到日级、县市级的病例数。缺失县市或日期的记录被剔除。
- **矩阵化**：对全部日期 \(\mathcal{T}\) 与全部县市 \(\mathcal{V}\) 构造矩阵 \(\mathbf{I} \in \mathbb{R}^{T \times N}\)，其中 \(I_{t,n}\) 为日期 \(t\)、县市 \(n\) 的病例数；缺失日期的县市在该日填 0。

### 2.2 序列构造与划分

- **滑动窗口**：给定窗口长度 \(w\)（默认 14 天）与预测步长 \(h=1\)，对 \(\mathbf{I}\) 构造样本：
  - 输入 \(\mathbf{X}^{(i)} = \mathbf{I}_{i:i+w,\,:} \in \mathbb{R}^{w \times N}\)
  - 目标 \(\mathbf{y}^{(i)} = \mathbf{I}_{i+w+h-1,\,:} \in \mathbb{R}^N\)
  样本数 \(S = T - w - h + 1\)。
- **划分**：按时间顺序将样本划分为训练集、验证集、测试集，比例默认 0.7 : 0.15 : 0.15（无打乱），保证时间因果性。

### 2.3 输入标准化（训练/推理一致）

为缓解病例数的长尾分布并便于训练，对输入与目标做 **log1p + 标准化**（在训练集上拟合，验证/测试集复用）：

\[
\tilde{x} = \frac{\log(1 + x) - \mu_{\log}}{\sigma_{\log} + \epsilon},
\]

其中 \(\mu_{\log}\)、\(\sigma_{\log}\) 为训练集上 \(\log(1+\mathbf{X})\) 的均值和标准差，\(\epsilon=10^{-8}\)。预测时对模型输出做逆变换：先反标准化再 \(\max(\exp(\tilde{y})-1, 0)\) 得到原始尺度的病例预测。

---

## 3. Graph Construction

将 \(N\) 个县市视为图 \(\mathcal{G} = (\mathcal{V}, \mathcal{E})\) 的节点。

- **节点**：\(\mathcal{V}\) 与县市一一对应，\(|\mathcal{V}| = N\)。
- **边**：本实现采用 **全连接图**，即 \(\mathcal{E} = \{(i,j) : i,j \in \{1,\ldots,N\},\, i \neq j\}\)。边表以 COO 形式存储为 \(\texttt{edge\_index} \in \mathbb{Z}^{2 \times |\mathcal{E}|}\)。
- **动机**：允许任意县市间进行信息传递，空间模块通过可学习注意力对边权重进行隐式建模；若后续引入地理或流动数据，可将 \(\mathcal{E}\) 改为邻接或加权边并配合邻接掩码使用。

---

## 4. Model Architecture

整体模型 \(f_\theta\) 为 **EpidemiologyGNNv2**，由以下部分组成：输入特征编码 → 时空编码（每时间步空间图编码 + 时间编码）→ 输出头 → 可选 SIS 模块与软/硬约束输出。下面逐块给出维度与公式。

### 4.1 输入与特征编码

- **输入**：\(\mathbf{X} \in \mathbb{R}^{B \times w \times N}\)，\(B\) 为 batch size。
- **特征编码**：对每个标量病例值先扩展为 1 维再通过共享 MLP 映射到 \(d_f\) 维（代码中 \(d_f=8\)），便于后续空间与时间模块利用更丰富表示：

\[
\mathbf{Z}_{t,n} = \mathrm{FeatureEncoder}(I_{t,n}) \in \mathbb{R}^{d_f}, \quad \mathbf{Z} \in \mathbb{R}^{B \times w \times N \times d_f}.
\]

FeatureEncoder 为 Linear(1→32) → GELU → Linear(32→\(d_f\)) → LayerNorm。

### 4.2 空间模块（Graph-Level Spatial Encoding）

对每个时间步 \(t\)，将 \(\mathbf{Z}_{t,:,:} \in \mathbb{R}^{B \times N \times d_f}\) 视为图上节点特征，做 **空间图编码**，得到 \(\mathbf{H}^{(\mathrm{sp})}_t \in \mathbb{R}^{B \times N \times d_s}\)（\(d_s\) 为空间隐藏维）。

**空间编码器**由以下组成：

1. **线性投影**：\(\mathbf{U}_t = \mathrm{Linear}_{d_f \to d_s}(\mathbf{Z}_t)\)，再经 GELU。
2. **多层空间注意力（简化版 GAT）**：对节点维做多头自注意力，使每个节点聚合所有节点的信息（等价于在全连接图上做图注意力）。设第 \(\ell\) 层输入为 \(\mathbf{H}^{(\ell)} \in \mathbb{R}^{B \times N \times d_s}\)：
   - 计算 \(Q,K,V = \mathrm{Linear}_{d_s \to d_s}(\mathbf{H}^{(\ell)})\)，并拆成 \(H\) 个头，头维度 \(d_h = d_s / H\)。
   - 注意力权重（对最后一维做 softmax 前可施加邻接掩码，本实现中未使用）：
     \[
     A_{n,n'} = \mathrm{softmax}_{n'}\left( \frac{(Q_n)^\top K_{n'}}{\sqrt{d_h}} \right).
     \]
   - 输出：\(\mathbf{H}^{(\ell+1)} = \mathrm{LayerNorm}\big( \mathbf{H}^{(\ell)} + \mathrm{MultiHeadAttn}(Q,K,V) \big)\)，再接 Dropout。
3. **输出投影**：\(\mathbf{H}^{(\mathrm{sp})}_t = \mathrm{LayerNorm}\big( \mathrm{Linear}_{d_s \to d_s}(\mathbf{H}^{(\mathrm{last})}) \big)\).

因此，对每个 \(t\)，空间模块输出 \(\mathbf{H}^{(\mathrm{sp})}_t \in \mathbb{R}^{B \times N \times d_s}\)，汇总为 \(\mathbf{H}^{(\mathrm{sp})} \in \mathbb{R}^{B \times w \times N \times d_s}\)。

### 4.3 时间模块（Per-Node Temporal Encoding）

对每个节点 \(n\)，取其时间维序列 \(\mathbf{H}^{(\mathrm{sp})}_{:,n,:} \in \mathbb{R}^{B \times w \times d_s}\)，通过 **共享** 的时间编码器得到向量 \(\mathbf{h}_n \in \mathbb{R}^{d_t}\)（\(d_t\) 为时间隐藏维）。实现上先将张量重塑为 \((B \cdot N, w, d_s)\)，经时间编码后再恢复为 \((B, N, d_t)\)。

**时间编码器**结构为：

1. **多尺度时间卷积**：3 个 1D 卷积分支，膨胀率分别为 1, 2, 4，kernel size 3，输出拼接后经 Linear + LayerNorm，得到多尺度局部特征。
2. **位置编码**：加性正弦/余弦位置编码（Transformer 风格），维度与序列维度一致。
3. **LSTM**：多层单向 LSTM，输入维 \(d_s/2\)，隐藏维 \(d_t\)，输出整段序列的隐藏状态。
4. **时间注意力**：对 LSTM 输出序列做 Multi-Head Self-Attention + 残差与 LayerNorm。
5. **取末步**：取最后时间步的表示并 Dropout，得到 \(\mathbf{h}_n \in \mathbb{R}^{d_t}\)。

因此，时间模块输出 \(\mathbf{H}^{(\mathrm{temp})} \in \mathbb{R}^{B \times N \times d_t}\)。

### 4.4 输出头与残差

- **原始输出**：\(\mathbf{o} = \mathrm{OutputHead}(\mathbf{H}^{(\mathrm{temp})}) \in \mathbb{R}^{B \times N}\)，OutputHead 为 MLP（含 GELU、Dropout），输出标量/节点。
- **残差**：记 \(\mathbf{I}_{\mathrm{last}} = \mathbf{X}_{:,-1,:} \in \mathbb{R}^{B \times N}\) 为输入窗口最后一天病例，则
  \[
  \mathbf{o}_{\mathrm{raw}} = \mathbf{o} + \alpha \cdot \mathbf{I}_{\mathrm{last}},
  \]
  其中 \(\alpha \in \mathbb{R}\) 为可学习标量（代码中记为 \(\texttt{residual\_weight}\)）。

### 4.5 SIS 动力学模块（可学习）

为引入流行病学先验，定义 **SIS 正则模块**，基于当前病例 \(\mathbf{I}_t\) 预测下一步的“动力学”病例数（归一化到 \([0,1]\) 后应用 SIS 再反归一化）：

- **归一化**：\(\tilde{I}_n = \mathrm{clamp}(I_n / \rho,\, 0,\, 1)\)，\(\rho\) 为尺度常数（如 100）。
- **易感者**：\(S_n = 1 - \tilde{I}_n\)。
- **SIS 更新**（每节点可学习 \(\beta_n, \gamma_n\)，由 \(\mathrm{sigmoid}(\cdot)\) 约束在 \((0,0.5)\) 与 \((0,0.3)\)）：
  \[
  \tilde{I}_{n,\mathrm{next}} = \mathrm{clamp}\Big( \tilde{I}_n + \beta_n S_n \tilde{I}_n - \gamma_n \tilde{I}_n,\; 0,\; 1 \Big).
  \]
- **反归一化**：\(I_{n,\mathrm{next}}^{\mathrm{SIS}} = \tilde{I}_{n,\mathrm{next}} \cdot \rho\)。

模块输出记为 \(\mathbf{I}^{\mathrm{SIS}} \in \mathbb{R}^{B \times N}\)，用于软约束损失或硬约束输出。

### 4.6 软约束与硬约束输出

- **软约束（Soft Constraint）**：模型输出仅由神经网络与残差决定，SIS 仅通过损失项进行约束：
  \[
  \hat{\mathbf{I}}_{t+1} = \mathrm{softplus}(\mathbf{o}_{\mathrm{raw}}).
  \]
  训练时损失中包含与 \(\mathbf{I}^{\mathrm{SIS}}\) 的一致性项（见下节）。

- **硬约束（Hard Constraint）**：输出**结构上**依赖 SIS，神经网络只学习残差：
  \[
  \hat{\mathbf{I}}_{t+1} = \mathbf{I}^{\mathrm{SIS}} + \mathrm{softplus}(\mathbf{o}).
  \]
  此处 \(\mathbf{o}\) 为 OutputHead 直接输出（不加残差 \(\alpha \cdot \mathbf{I}_{\mathrm{last}}\)），表示对 SIS 预测的修正；训练时通常设 \(\lambda_{\mathrm{SIS}}=0\)，不再加 SIS 一致性损失。

两种模式通过前向时的布尔标志 \(\texttt{use\_hard\_constraint}\) 切换，便于同一套参数或同一类架构下对比软/硬约束效果。

---

## 5. Loss Function

在 **软约束** 模式下，监督损失由以下四项组成（硬约束下可设 \(\lambda_{\mathrm{SIS}}=0\)）。

### 5.1 符号

- \(\hat{\mathbf{I}} \in \mathbb{R}^{B \times N}\)：模型预测（已为 log 空间或与目标同尺度，取决于实现；评估时统一反变换到病例数尺度）。
- \(\mathbf{Y} \in \mathbb{R}^{B \times N}\)：目标（与 \(\hat{\mathbf{I}}\) 同尺度）。
- \(\mathbf{I}^{\mathrm{SIS}}\)：SIS 模块输出（与 \(\hat{\mathbf{I}}\) 同尺度）。

### 5.2 各项定义

1. **MSE 损失**：
   \[
   \mathcal{L}_{\mathrm{MSE}} = \frac{1}{BN} \sum_{b,n} (\hat{I}_{b,n} - Y_{b,n})^2.
   \]

2. **MAE 损失**：
   \[
   \mathcal{L}_{\mathrm{MAE}} = \frac{1}{BN} \sum_{b,n} \big| \hat{I}_{b,n} - Y_{b,n} \big|.
   \]

3. **爆发期加权损失（Smooth L1 + 权重）**：设权重 \(w_{b,n} = 2\) 若 \(Y_{b,n} > \tau\)，否则 \(w_{b,n}=1\)（\(\tau\) 为爆发阈值，如 0.5 在标准化空间）：
   \[
   \mathcal{L}_{\mathrm{outbreak}} = \frac{1}{BN} \sum_{b,n} w_{b,n} \cdot \mathrm{SmoothL1}(\hat{I}_{b,n},\, Y_{b,n}).
   \]

4. **SIS 一致性损失**（仅软约束时启用）：
   \[
   \mathcal{L}_{\mathrm{SIS}} = \frac{1}{BN} \sum_{b,n} \big| \hat{I}_{b,n} - I^{\mathrm{SIS}}_{b,n} \big|.
   \]

### 5.3 总损失

\[
\mathcal{L} = \mathcal{L}_{\mathrm{MSE}} + \lambda_{\mathrm{MAE}} \mathcal{L}_{\mathrm{MAE}} + \lambda_{\mathrm{outbreak}} \mathcal{L}_{\mathrm{outbreak}} + \lambda_{\mathrm{SIS}} \mathcal{L}_{\mathrm{SIS}}.
\]

典型设置：\(\lambda_{\mathrm{MAE}}=0.5\)，\(\lambda_{\mathrm{outbreak}}=0.3\)，\(\lambda_{\mathrm{SIS}}=0.05\)（软约束）；硬约束时 \(\lambda_{\mathrm{SIS}}=0\)。

---

## 6. Training Procedure

### 6.1 优化与学习率

- **优化器**：AdamW，权重衰减 \(10^{-5}\)，初始学习率 \(5\times10^{-4}\)，梯度范数裁剪（最大范数 1.0）。
- **学习率调度**：CosineAnnealingWarmRestarts，周期 \(T_0=30\)，\(T_{\mathrm{mult}}=2\)，最小学习率 \(\eta_{\min}=10^{-6}\)。

### 6.2 早停与模型选择

- **早停**：基于验证集上**原始尺度 MAE**（对预测与真实值做反标准化与 \(\mathrm{expm1}\) 后计算），若连续若干 epoch（如 50）无改善则停止。
- **检查点**：保存验证 loss 最优与验证 MAE 最优的两套权重，分别用于报告 loss 与 MAE；软/硬约束分别保存为 \(\texttt{best\_model\_soft.pth}\) 与 \(\texttt{best\_model\_hard.pth}\)。

### 6.3 训练模式

- **软约束**：前向使用 \(\texttt{use\_hard\_constraint=False}\)，损失中 \(\lambda_{\mathrm{SIS}}>0\)。
- **硬约束**：前向使用 \(\texttt{use\_hard\_constraint=True}\)，损失中 \(\lambda_{\mathrm{SIS}}=0\)。

两种模式除上述外（输出形式与 \(\lambda_{\mathrm{SIS}}\)）超参数一致，便于公平对比。

---

## 7. Inference and Multi-Step Forecasting

- **单步**：输入 \(\mathbf{X} \in \mathbb{R}^{w \times N}\)，经标准化后前向一次，得到 \(\hat{\mathbf{I}}_{t+1}\)，再反变换到病例数尺度。
- **多步**：自回归：以 \(\mathbf{X}_0 = \mathbf{X}\) 为初值，对 \(k=1,\ldots,K\) 预测 \(\hat{\mathbf{I}}_{t+k}\)，并将 \(\hat{\mathbf{I}}_{t+k}\) 拼入/滑动窗口得到 \(\mathbf{X}_k\)，再预测下一步；窗口长度保持 \(w\)，由最近 \(w\) 天（含已预测值）组成。

评估时对 horizon 3/7/14/30 天分别做多步预测，在原始尺度上计算 MAE、RMSE 等指标。

---

## 8. Summary of Algorithm Core

| 模块 | 作用 | 关键形式 |
|------|------|----------|
| 数据 | 日级县市病例矩阵 → 滑动窗口样本，log1p+标准化 | \(\tilde{x}=(\log(1+x)-\mu)/\sigma\) |
| 图 | 县市为节点，全连接边 | \(\mathcal{E} = \{(i,j): i\neq j\}\) |
| 空间 | 每时间步图上的节点表示 | 多层 GAT 式注意力 over nodes |
| 时间 | 每节点时间序列 → 单向量 | 多尺度 Conv + PE + LSTM + Attn → last step |
| 输出 | 节点标量 + 残差 + 非负 | softplus 或 SIS + softplus(NN) |
| SIS | 流行病学先验 | \(\tilde{I}_{n}' = \tilde{I}_n + \beta_n S_n \tilde{I}_n - \gamma_n \tilde{I}_n\) |
| 损失 | 数据拟合 + 爆发加权 + SIS 一致 | \(\mathcal{L}_{\mathrm{MSE}} + \lambda \mathcal{L}_{\mathrm{MAE}} + \lambda_o \mathcal{L}_{\mathrm{outbreak}} + \lambda_s \mathcal{L}_{\mathrm{SIS}}\) |

以上即为本项目的完整方法描述，可直接作为论文 Method 部分的算法与公式依据。
