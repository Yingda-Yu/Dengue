# Windows 上 PyTorch / CUDA 报错说明

## `curand64_10.dll` / `cufft64_11.dll` 加载失败

表示 **PyTorch 自带的 CUDA 动态库** 或 **其依赖** 未能加载，与 `train.py` 代码无关。请依次检查：

1. **NVIDIA 驱动**：到官网升级笔记本独显驱动。  
2. **Visual C++ 可再发行组件**：安装 [最新 VC++ x64](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)。  
3. **重装与驱动匹配的 PyTorch**：打开 https://pytorch.org/get-started/locally/ ，在 **mmseg**（或你用的）环境里按说明执行 `pip install` / `conda install`，避免同一环境里混装多个 CUDA 版本的 torch。

验证：

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

此处仍报错时，**不要运行** `train.py`，先修好环境。

## `MemoryError`（import torch 时）

- 关闭占用内存的程序后 **重启电脑** 再试。  
- 若仍失败，**新建 conda 环境** 并只安装 PyTorch + 本项目依赖。

## `CUDA error: out of memory`（训练时）

本模型内部相当于 **`batch_size × 城市数`** 条序列同时进 LSTM，**8GB 显卡用默认 512 几乎必爆显存**。

`train.py` 会在 **CUDA 且未设置 `DENGUE_DISABLE_BATCH_CAP=1`** 时按显存自动降低 `batch_size`（例如 ≤9GiB 时上限 **64**）。仍 OOM 时请再减小：

```powershell
python train.py --constraint hard --exp-name my_try --batch-size 32 --lr 0.0003 --num_epochs 150
```

大显存卡若要坚持 512，可设：`$env:DENGUE_DISABLE_BATCH_CAP = "1"`（不推荐在 8GB 上使用）。

训练脚本 **不再把 `edge_index` 放到 GPU**（当前模型未使用边张量），避免误报 OOM 位置在 `.to(cuda)`。  
**显存 &lt;10GiB** 时会 **强制 `num_workers=0`**，即使你设置了 `DENGUE_DATALOADER_WORKERS=4`。  
**≤9GiB** 会自动把 **隐藏维降为 64/128、层数≤2**；也可手动 `DENGUE_SMALL_MODEL=1`。  
仍 OOM 时可试：`python train.py ... --no-amp`

---

## 本项目已做的减轻措施

- **`train.py`**：Windows 下 **DataLoader 默认 `num_workers=0`**，减少多进程与 CUDA 同时加载时的冲突。需要加速时可设：

  ```powershell
  $env:DENGUE_DATALOADER_WORKERS = "4"
  python train.py ...
  ```

- **命令行笔误**：若误写 **`--1r`**，脚本在加载 torch 前会自动改为 **`--lr`** 并打印提示。  
- 支持 **`--learning-rate`**、**`--num-epochs`**（带连字符）与 **`--lr`**、**`--num_epochs`** 等价。

## 一键示例（参数已写对）

在项目根目录执行 **`scripts\train_hard_example.ps1`**，或手动：

```powershell
python train.py --constraint hard --exp-name my_try --lr 0.0003 --dropout 0.2 --num_epochs 150
```
