# 硬约束训练示例（8GB 显存请保持较小 batch；学习率用 --lr 字母 L）
Set-Location $PSScriptRoot\..
if (-not $env:DENGUE_DATALOADER_WORKERS) { $env:DENGUE_DATALOADER_WORKERS = "0" }
python train.py --constraint hard --exp-name my_try --lr 0.0003 --dropout 0.2 --num_epochs 150 --batch-size 64
