#!/bin/bash

# --- 训练参数配置 ---
# 这些参数可以直接在 train.py 中通过命令行选项被识别和覆盖。
# 参数名必须与 train.py 中定义的命令行参数相匹配。
# 如果某个参数没有对应的命令行选项，你仍然需要修改 train.py 源代码。

# 模型路径和数据
OUT_DIR="/root/workspace/Infrared_project/stylegan2-ada-pytorch_change_gemini/training-runs/test-run"
DATASET_PATH="/root/workspace/datasets/improved_StyleGAN2/ImageSize/Ship_dataset_512.zip"

# 核心训练参数
GPUS=1
BATCH_SIZE=16 # 论文要求64
TOTAL_KIMG=7881  # 1000 个 Epochs 约等于 7881 kimg
SNAPSHOT_TICKS=1

# 优化器和正则化参数
# StyleGAN2-ADA 脚本通常通过 --cfg 控制学习率和 gamma 等，
# 但许多脚本也支持通过 --lr 和 --gamma 等参数来覆盖默认值。
LRATE=1e-3
ADAM_BETA1=0
ADAM_BETA2=0.5
GAMMA=8

# --- 运行训练 ---
echo "--- Starting StyleGAN2-ADA training with specified parameters ---"

python train.py \
  --outdir="$OUT_DIR" \
  --data="$DATASET_PATH" \
  --gpus="$GPUS" \
  --cfg=paper512 \
  --batch="$BATCH_SIZE" \
  --kimg="$TOTAL_KIMG" \
  --snap="$SNAPSHOT_TICKS" \
  --gamma="$GAMMA" 

echo "--- Training process finished. ---"