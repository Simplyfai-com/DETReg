#!/usr/bin/env bash

set -x

EXP_DIR=./exps/DETReg_fine_tune_10pct_packaging
PY_ARGS=${@:1}

# Run training
python3 -u main.py --output_dir ${EXP_DIR} \
	 --filter_pct 0.1 \
     --dataset_file coco \
	 --dataset packaging \
	 --pretrain exps/DETReg_top30_in100/checkpoint_coco.pth \
	 --save_every 4 \
	 --epochs 500 \
	 --lr_drop 500 \
	 --batch_size 3 \
	 ${PY_ARGS}