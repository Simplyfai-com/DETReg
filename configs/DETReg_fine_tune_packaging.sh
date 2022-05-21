#!/usr/bin/env bash

set -x

EXP_DIR=./exps/DETReg_fine_tune_packaging
PY_ARGS=${@:1}

# Run training
python3 -u main.py --output_dir ${EXP_DIR} \
	 --filter_pct 0.01 \
	 --dataset packaging \
	 --pretrain exps/DETReg_top30_in100/checkpoint_coco.pth \
	 --resume exps/DETReg_fine_tune_packaging/checkpoint0479.pth \
	 --save_every 4 \
	 --epochs 500 \
	 --lr_drop 500 \
	 --batch_size 3 \
	 ${PY_ARGS}

# For generating viz
# python3 -u main.py --output_dir ${EXP_DIR} \
#                 --dataset packaging \
#                 --resume exps/DETReg_fine_tune_packaging/checkpoint0499.pth \
#                 --viz \
#                 ${PY_ARGS}
