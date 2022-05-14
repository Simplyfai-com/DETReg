#!/usr/bin/env bash

set -x

EXP_DIR=exps/DETReg_fine_tune_packaging
PY_ARGS=${@:1}

python3 -u main.py --output_dir ${EXP_DIR} \
                --filter_pct 0.01 \
                --dataset packaging \
                --pretrain exps/DETReg_top30_in100/checkpoint_coco.pth \
                --epochs 500 \
                --lr_drop 500 \
                ${PY_ARGS}