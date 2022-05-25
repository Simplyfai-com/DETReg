#!/usr/bin/env/bash

## This file configures pretraining for packaing dataset
EXP_DIR=exps/DETReg_top30_packaging

python3 -u main.py --output_dir ${EXP_DIR} \
    --dataset packaging_pretrain \
    --obj_embedding_head "head" \
    --strategy topk \
    --load_backbone swav \
    --max_prop 30 \
    --object_embedding_loss \
    --lr_backbone 0
