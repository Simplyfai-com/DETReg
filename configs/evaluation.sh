#!/usr/bin/env bash

python3 main.py --dataset packaging \
                --resume exps/DETReg_fine_tune_packaging/checkpoint.pth \
                --eval