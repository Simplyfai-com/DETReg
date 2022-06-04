import torch
from datasets.coco import make_coco_transforms
from PIL import Image
import requests
from main import get_args_parser
from models import build_model
from argparse import Namespace
import glob
from util.box_ops import box_cxcywh_to_xyxy
from util.plot_utils import plot_results
from matplotlib import pyplot as plt
import numpy as np
import random
import time

args = {'lr': 0.0002, 'max_prop': 30, 'lr_backbone_names': ['backbone.0'], 'lr_backbone': 2e-05, 'lr_linear_proj_names': ['reference_points', 'sampling_offsets'], 'lr_linear_proj_mult': 0.1, 'batch_size': 4, 'weight_decay': 0.0001, 'epochs': 50, 'lr_drop': 40, 'lr_drop_epochs': None, 'clip_max_norm': 0.1, 'sgd': False, 'filter_pct': -1, 'with_box_refine': False, 'two_stage': False, 'strategy': 'topk', 'obj_embedding_head': 'intermediate', 'frozen_weights': None, 'backbone': 'resnet50', 'dilation': False, 'position_embedding': 'sine', 'position_embedding_scale': 6.283185307179586, 'num_feature_levels': 4, 'enc_layers': 6, 'dec_layers': 6, 'dim_feedforward': 1024, 'hidden_dim': 256, 'dropout': 0.1, 'nheads': 8, 'num_queries': 300, 'dec_n_points': 4, 'enc_n_points': 4, 'pretrain': '', 'load_backbone': 'swav', 'masks': False, 'aux_loss': True, 'set_cost_class': 2, 'set_cost_bbox': 5, 'set_cost_giou': 2, 'object_embedding_loss_coeff': 1, 'mask_loss_coef': 1, 'dice_loss_coef': 1, 'cls_loss_coef': 2, 'bbox_loss_coef': 5, 'giou_loss_coef': 2, 'focal_alpha': 0.25, 'dataset_file': 'coco', 'dataset': 'imagenet', 'data_root': 'data', 'coco_panoptic_path': None, 'remove_difficult': False, 'output_dir': '', 'cache_path': 'cache/ilsvrc/ss_box_cache', 'device': 'cuda', 'seed': 42, 'resume': '', 'eval_every': 1, 'start_epoch': 0, 'eval': False, 'viz': False, 'num_workers': 2, 'cache_mode': False, 'object_embedding_loss': False}
args = Namespace(**args)
model, criterion, postprocessors = build_model(args)
checkpoint = torch.hub.load_state_dict_from_url("https://github.com/amirbar/DETReg/releases/download/1.0.0/checkpoint_coco.pth", progress=True, map_location=torch.device('cpu'))
load_msg = model.load_state_dict(checkpoint['model'], strict=False)
transforms = make_coco_transforms('val')

all_filenames = glob.glob("data/packaging_coco_format/validation/data/*.jpg")
print(type(all_filenames[0]))
subset = random.sample(all_filenames, k=5)
times = []
for idx, img_path in enumerate(subset):
    # img_url = "https://ak.picdn.net/shutterstock/videos/24611465/thumb/11.jpg"
    # # img_url = "https://media.ktoo.org/2013/10/Brown-Bears.jpg"
    im = Image.open(img_path)
    im_t, _ = transforms(im, None)
    t_start = time.perf_counter()
    res = model(im_t.unsqueeze(0))
    t_end = time.perf_counter()
    times.append(t_end - t_start)
    scores = torch.sigmoid(res['pred_logits'][..., 1])
    pred_boxes = res['pred_boxes']


    img_w, img_h = im.size
    pred_boxes_ = box_cxcywh_to_xyxy(pred_boxes) * torch.Tensor([img_w, img_h, img_w, img_h])
    I = scores.argsort(descending = True) # sort by model confidence
    pred_boxes_ = pred_boxes_[0, I[0, :3]] # pick top 3 proposals
    scores_ = scores[0, I[0, :3]]

    plt.figure()
    plot_results(np.array(im), scores_, pred_boxes_, plt.gca(), norm=False)
    plt.axis('off')
    plt.savefig(f"result_{idx}.png")
mean_inference_time = np.mean(times)