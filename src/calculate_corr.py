# # Import packages & functions

import os
import shutil
import sys
import traceback
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import kornia
from kornia.augmentation.container import AugmentationSequential

import utils
from utils import torch_to_matplotlib, torch_to_Image
from models import Clipper, OpenClipper, BrainNetworkNoDETR, ReverseBrainNetwork, VersatileDiffusionPriorNetwork, BrainDiffusionPrior

import torch.distributed as dist
from accelerate import Accelerator

import argparse

import math
import random
import webdataset as wds

from torchmetrics import PearsonCorrCoef

def parse_args():
    parser = argparse.ArgumentParser(description="Train prior")
    parser.add_argument(
        "--model_name",
        type=str,
        default="prior_257_test",
        help="name of model, used for wandb logging",
    )
    parser.add_argument(
        "--modality",
        type=str,
        default="image",
        choices=["image", "text"],
        help="image or text",
    )
    parser.add_argument(
        "--clip_variant",
        type=str,
        default="ViT-L/14",
        choices=["RN50", "ViT-L/14", "ViT-B/32"],
        help='clip variant',
    )
    # parser.add_argument(
    #     "--outdir",
    #     type=str,
    #     default=None,
    #     help="output directory for logs and checkpoints",
    # )
    parser.add_argument(
        "--wandb_log",
        action="store_true",
        help="whether to log to wandb",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="stability",
        help="wandb project name",
    )
    parser.add_argument(
        "--h5_dir",
        type=str,
        default='/scratch/gpfs/KNORMAN/nsdgeneral_hdf5/',
        help="directory containing COCO h5 files (only used for modality=text)",
    )
    parser.add_argument(
        "--voxel_dims",
        type=int,
        default=1,
        choices=[1, 3],
        help="1 for flattened input, 3 for 3d input",
    )
    parser.add_argument(
        "--remote_data",
        action="store_true",
        help="whether to pull data from huggingface",
    )
    parser.add_argument(
        "--wds_cache_dir",
        type=str,
        default='/tmp/wds-cache',
        help="directory for caching webdatasets fetched from huggingface",
    )
    parser.add_argument(
        "--disable_image_aug",
        action="store_true",
        help="whether to disable image augmentation (only used for modality=image)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="output location",
    )
    parser.add_argument(
        "--learned_query_mode",
        type=str,
        default="pos_emb",
        choices=["none", "token", "pos_emb", "all_pos_emb"],
        help="output location",
    )
    parser.add_argument(
        "--normed_mse",
        action="store_true",
        help="output location",
    )
    parser.add_argument(
        "--cont_loss_type",
        type=str,
        default="flatten",
        choices=["all", "flatten"],
        help="loss type",
    )
    parser.add_argument(
        "--no_full_train_set",
        action="store_true",
        help="whether to disable image augmentation (only used for modality=image)",
    )
    parser.add_argument(
        "--v2c_projector",
        action="store_true",
        help="whether to disable image augmentation (only used for modality=image)",
    )
    parser.add_argument(
        "--pretrained_v2c",
        action="store_true",
        help="whether to disable image augmentation (only used for modality=image)",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None
    )
    parser.add_argument(
        "--cont_loss_cands",
        choices=["none", "v2c", "prior", "v2c_prior"],
        default="v2c"
    )
    parser.add_argument(
        "--mse_loss_cands",
        choices=["prior", "v2c_prior"],
        default="prior"
    )
    parser.add_argument(
        "--mixup_pct",
        type=float,
        default=0.33
    )
    parser.add_argument(
        "--bidir_mixco",
        action="store_true",
        help="make mixco bidirectional"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=240
    )
    parser.add_argument(
        "--mixco_sel_thresh",
        type=float,
        default=0.5
    )
    parser.add_argument(
        "--soft_loss_type",
        choices=["clip", "cont_flatten", "cont_inter"],
        default="clip"
    )
    parser.add_argument(
        "--subj_id",
        choices=["01", "02", "03", "04", "05", "06", "07", "08"],
        default="01"
    )
    parser.add_argument(
        "--no_versatile",
        action="store_true",
    )
    parser.add_argument(
        "--causal",
        action="store_true",
    )
    return parser.parse_args()

if __name__ == '__main__':
    # Multi-GPU config #
    accelerator = Accelerator()
    print = accelerator.print # only print if local_rank=0

    device = accelerator.device
    print("device:",device)

    args = parse_args()
    print('args', args)

    model_name = args.model_name
    clip_variant = args.clip_variant  # "convnext_xxlarge"  # "ViT-L/14" # ("RN50", "ViT-L/14", "ViT-B/32")
    weights_path = None
    
    # params for all models
    seed = 0
    batch_size = 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    

    num_epochs = args.num_epochs
    lr_scheduler = 'cycle'
    initial_lr = 1e-3 #3e-5
    max_lr = 3e-4
    
    wandb_log = args.wandb_log
    wandb_project = 'laion-fmri'
    wandb_run_name = ''
    wandb_notes = ''
    
    ckpt_saving = True
    use_mp = False
    distributed = False
    save_at_end = False
    subj_id = args.subj_id

    cache_dir = 'cache'
    mixup_pct = args.mixup_pct
    loss_type = args.cont_loss_type

    resume_from_ckpt = args.ckpt_path is not None
    ckpt_path = args.ckpt_path

    if args.outdir is None:
        # outdir = os.path.expanduser(f'../train_logs/models/{model_name}/test')
        outdir = f'../train_logs/models/{args.model_name}'
    else:
        outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    
    # uses tf32 data type which is faster than standard float32
    torch.backends.cuda.matmul.allow_tf32 = True
    # need non-deterministic CuDNN for conv3D to work
    utils.seed_everything(seed, cudnn_deterministic=False)
    
    num_devices = torch.cuda.device_count()
    if num_devices==0: num_devices = 1
    num_workers = 1

    print(accelerator.state)
    local_rank = accelerator.state.local_process_index
    world_size = accelerator.state.num_processes
    if num_devices <= 1 and world_size <= 1:
        distributed = False
    else:
        distributed = True

    print('Pulling NSD webdataset data...')
    # local paths
    if args.subj_id in [3,6]:
        max_tar = 90
    elif args.subj_id in [4,8]:
        max_tar = 87
    else:
        max_tar = 98
    train_url = f"/fsx/proj-fmri/shared/algonauts_wds/subj{args.subj_id}_{{3..{max_tar}}}.tar"
    val_url = f"/fsx/proj-fmri/shared/algonauts_wds/subj{args.subj_id}_{{0..2}}.tar"
    meta_url = f"/fsx/proj-fmri/shared/algonauts_wds/metadata_subj{args.subj_id}.json"

    train_dl, val_dl, num_train, num_val = utils.get_dataloaders(
        batch_size,
        num_devices=num_devices,
        num_workers=num_workers,
        train_url=train_url,
        val_url=val_url,
        meta_url=meta_url,
        val_batch_size=300,
        cache_dir=args.wds_cache_dir,
        seed=seed,
        local_rank=local_rank,
    )

    print('Creating voxel2clip...')
    # size of the CLIP embedding for each variant
    clip_sizes = {"RN50": 1024, "ViT-L/14": 768, "ViT-B/32": 512}
    # output dim for voxel2clip model
    out_dim = clip_sizes[clip_variant]

    voxel2clip_kwargs = dict(out_dim=out_dim, norm_type='ln', act_first=False, encoder_tokens=257, use_projector=False)
    in_dims = {'01': 39548, '02': 39548, '03': 39548, '04': 39548, '05': 39548, '06': 39198, '07': 39548, '08': 39511}
    voxel2clip_kwargs["in_dim"] = in_dims[subj_id]
    voxel2clip = BrainNetworkNoDETR(**voxel2clip_kwargs)
    rev_v2c = ReverseBrainNetwork(**voxel2clip_kwargs)

    diff_prior_path = f'/fsx/proj-fmri/paulscotti/fMRI-Algonauts-Challenge-2023/train_logs/v2c_subj{args.subj_id}/last.pth'
    diff_prior_weights = torch.load(diff_prior_path, map_location=device)['model_state_dict']
    v2c_weights = {}
    for k,v in diff_prior_weights.items():
        if 'voxel2clip' in k:
            v2c_weights['.'.join(k.split('.')[1:])] = v
    voxel2clip.load_state_dict(v2c_weights, strict=False)
    del diff_prior_weights, v2c_weights
    voxel2clip.eval()
    voxel2clip.requires_grad_(False)
    voxel2clip.to(device)


    # rev_v2c.load_state_dict(torch.load('../train_logs/models/s2_test/last.pth', map_location=device)['model_state_dict'])
    # voxel2clip.eval()
    # voxel2clip.requires_grad_(False)
    # voxel2clip.to(device)
    # rev_v2c.eval()
    # rev_v2c.requires_grad_(False)
    # rev_v2c.to(device)


    # setup prior network
    depth = 6
    dim_head = 64
    heads = 12 # heads * dim_head = 12 * 64 = 768
    timesteps = 100
    prior_network = VersatileDiffusionPriorNetwork(
        dim=out_dim,
        depth=depth,
        dim_head=dim_head,
        heads=heads,
        causal=False,
        learned_query_mode="pos_emb"
    ).to(device)
    # custom version that can fix seeds
    rev_diffusion_prior = BrainDiffusionPrior(
        net=prior_network,
        image_embed_dim=out_dim,
        condition_on_text_encodings=False,
        timesteps=timesteps,
        cond_drop_prob=0.2,
        image_embed_scale=None,
        voxel2clip=None
    ).to(device)
    suff = f'_s{args.subj_id}' if args.subj_id != '01' else ''
    sd = torch.load(f'../train_logs/models/s3_s2_comb_nonpre{suff}/best.pth', map_location=device)
    rev_diffusion_prior.load_state_dict(sd['model_state_dict'])
    rev_diffusion_prior.eval()
    rev_diffusion_prior.requires_grad_(False)
    rev_v2c.load_state_dict(sd['rv2c_state_dict'])
    rev_v2c.eval()
    rev_v2c.requires_grad_(False)
    rev_v2c.to(device)
    del sd
    pearson = PearsonCorrCoef(in_dims[subj_id]).to(device)

    for val_i, (voxel, latent) in enumerate(val_dl): 
        with torch.inference_mode():
            voxel = voxel.float().to(device)
            voxel = voxel.mean(1)
            clip_target = latent.to(device).float().squeeze(1)

            intermediate_embs_orig = voxel2clip(voxel)
            
            g_cuda = torch.Generator(device=device)
            g_cuda.manual_seed(seed)
            intermediate_embs = rev_diffusion_prior.p_sample_loop(
                clip_target.shape, 
                dict(text_embed = clip_target), 
                cond_scale = 1., timesteps = timesteps,
                generator=g_cuda
            )
            _, ie_10, _ = rev_diffusion_prior(
                text_embed=clip_target, 
                image_embed=intermediate_embs_orig.to(device), 
                times=torch.ones(clip_target.shape[0], dtype=torch.long).to(device)*10
            )
            
            p10 = rev_v2c(ie_10)
            preds_orig = rev_v2c(intermediate_embs_orig)
            preds = rev_v2c(intermediate_embs)
            
            import pdb; pdb.set_trace()
            corr_coefs = pearson(preds, voxel)  # 30k
            metric = ((corr_coefs**2)*100).mean()
    print("Corr metric: ", metric)
    print("\n===Finished!===\n")