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
from models import Clipper, OpenClipper, BrainNetworkNoDETR, BrainDiffusionPrior, VersatileDiffusionPriorNetwork

import torch.distributed as dist
from accelerate import Accelerator

import argparse

import math
import random
import webdataset as wds


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
        choices=["01", "02", "05", "07"],
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

def do_contrastive_loss(clip_voxels, clip_target, temp, mixco, training, loss_type,
                        perm=None, betas=None, select=None, distributed=False, accelerator=None, 
                        local_rank=0, clip_trans=None):
    if mixco:
        if loss_type == 'all':
            loss_nce = utils.mixco_nce_all(
                nn.functional.normalize(clip_voxels, dim=-1), 
                nn.functional.normalize(clip_target, dim=-1),
                temp=temp, perm=perm, betas=betas, select=select,
                distributed=distributed, accelerator=accelerator, local_rank=local_rank)
        else:
            # this is used
            loss_nce = utils.mixco_nce(
                nn.functional.normalize(clip_voxels.flatten(1), dim=-1), 
                nn.functional.normalize(clip_target.flatten(1), dim=-1),
                temp=temp, perm=perm, betas=betas, select=select,
                distributed=distributed, accelerator=accelerator, local_rank=local_rank,
                bidirectional=args.bidir_mixco)
    else:
        if loss_type == 'all':
            if training:
                 loss_nce = utils.soft_cont_loss_all(
                    nn.functional.normalize(clip_voxels, dim=-1), 
                    nn.functional.normalize(clip_target, dim=-1),
                    nn.functional.normalize(clip_trans, dim=-1),
                    temp=temp,
                    distributed=distributed, accelerator=accelerator)
            else:
                loss_nce = utils.soft_clip_loss_all(
                    nn.functional.normalize(clip_voxels, dim=-1), 
                    nn.functional.normalize(clip_target, dim=-1),
                    temp=temp,
                    distributed=distributed, accelerator=accelerator)
        else:
            if training:
                if args.soft_loss_type == "cont_flatten":
                    loss_nce = utils.soft_cont_loss(
                        nn.functional.normalize(clip_voxels.flatten(1), dim=-1), 
                        nn.functional.normalize(clip_target.flatten(1), dim=-1),
                        nn.functional.normalize(clip_trans.flatten(1), dim=-1),
                        temp=temp,
                        distributed=distributed, accelerator=accelerator)
                elif args.soft_loss_type == "cont_inter":
                    loss_nce_1 = utils.soft_clip_loss(
                        nn.functional.normalize(clip_voxels[:, 0], dim=-1), 
                        nn.functional.normalize(clip_target[:, 0], dim=-1),
                        temp=temp,
                        distributed=distributed, accelerator=accelerator)
                    loss_nce_2 = utils.soft_cont_loss(
                        nn.functional.normalize(clip_voxels[:, 1:].flatten(0, 1), dim=-1), 
                        nn.functional.normalize(clip_target[:, 1:].flatten(0, 1), dim=-1),
                        nn.functional.normalize(clip_trans[:, 1:].flatten(0, 1), dim=-1),
                        temp=temp,
                        distributed=distributed, accelerator=accelerator)
                    loss_nce = (loss_nce_1 + loss_nce_2)/2
                else:
                    # this is used
                    loss_nce = utils.soft_clip_loss(
                        nn.functional.normalize(clip_voxels.flatten(1), dim=-1), 
                        nn.functional.normalize(clip_target.flatten(1), dim=-1),
                        temp=temp,
                        distributed=distributed, accelerator=accelerator)
            else:
                loss_nce = utils.soft_clip_loss(
                    nn.functional.normalize(clip_voxels.flatten(1), dim=-1), 
                    nn.functional.normalize(clip_target.flatten(1), dim=-1),
                    temp=temp,
                    distributed=distributed, accelerator=accelerator)
    return loss_nce


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
    
    normed_mse = args.normed_mse
    learned_query_mode = args.learned_query_mode
    use_projector = True

    resume_from_ckpt = args.ckpt_path is not None
    ckpt_path = args.ckpt_path
    use_versatile = not args.no_versatile

    lr_scheduler = 'cycle'
    initial_lr = 5e-4 # only used if lr_scheduler is 'fixed'
    max_lr = 3e-4
    alpha = 3

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

    if not args.disable_image_aug:
        train_augs = AugmentationSequential(
            kornia.augmentation.RandomResizedCrop((240,240), (0.6,1), p=0.3),
            kornia.augmentation.Resize((224, 224)),
            kornia.augmentation.RandomGaussianBlur(kernel_size=(7,7), sigma=(5,5), p=0.3),
            # kornia.augmentation.RandomHorizontalFlip(p=0.5),
            kornia.augmentation.ColorJiggle(brightness=0.2, contrast=0.2, saturation=0.3, hue=0., p=0.3),
        )
    else:
        train_augs = nn.Identity()

    # auto resume
    if os.path.exists(os.path.join(outdir, 'last.pth')):
        ckpt_path = os.path.join(outdir, 'last.pth')
        resume_from_ckpt = True

    print(accelerator.state)
    local_rank = accelerator.state.local_process_index
    world_size = accelerator.state.num_processes
    if num_devices <= 1 and world_size <= 1:
        distributed = False
    else:
        distributed = True
    
    if not args.disable_image_aug:
        try:
            clip_extractor = Clipper(clip_variant, clamp_embs=False, norm_embs=False, hidden_state=use_versatile, refine=False, 
                device=device, train_transforms=train_augs)
            print('Creating Clipper...')
        except AssertionError:
            clip_extractor = OpenClipper(clip_variant, weights_path, clamp_embs=False, norm_embs=False, device=device, train_transforms=train_augs)
            print('Creating Open Clipper...')
    print("distributed =",distributed,"num_devices =", num_devices, "local rank =", local_rank, "world size =", world_size)

    print('Pulling NSD webdataset data...')
    # local paths
    train_url = "/fsx/proj-medarc/fmri/natural-scenes-dataset/algonauts_data/wds/subj01_{2..98}.tar"
    val_url = "/fsx/proj-medarc/fmri/natural-scenes-dataset/algonauts_data/wds/subj01_{0..2}.tar"
    meta_url = "/fsx/proj-medarc/fmri/natural-scenes-dataset/algonauts_data/wds/metadata_subj01.json"

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

    voxel2clip_kwargs = dict(out_dim=out_dim, norm_type='ln', act_first=False, encoder_tokens=257, use_projector=use_projector)
    in_dims = {'01': 39548, '02': 14278, '05': 13039, '07':12682}
    voxel2clip_kwargs["in_dim"] = in_dims[subj_id]
    voxel2clip = BrainNetworkNoDETR(**voxel2clip_kwargs)

    print("params of voxel2clip:")
    if local_rank==0:
        utils.count_params(voxel2clip)

    # setup prior network
    depth = 6
    dim_head = 64
    heads = 12 # heads * dim_head = 12 * 64 = 768
    timesteps = 100
    if use_versatile:
        prior_network = VersatileDiffusionPriorNetwork(
            dim=out_dim,
            depth=depth,
            dim_head=dim_head,
            heads=heads,
            causal=args.causal,
            learned_query_mode=learned_query_mode
        ).to(device)
        # custom version that can fix seeds
        diffusion_prior = BrainDiffusionPrior(
            net=prior_network,
            image_embed_dim=out_dim,
            condition_on_text_encodings=False,
            timesteps=timesteps,
            cond_drop_prob=0.2,
            image_embed_scale=None,
            voxel2clip=voxel2clip
        ).to(device)
    else:
        diffusion_prior = BrainDiffusionPrior.from_pretrained(
        # kwargs for DiffusionPriorNetwork
        dict(),
        # kwargs for DiffusionNetwork
        dict(
            condition_on_text_encodings=False,
            timesteps=1000,
            voxel2clip=voxel2clip,
        ),
        voxel2clip_path=None,
        ckpt_dir='../train_logs/models/nousr_aesthetic_prior/'
    )
    
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    opt_grouped_parameters = [
        {'params': [p for n, p in diffusion_prior.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-2},
        {'params': [p for n, p in diffusion_prior.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(opt_grouped_parameters, lr=initial_lr) # lr doesnt get used if lr_scheduler='cycle'

    if lr_scheduler == 'fixed':
        lr_scheduler = None
    elif lr_scheduler == 'cycle':
        global_batch_size = batch_size * num_devices
        total_steps = num_epochs*(num_train//global_batch_size)
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=max_lr,
            total_steps=total_steps,
            final_div_factor=1000,
            last_epoch=-1, pct_start=2/num_epochs
        )

    def save_ckpt(tag):
        if tag == "last" and os.path.exists(os.path.join(outdir, f'{tag}.pth')):
            shutil.copyfile(os.path.join(outdir, f'{tag}.pth'), os.path.join(outdir, f'{tag}_old.pth'))
            # shutil.move(os.path.join(outdir, f'{tag}.pth'), os.path.join(outdir, f'{tag}_old.pth'))
        ckpt_path = os.path.join(outdir, f'{tag}.pth')
        print(f'saving {ckpt_path}',flush=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': diffusion_prior.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': losses,
            'val_losses': val_losses,
            'fwd_percent_correct': fwd_percent_correct,
            'bwd_percent_correct': bwd_percent_correct,
            'val_fwd_percent_correct': val_fwd_percent_correct,
            'val_bwd_percent_correct': val_bwd_percent_correct,
            'lrs': lrs,
            }, ckpt_path)
        if tag == "last" and os.path.exists(os.path.join(outdir, f'{tag}_old.pth')):
            os.remove(os.path.join(outdir, f'{tag}_old.pth'))

    print("\nDone with model preparations!")
    
    #--------WANDB-----------------
    if local_rank==0 and args.wandb_log:
        wandb_run = args.model_name
        wandb_notes = ''

        import wandb
        print(f"wandb {args.wandb_project} run {wandb_run}")
        wandb.login(host='https://stability.wandb.io')#, relogin=True)
        wandb_config = {
            "model_name": args.model_name,
            "modality": args.modality,
            "voxel_dims": args.voxel_dims,
            "clip_variant": args.clip_variant,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "disable_image_aug": args.disable_image_aug,
            "max_lr": max_lr,
            "lr_scheduler": lr_scheduler,
            # "clamp_embs": clamp_embs,
            "mixup_pct": mixup_pct,
            "num_train": num_train,
            "num_val": num_val,
            "seed": seed,
            "distributed": distributed,
            "num_devices": num_devices,
            "world_size": world_size,
            # "resume_from_ckpt": resume_from_ckpt,
            # "ckpt_path": ckpt_path,
            "train_url": train_url,
            "val_url": val_url,
        }
        print("wandb_config:\n",wandb_config)
        wandb.init(
            id = model_name,
            project=args.wandb_project,
            name=wandb_run,
            config=wandb_config,
            notes=wandb_notes,
            resume="allow"
        )
            
    #----ACCELERATE------------
    diffusion_prior, optimizer, train_dl, val_dl, lr_scheduler = accelerator.prepare(
        diffusion_prior, optimizer, train_dl, val_dl, lr_scheduler
    )

    epoch = 0
    losses, mse_losses, val_losses, lrs = [], [], [], []
    best_val_loss = 1e9
    soft_loss_temps = utils.cosine_anneal(0.004, 0.0075, num_epochs - int(mixup_pct * num_epochs))

    voxel0 = image0 = val_voxel0 = val_image0 = None

    # Optionally resume from checkpoint #
    if resume_from_ckpt:
        print("\n---resuming from ckpt_path---\n", ckpt_path)
        checkpoint = torch.load(ckpt_path, map_location=device)
        epoch = checkpoint['epoch']+1
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])        
        diffusion_prior.load_state_dict(checkpoint['model_state_dict'])
        global_batch_size = batch_size * num_devices
        total_steps_done = epoch*(num_train//global_batch_size)
        for _ in range(total_steps_done):
            lr_scheduler.step()
        del checkpoint
        torch.cuda.empty_cache()

    progress_bar = tqdm(range(epoch,num_epochs), disable=(local_rank!=0))
    for epoch in progress_bar:
        diffusion_prior.train()

        sims = 0.
        sims_base = 0.
        val_sims = 0.
        val_sims_base = 0.
        fwd_percent_correct = 0.
        bwd_percent_correct = 0.
        val_fwd_percent_correct = 0.
        val_bwd_percent_correct = 0.
        loss_nce_sum = 0.
        loss_prior_sum = 0.
        val_loss_nce_sum = 0.
        val_loss_prior_sum = 0.

        for train_i, (voxel, image, latent) in enumerate(train_dl):
            optimizer.zero_grad()

            image = image.float().to(device)
            voxel = voxel.float().to(device)
            voxel = utils.voxel_select(voxel)

            if epoch < int(mixup_pct * num_epochs):
                voxel, perm, betas, select = utils.mixco(voxel)

            clip_target = latent.to(device).float().squeeze(1)
            
            if not args.disable_image_aug:
                apply_spatial_transforms = epoch >= int(mixup_pct * num_epochs)
                clip_trans = clip_extractor.embed_image(image, apply_transforms=True, 
                    apply_spatial_transforms=apply_spatial_transforms).float()
                clip_target.to(voxel.dtype)
                clip_trans.to(voxel.dtype)
            else:
                clip_trans = clip_target

            # mixup diffusion targets as well
            if epoch < int(mixup_pct * num_epochs):
                betas_shape = [-1] + [1]*(len(clip_target.shape)-1)
                clip_target_prior = clip_target * betas.reshape(*betas_shape) + clip_target[perm] * (1-betas.reshape(*betas_shape))
            else:
                clip_target_prior = clip_target
            if normed_mse:
                clip_target_prior = clip_target_prior/(clip_target_prior[:, 0].norm(dim=-1).reshape(-1, 1, 1) + 1e-6)
            loss, pred, (clip_voxels_mse, clip_voxels) = diffusion_prior(image_embed=clip_target_prior, voxel=voxel)

            # distributed is not implemented for "_all" loss functions
            if epoch < int(mixup_pct * num_epochs):
                loss_nce = torch.tensor(0.).to(device)
                if 'v2c' in args.cont_loss_cands:
                    loss_nce += do_contrastive_loss(
                        clip_voxels, clip_trans, 0.006, mixco=True, training=True, 
                        loss_type=loss_type,
                        perm=perm, betas=betas, select=select, 
                        distributed=distributed, accelerator=accelerator, 
                        local_rank=local_rank, clip_trans=None
                    )
                if 'prior' in args.cont_loss_cands:
                    loss_nce += do_contrastive_loss(
                        pred, clip_trans, 0.006, mixco=True, training=True, 
                        loss_type=loss_type,
                        perm=perm, betas=betas, select=select, 
                        distributed=distributed, accelerator=accelerator, 
                        local_rank=local_rank, clip_trans=None
                    )
            else:
                epoch_temp = soft_loss_temps[epoch-int(mixup_pct*num_epochs)]
                loss_nce = torch.tensor(0.).to(device)
                if 'v2c' in args.cont_loss_cands:
                    loss_nce += do_contrastive_loss(
                        clip_voxels, clip_target, epoch_temp, mixco=False, training=True, 
                        loss_type=loss_type,
                        perm=None, betas=None, select=None, 
                        distributed=distributed, accelerator=accelerator, 
                        local_rank=local_rank, clip_trans=clip_trans
                    )
                if 'prior' in args.cont_loss_cands:
                    loss_nce += do_contrastive_loss(
                        pred, clip_target, epoch_temp, mixco=False, training=True, 
                        loss_type=loss_type,
                        perm=None, betas=None, select=None, 
                        distributed=distributed, accelerator=accelerator, 
                        local_rank=local_rank, clip_trans=clip_trans
                    )

            if 'v2c' in args.mse_loss_cands:
                loss += 1000 * F.mse_loss(
                    clip_voxels_mse,
                    clip_target/(clip_target[:, 0].norm(dim=-1).reshape(-1, 1, 1) + 1e-6)
                )

            loss_nce_sum += loss_nce.item()
            loss_prior_sum += loss.item()
            
            loss = alpha * loss + loss_nce
            utils.check_loss(loss)
            losses.append(loss.item())
            lrs.append(optimizer.param_groups[0]['lr'])

            if distributed:
                sims_base += F.cosine_similarity(accelerator.gather(clip_target),
                                                      accelerator.gather(clip_voxels), dim=-1).mean().item()
            else:
                sims_base += F.cosine_similarity(clip_target,clip_voxels, dim=-1).mean().item()

            # forward and backward top 1 accuracy
            labels = torch.arange(len(clip_target)).to(device)
            fwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_target.flatten(1), clip_voxels.flatten(1)), labels, k=1).item()
            bwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_voxels.flatten(1), clip_target.flatten(1)), labels, k=1).item()

            accelerator.backward(loss)
            optimizer.step()

            if lr_scheduler is not None:
                lr_scheduler.step()
            
            # logs = {"train/loss": np.mean(losses[-(train_i+1):]),
            #         "train/lr": lrs[-1],
            #         "train/num_steps": len(losses),
            #         "train/cosine_sim_base": sims_base / (train_i + 1),
            #         "train/fwd_pct_correct": fwd_percent_correct / (train_i + 1),
            #         "train/bwd_pct_correct": bwd_percent_correct / (train_i + 1),
            #         "train/loss_nce": loss_nce_sum / (train_i + 1),
            #         "train/mse_loss": loss_prior_sum / (train_i + 1),
            #         # "train/alpha": alpha,
            #     }
            # progress_bar.set_postfix(**logs)
        
        if local_rank==0 and epoch==num_epochs-1: 
            diffusion_prior.eval()
            for val_i, (voxel, latent) in enumerate(val_dl): 
                with torch.no_grad():
                    voxel = voxel.float().to(device)
                    voxel = voxel.mean(1)
                    
                    clip_target = latent.to(device).float().squeeze(1)
                    
                    if normed_mse:
                        clip_target_prior = clip_target/(clip_target[:, 0].norm(dim=-1).reshape(-1, 1, 1) + 1e-6)
                    else:
                        clip_target_prior = clip_target
                    
                    loss, pred, (clip_voxels_mse, clip_voxels) = diffusion_prior(
                        image_embed=clip_target_prior,
                        voxel=voxel
                    )
                    if epoch < int(mixup_pct * num_epochs):
                        loss_nce = torch.tensor(0.).to(device)
                        if 'v2c' in args.cont_loss_cands:
                            loss_nce += do_contrastive_loss(
                                clip_voxels, clip_target, 0.006, mixco=True, training=False,
                                loss_type=loss_type,
                                perm=None, betas=None, select=None, 
                                distributed=distributed, accelerator=accelerator, 
                                local_rank=local_rank, clip_trans=None
                            )
                        if 'prior' in args.cont_loss_cands:
                            loss_nce += do_contrastive_loss(
                                pred, clip_target, 0.006, mixco=True, training=False,
                                loss_type=loss_type,
                                perm=None, betas=None, select=None, 
                                distributed=distributed, accelerator=accelerator, 
                                local_rank=local_rank, clip_trans=None
                            )
                    else:
                        epoch_temp = soft_loss_temps[epoch-int(mixup_pct*num_epochs)]
                        loss_nce = torch.tensor(0.).to(device)
                        if 'v2c' in args.cont_loss_cands:
                            loss_nce += do_contrastive_loss(
                                clip_voxels, clip_target, epoch_temp, mixco=False, training=False,
                                loss_type=loss_type,
                                perm=None, betas=None, select=None, 
                                distributed=distributed, accelerator=accelerator, 
                                local_rank=local_rank, clip_trans=None
                            )
                        if 'prior' in args.cont_loss_cands:
                            loss_nce += do_contrastive_loss(
                                pred, clip_target, epoch_temp, mixco=False, training=False,
                                loss_type=loss_type,
                                perm=None, betas=None, select=None, 
                                distributed=distributed, accelerator=accelerator, 
                                local_rank=local_rank, clip_trans=None
                            )
                    
                    val_loss_nce_sum += loss_nce.item()
                    val_loss_prior_sum += loss.item()
                    val_loss = alpha * loss + loss_nce
                    val_losses.append(val_loss.item())

                    if distributed:
                        val_sims_base += F.cosine_similarity(accelerator.gather(clip_target),
                                                            accelerator.gather(clip_voxels),dim=-1).mean().item()
                    else:
                        val_sims_base += F.cosine_similarity(clip_target,clip_voxels,dim=-1).mean().item()

                    labels = torch.arange(len(clip_target)).to(device)
                    val_fwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_target.flatten(1), clip_voxels.flatten(1)), labels, k=1).item()
                    val_bwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_voxels.flatten(1), clip_target.flatten(1)), labels, k=1).item()
            
            logs = {"train/loss": np.mean(losses[-(train_i+1):]),
                    "val/loss": np.mean(val_losses[-(val_i+1):]),
                    "train/lr": lrs[-1],
                    "train/num_steps": len(losses),
                    "val/num_steps": len(val_losses),
                    "train/cosine_sim_base": sims_base / (train_i + 1),
                    "val/cosine_sim_base": val_sims_base / (val_i + 1),
                    "train/fwd_pct_correct": fwd_percent_correct / (train_i + 1),
                    "train/bwd_pct_correct": bwd_percent_correct / (train_i + 1),
                    "val/val_fwd_pct_correct": val_fwd_percent_correct / (val_i + 1),
                    "val/val_bwd_pct_correct": val_bwd_percent_correct / (val_i + 1),
                    "train/loss_nce": loss_nce_sum / (train_i + 1),
                    "train/mse_loss": loss_prior_sum / (train_i + 1),
                    "val/loss_nce": val_loss_nce_sum / (val_i + 1),
                    "val/mse_loss": val_loss_prior_sum / (val_i + 1),
                    # "train/alpha": alpha,
                }
            progress_bar.set_postfix(**logs)
            
            if args.wandb_log:
                while True:
                    try:
                        wandb.log(logs)
                        break
                    except:
                        print('Wandb log failed. Retrying')
                        time.sleep(1)
            
            try:
                del clip_voxels, clip_voxels_mse, clip_target, image, voxel, pred, clip_target_prior
            except:
                pass

        if ckpt_saving and local_rank==0:
            try:
                save_ckpt(f'last')
            except: 
                pass

    if args.wandb_log and local_rank==0:
        wandb.finish()

    print("\n===Finished!===\n")