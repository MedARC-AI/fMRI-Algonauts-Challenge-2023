import numpy as np
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import PIL
from PIL import Image
import random
import os
import matplotlib.pyplot as plt
import pandas as pd
import math
import webdataset as wds
import json
import requests
import io
from urllib.request import Request, urlopen
import socket
from clip_retrieval.clip_client import ClipClient
import time 
import braceexpand
from models import Clipper,OpenClipper
from torchvision.utils import make_grid

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')

def seed_everything(seed=0, cudnn_deterministic=False):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
    else:
        ## needs to be False to use conv3D
        print('Note: not using cudnn.deterministic')

def np_to_Image(x):
    if x.ndim==4:
        x=x[0]
    return PIL.Image.fromarray((x.transpose(1, 2, 0)*127.5+128).clip(0,255).astype('uint8'))

def torch_to_Image(x):
    if x.ndim==4:
        x=x[0]
    return transforms.ToPILImage()(x)

def Image_to_torch(x):
    try:
        x = (transforms.ToTensor()(x)[:3].unsqueeze(0)-.5)/.5
    except:
        x = (transforms.ToTensor()(x[0])[:3].unsqueeze(0)-.5)/.5
    return x

def torch_to_matplotlib(x,device=device):
    if torch.mean(x)>10:
        x = (x.permute(0, 2, 3, 1)).clamp(0, 255).to(torch.uint8)
    else:
        x = (x.permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8)
    if device=='cpu':
        return x[0]
    else:
        return x.cpu().numpy()[0]

def pairwise_cosine_similarity(A, B, dim=1, eps=1e-8):
    #https://stackoverflow.com/questions/67199317/pytorch-cosine-similarity-nxn-elements
    numerator = A @ B.T
    A_l2 = torch.mul(A, A).sum(axis=dim)
    B_l2 = torch.mul(B, B).sum(axis=dim)
    denominator = torch.max(torch.sqrt(torch.outer(A_l2, B_l2)), torch.tensor(eps))
    return torch.div(numerator, denominator)

def batchwise_pearson_correlation(Z, B):
    # Calculate means
    Z_mean = torch.mean(Z, dim=1, keepdim=True)
    B_mean = torch.mean(B, dim=1, keepdim=True)

    # Subtract means
    Z_centered = Z - Z_mean
    B_centered = B - B_mean

    # Calculate Pearson correlation coefficient
    numerator = Z_centered @ B_centered.T
    Z_centered_norm = torch.linalg.norm(Z_centered, dim=1, keepdim=True)
    B_centered_norm = torch.linalg.norm(B_centered, dim=1, keepdim=True)
    denominator = Z_centered_norm @ B_centered_norm.T

    pearson_correlation = (numerator / denominator)
    return pearson_correlation

def batchwise_cosine_similarity(Z,B):
    # https://www.h4pz.co/blog/2021/4/2/batch-cosine-similarity-in-pytorch-or-numpy-jax-cupy-etc
    B = B.T
    Z_norm = torch.linalg.norm(Z, dim=1, keepdim=True)  # Size (n, 1).
    B_norm = torch.linalg.norm(B, dim=0, keepdim=True)  # Size (1, b).
    cosine_similarity = ((Z @ B) / (Z_norm @ B_norm)).T
    return cosine_similarity

def topk(similarities,labels,k=5):
    if k > similarities.shape[0]:
        k = similarities.shape[0]
    topsum=0
    for i in range(k):
        topsum += torch.sum(torch.argsort(similarities,axis=1)[:,-(i+1)] == labels)/len(labels)
    return topsum

def soft_clip_loss(preds, targs, temp=0.125):
    clip_clip = (targs @ targs.T)/temp
    brain_clip = (preds @ targs.T)/temp
    
    loss1 = -(brain_clip.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
    loss2 = -(brain_clip.T.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
    
    loss = (loss1 + loss2)/2
    return loss

def mixco(voxels, beta=0.15, s_thresh=0.5):
    perm = torch.randperm(voxels.shape[0])
    voxels_shuffle = voxels[perm].to(voxels.device,dtype=voxels.dtype)
    betas = torch.distributions.Beta(beta, beta).sample([voxels.shape[0]]).to(voxels.device,dtype=voxels.dtype)
    select = (torch.rand(voxels.shape[0]) <= s_thresh).to(voxels.device)
    betas_shape = [-1] + [1]*(len(voxels.shape)-1)
    voxels[select] = voxels[select] * betas[select].reshape(*betas_shape) + \
        voxels_shuffle[select] * (1 - betas[select]).reshape(*betas_shape)
    betas[~select] = 1
    return voxels, perm, betas, select

def mixco_clip_target(clip_target, perm, select, betas):
    clip_target_shuffle = clip_target[perm]
    clip_target[select] = clip_target[select] * betas[select].reshape(-1, 1) + \
        clip_target_shuffle[select] * (1 - betas[select]).reshape(-1, 1)
    return clip_target

def mixco_nce(preds, targs, temp=0.1, perm=None, betas=None, select=None, distributed=False, 
              accelerator=None, local_rank=None, bidirectional=True):
    brain_clip = (preds @ targs.T)/temp
    
    if perm is not None and betas is not None and select is not None:
        probs = torch.diag(betas)
        probs[torch.arange(preds.shape[0]).to(preds.device), perm] = 1 - betas

        loss = -(brain_clip.log_softmax(-1) * probs).sum(-1).mean()
        if bidirectional:
            loss2 = -(brain_clip.T.log_softmax(-1) * probs.T).sum(-1).mean()
            loss = (loss + loss2)/2
        return loss
    else:
        loss =  F.cross_entropy(brain_clip, torch.arange(brain_clip.shape[0]).to(brain_clip.device))
        if bidirectional:
            loss2 = F.cross_entropy(brain_clip.T, torch.arange(brain_clip.shape[0]).to(brain_clip.device))
            loss = (loss + loss2)/2
        return loss

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('param counts:\n{:,} total\n{:,} trainable'.format(total, trainable))

def image_grid(imgs, rows, cols):
    w, h = imgs[0].size
    grid = PIL.Image.new('RGB', size=(cols*w, rows*h))
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid
    
def check_loss(loss):
    if loss.isnan().any():
        raise ValueError('NaN loss')
        
def _check_whether_images_are_identical(image1, image2):
    pil_image1 = transforms.ToPILImage()(image1)
    pil_image2 = transforms.ToPILImage()(image2)

    SIMILARITY_THRESHOLD = 90

    image_hash1 = phash(pil_image1, hash_size=16)
    image_hash2 = phash(pil_image2, hash_size=16)

    return (image_hash1 - image_hash2) < SIMILARITY_THRESHOLD

def cosine_anneal(start, end, steps):
    return end + (start - end)/2 * (1 + torch.cos(torch.pi*torch.arange(steps)/(steps-1)))

def select_annotations(annots, random=False):
    """
    There are 5 annotations per image. Select one of them for each image.
    """
    for i, b in enumerate(annots):
        t = ''
        if random:
            # select random non-empty annotation
            while t == '':
                rand = torch.randint(5, (1,1))[0][0]
                t = b[0, rand]
        else:
            # select first non-empty annotation
            for j in range(5):
                if b[0, j] != '':
                    t = b[0, j]
                    break
        if i == 0:
            txt = np.array(t)
        else:
            txt = np.vstack((txt, t))
    txt = txt.flatten()
    return txt

def get_dataloaders(
    batch_size,
    num_devices=None,
    num_workers=None,
    train_url=None,
    val_url=None,
    meta_url=None,
    num_train=None,
    num_val=None,
    cache_dir="/tmp/wds-cache",
    seed=0,
    voxels_key="vert.npy",
    val_batch_size=None,
    local_rank=0,
):
    if local_rank==0: print("Getting dataloaders...")

    metadata = json.load(open(meta_url))
    if num_val is None:
        num_val = 300
    if num_train is None:
        num_train = metadata['total'] - num_val

    if local_rank==0: print('Prepping train and validation dataloaders...')
    
    def my_split_by_node(urls):
        return urls

    global_batch_size = batch_size * num_devices
    num_batches = math.floor(num_train / global_batch_size)
    num_worker_batches = math.floor(num_batches / num_workers)
    
    if local_rank==0: print("\nnum_train",num_train)
    if local_rank==0: print("global_batch_size",global_batch_size)
    if local_rank==0: print("num_batches",num_batches)

    train_data = wds.WebDataset(train_url, resampled=False, cache_dir=cache_dir, nodesplitter=wds.split_by_node)\
        .shuffle(500, initial=500, rng=random.Random(seed))\
        .decode("torch")\
        .rename(images="jpg;png", voxels="vert.npy", latent="clip_emb_hidden.npy")\
        .to_tuple("voxels", "images", "latent")\
        .batched(batch_size, partial=False)\
        .with_epoch(num_worker_batches)
    train_dl = torch.utils.data.DataLoader(train_data, 
                            num_workers=min(num_workers, num_worker_batches),
                            batch_size=None, shuffle=False, persistent_workers=True)
    
    if local_rank==0: print("\nnum_val", num_val)
    if local_rank==0: print("val_batch_size", val_batch_size)

    val_data = wds.WebDataset(val_url, resampled=False, nodesplitter=wds.split_by_node)\
        .decode("torch")\
        .rename(images="jpg;png", voxels="vert.npy", latent="clip_emb_hidden.npy")\
        .to_tuple("voxels", "images", "latent")\
        .batched(val_batch_size, partial=False)
    val_dl = torch.utils.data.DataLoader(val_data, num_workers=1,
                    batch_size=None, shuffle=False, persistent_workers=True)

    return train_dl, val_dl, num_train, num_val

def voxel_select(voxels):
    if voxels.ndim == 2:
        return voxels
    choice = torch.rand(1)
    # random combine
    if choice <= 0.5:
        weights = torch.rand(voxels.shape[0], voxels.shape[1])[:,:,None].to(voxels.device)
        return (weights * voxels).sum(1)/weights.sum(1)
    # mean
    if choice <= 0.8:
        return voxels.mean(1)
    # random select
    randints = torch.randint(0, voxels.shape[1], (voxels.shape[0],))
    return voxels[torch.arange(voxels.shape[0]), randints]

def pearson_correlation(x, y):
    xm = x - x.mean(0, keepdim=True)
    ym = y - y.mean(0, keepdim=True)
    
    r_num = xm.T @ ym  # 39k, 39k
    r_den = ((xm.T**2).sum(1, keepdim=True) @ (ym**2).sum(0, keepdim=True))**0.5
    r_val = r_num / r_den
    
    return r_val

def soft_corr_loss(pred, targ, temp=0.125):
    pred_targ = pearson_correlation(pred, targ)/temp
    targ_targ = pearson_correlation(targ, targ)/temp

    loss1 = -(pred_targ.log_softmax(-1) * targ_targ.softmax(-1)).sum(-1).mean()
    loss2 = -(pred_targ.T.log_softmax(-1) * targ_targ.softmax(-1)).sum(-1).mean()
    
    loss = (loss1 + loss2)/2
    return loss


class DINOLoss(nn.Module):
    def __init__(self,dim,student_temp=0.1,center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, dim))

    def forward(self, pred, targ, temp):
        pred = pred/self.student_temp
        loss = (
            -(
                ((pred - self.center) / self.student_temp).softmax(-1)
                * ((targ - self.center) / temp).log_softmax(-1)
            )
            # .sum(-1)
            .mean()
        )

        self.update_center(targ)
        return loss

    @torch.no_grad()
    def update_center(self, targ):
        batch_center = targ.mean(dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

@torch.no_grad()
def vd_sample_images(
    clip_extractor, brain_net, vd_pipe, diffusion_prior, voxel, img_input, 
    annotations=None,
    num_inference_steps=50,
    clip_guidance_scale=7.5,
    vox_guidance_scale=7.5,
    num_per_sample=4,
    prior_timesteps=None,
    seed=None,
    verbose=True,
    device='cuda',
):

    def null_sync(t, *args, **kwargs):
        return [t]
    
    assert voxel.shape[0] == img_input.shape[0], 'batch dim must be the same for voxels and images'
    n_examples = voxel.shape[0]

    clip_extractor.eval()
    brain_net.eval()
    if diffusion_prior is not None:
        diffusion_prior.eval()

    if seed is not None:
        # set seed
        g_cuda = torch.Generator(device=device)
        g_cuda.manual_seed(seed)

    # for brain guided images (specific to 512 x 512 generation size)
    latents = torch.randn([num_per_sample, 4, 64, 64], device=device, generator=g_cuda)
    
    # use the same latent as the first brain guided image for max similarity
    # clip_latents = torch.randn([1, 4, 64, 64], device=device, generator=g_cuda)
    clip_latents = latents[0].unsqueeze(0).clone()

    grids = []

    for idx in range(n_examples):
        print('sampling for image', idx+1, 'of', n_examples, flush=True)

        img_orig = img_input[[idx]]
        image = clip_extractor.resize_image(img_orig)

        # Original clip embedding: 
        clip_image_emb = clip_extractor.embed_image(image, apply_transforms=False)
        uncond_image = torch.zeros_like(image) + 0.5
        uncond_image = clip_extractor.embed_image(uncond_image, apply_transforms=False)
        # uncond_image = torch.zeros_like(clip_image_emb)
        if annotations is not None:
            # @todo implement versatile's embed text from 
            # https://github.com/huggingface/diffusers/blob/716286f19ddd9eb417113e064b538706884c8e73/src/diffusers/pipelines/versatile_diffusion/pipeline_versatile_diffusion_dual_guided.py#L184
            print('Sampling with CLIP text guidance')
            annots = select_annotations(annotations[[idx]], random=False)
            clip_text_emb = clip_extractor.embed_text(annots)
            # uncond_tokens = ""
            # uncond_text = clip_extractor.embed_text(annots)
            uncond_text = torch.zeros_like(clip_text_emb)
        else:
            clip_text_emb = torch.randn(1, 77, 768).to(clip_image_emb.device)
            uncond_text = torch.zeros_like(clip_text_emb)

        # Encode voxels to CLIP space
        image_embeddings = brain_net(voxel[[idx]].to(device).float())
        if brain_net.use_projector:
            # tuple of mse embeds and contrastive embeds
            image_embeddings = image_embeddings[0]
        
        
        # image_embeddings = nn.functional.normalize(image_embeddings, dim=-1) 
        # image_embeddings *= clip_emb[1].norm()/image_embeddings.norm() # note: this is cheating to equate norm scaling
        if diffusion_prior is not None:
            image_embeddings = diffusion_prior.p_sample_loop(image_embeddings.shape, 
                                                text_cond = dict(text_embed = image_embeddings), 
                                                cond_scale = 1., timesteps = prior_timesteps,
                                                generator=g_cuda
                                                )
        
        # cls token norming
        clip_image_emb = clip_image_emb/(clip_image_emb[:, 0].norm(dim=-1).reshape(-1, 1, 1) + 1e-6)
        image_embeddings = image_embeddings/(image_embeddings[:, 0].norm(dim=-1).reshape(-1, 1, 1) + 1e-6)
        uncond_image = uncond_image/(uncond_image[:, 0].norm(dim=-1).reshape(-1, 1, 1) + 1e-6)

        # duplicate the embedding to serve classifier free guidance
        image_embeddings = image_embeddings.repeat(num_per_sample, 1, 1)
        image_embeddings = torch.cat([uncond_image.repeat(num_per_sample, 1, 1), image_embeddings]).to(device)  # 8,257,768

        # duplicate the embedding to serve classifier free guidance
        clip_image_emb = torch.cat([uncond_image, clip_image_emb]).to(device).float()  # 2,257,768
        if clip_text_emb is not None:
            clip_text_emb = torch.cat([uncond_text, clip_text_emb]).to(device).float()

        # TODO: passing sizes doesn't seem to work, so we're using None for now
        # width, height = 256, 256
        width, height = None, None

        with torch.inference_mode(), torch.autocast(device):
            # [1, 3, 512, 512]
            img_clip = vd_pipe(
                image_embeddings=clip_image_emb,
                prompt_embeddings=clip_text_emb,
                text_to_image_strength = 0.,
                num_inference_steps=num_inference_steps,
                num_images_per_prompt=1,
                guidance_scale=clip_guidance_scale, 
                latents=clip_latents,
                width=width,
                height=height,
                generator=g_cuda,
            )

            # [4, 3, 512, 512]
            imgs_brain = vd_pipe(
                image_embeddings=image_embeddings,
                prompt_embeddings=clip_text_emb.repeat(num_per_sample, 1, 1),
                text_to_image_strength = 0.,
                num_inference_steps=num_inference_steps,
                num_images_per_prompt=num_per_sample,
                guidance_scale=vox_guidance_scale,
                latents=latents,
                width=width,
                height=height,
                generator=g_cuda,
            )

            # print('img_clip.shape', img_clip.shape)
            # print('imgs_brain.shape', imgs_brain.shape)
        
        # resizing for now since passing target sizes into sd_pipe doesn't work
        size = img_orig.shape[-2:]
        img_clip = nn.functional.interpolate(img_clip, size, mode="area", antialias=False)
        imgs_brain = nn.functional.interpolate(imgs_brain, size, mode="area", antialias=False)
        
        imgs_all = torch.cat((img_orig.to(device), img_clip, imgs_brain), 0)
        grid = torch_to_Image(
            make_grid(imgs_all, nrow=2+4, padding=10).detach()
        )
        grids.append(grid)

    return grids, None
