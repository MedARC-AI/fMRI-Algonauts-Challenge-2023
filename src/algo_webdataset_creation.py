import time 
import numpy as np
import matplotlib.pyplot as plt
import cv2
import h5py
import os

from tqdm import tqdm
from PIL import Image
from sklearn.preprocessing import StandardScaler

import webdataset as wds
import sys

from nsd_access import NSDAccess
nsda = NSDAccess('/fsx/proj-medarc/fmri/natural-scenes-dataset/s3')

import nibabel as nib

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device:",device)

sub=int(sys.argv[1])
preset_cnt=int(sys.argv[2])

subject=f'subj0{sub+1}'
subj=subject
print("subject",subject)

print("preset_cnt",preset_cnt)

# CLIP
import clip
from torchvision import transforms
from models import *
clip_extractor = Clipper("ViT-L/14", device=device, hidden_state=True, norm_embs=False)
clip_extractor_last = Clipper("ViT-L/14", device=device, hidden_state=False, norm_embs=False)

# openclip_extractor = OpenClipper('ViT-H-14', device=device, hidden_state=False, norm_embs=False)
# openclip_extractor_last = OpenClipper('ViT-H-14', device=device, hidden_state=True, norm_embs=False)

# ImageBind
import sys
sys.path.insert(0, "/fsx/proj-medarc/fmri/ImageBind")
sys.path.insert(0, "/fsx/proj-medarc/fmri/ImageBind/models")
import data
import imagebind_model
from imagebind_model import ModalityType

imagebind = imagebind_model.imagebind_huge(pretrained=True)
imagebind.eval().requires_grad_(False)
imagebind.to(device)

imagebind_transform = transforms.Compose(
    [transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711))])

import copy
imagebind_hidden = copy.deepcopy(imagebind)
imagebind_hidden.modality_heads.vision = nn.Identity()
imagebind_hidden.modality_postprocessors.vision = nn.Identity()
imagebind_hidden.eval().requires_grad_(False)
imagebind_hidden.to(device)

# FCN segmentation model
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
fcn_weights = FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
fcn_model = create_feature_extractor(fcn_resnet50(weights=fcn_weights), return_nodes=["backbone.maxpool"]).to(device)
fcn_model.eval().requires_grad_(False)
pass

# NOTE: you will want to parallelize this
print(time.strftime("\nCurrent time: %H:%M:%S", time.localtime())) 
total_cnt = 100

lh_file = f"/fsx/proj-medarc/fmri/natural-scenes-dataset/algonauts_data/dataset/{subject}/roi_masks/lh.all-vertices_fsaverage_space.npy"
lh_mask = np.load(lh_file)
rh_file = f"/fsx/proj-medarc/fmri/natural-scenes-dataset/algonauts_data/dataset/{subject}/roi_masks/rh.all-vertices_fsaverage_space.npy"
rh_mask = np.load(rh_file)
mask = np.hstack((lh_mask,rh_mask))
mask = np.asarray(mask,bool)

# load coco 73k indices
indices_path = "/fsx/proj-medarc/fmri/natural-scenes-dataset/COCO_73k_subj_indices.hdf5"
hdf5_file = h5py.File(indices_path, "r")
indices = hdf5_file[f"{subj}"][:]

# load orig images
f = h5py.File('/fsx/proj-medarc/fmri/natural-scenes-dataset/s3/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5', 'r')
images = f['imgBrick']

# load shared 73k indices
import scipy.io as sio
nsd_design = sio.loadmat('/fsx/proj-medarc/fmri/natural-scenes-dataset/s3/nsddata/experiments/nsd/nsd_expdesign.mat')
shared_idx = nsd_design['sharedix'][0]-1
order = nsd_design['subjectim'][sub]-1

# for each unique image, get the betas and average
all_betas = None
cnt = 0; abs_cnt = 0;
for uniq_idx in tqdm(np.unique(indices)):
    if abs_cnt==preset_cnt:
        trial_numbers = np.where(indices==uniq_idx)[0]
        for itrial, trial in enumerate(trial_numbers):
            sess = int(np.floor(trial / 750))
            sess_trial = trial - (sess * 750)
            sess += 1

            betas = nsda.read_betas(subject=subject, 
                    session_index=int(sess), 
                    trial_index=[int(sess_trial)], # empty list as index means get all for this session
                    data_type='betas_fithrf_GLMdenoise_RR',
                    data_format='fsaverage') 

            # load mean + scale to get zero mean and unit variance depending on the given session
            betas_nsd_mean = np.load(f"/fsx/proj-medarc/fmri/natural-scenes-dataset/challenge_scalars/challenge_mean_sess{sess}_{subj}.npy")
            betas_nsd_std = np.load(f"/fsx/proj-medarc/fmri/natural-scenes-dataset/challenge_scalars/challenge_std_sess{sess}_{subj}.npy")

            betas_nsd = betas[mask]
            shape = betas_nsd.shape
            betas_nsd = betas_nsd.reshape(-1,betas_nsd.shape[-2]*betas_nsd.shape[-1])  
            betas_nsd = (betas_nsd - betas_nsd_mean) / betas_nsd_std
            betas_nsd = betas_nsd.astype('float16') # (1, 15724)

            if itrial==0:
                same_betas_nsd = betas_nsd
            else:
                same_betas_nsd = np.concatenate((same_betas_nsd,betas_nsd),axis=0)

        if same_betas_nsd.shape[0] == 1:
            num_unique = 1
            same_betas_nsd = np.concatenate((same_betas_nsd, same_betas_nsd, same_betas_nsd), axis=0)
        elif same_betas_nsd.shape[0] == 2:
            num_unique = 2
            same_betas_nsd = np.concatenate((same_betas_nsd, np.mean(same_betas_nsd,axis=0)[None]), axis=0)
        else:
            num_unique = 3

        if all_betas is None:
            all_betas = same_betas_nsd[None]
            ordering = uniq_idx
            ordered_imgs = trial_numbers[0]
            num_uniques = num_unique
        else:
            all_betas = np.concatenate((all_betas,same_betas_nsd[None]),axis=0)
            ordering = np.append(ordering,uniq_idx)
            ordered_imgs = np.append(ordered_imgs, trial_numbers[0])
            num_uniques = np.append(num_uniques, num_unique)

    cnt += 1
    if cnt==total_cnt:
        if abs_cnt==preset_cnt: 
            sink = wds.TarWriter(f"/fsx/proj-medarc/fmri/natural-scenes-dataset/algonauts_data/wds/{subj}_{abs_cnt}.tar")
            for i,idx in enumerate(range(total_cnt*preset_cnt,total_cnt*preset_cnt+total_cnt)):
                image = Image.fromarray(images[ordering[i]].astype('uint8').clip(0,255))
                with torch.no_grad():
                    cur_imgs_tensor = torch.Tensor(images[ordering[i]]).permute(2,0,1).unsqueeze(0).to(device) / 255
                    ib_inputs = {ModalityType.VISION: imagebind_transform(cur_imgs_tensor)}
                    sink.write({
                        "__key__": "sample%09d" % idx,
                        "num_uniques.npy": np.array([num_uniques[i]]),
                        "coco73k.npy": np.array([ordering[i]]),
                        "vert.npy": all_betas[i],
                        "trial.npy": np.array([ordered_imgs[i]]),
                        "jpg": image,
                        "clip_emb_hidden.npy": clip_extractor.embed_image(cur_imgs_tensor).detach().cpu().numpy(),
                        "clip_emb_final.npy": clip_extractor_last.embed_image(cur_imgs_tensor).detach().cpu().numpy(),
                        "imagebind_hidden.npy": imagebind_hidden(ib_inputs)['vision'].detach().cpu().numpy(),
                        "imagebind_final.npy": imagebind(ib_inputs)['vision'].detach().cpu().numpy(),
                        "fcn_maxpool.npy": fcn_model(cur_imgs_tensor)['backbone.maxpool'].detach().cpu().numpy(),
                    })
            sink.close()

        all_betas = None
        cnt = 0
        abs_cnt += 1

if abs_cnt==preset_cnt: # for the last batch, won't be equal to exactly total_cnt probably
    sink = wds.TarWriter(f"/fsx/proj-medarc/fmri/natural-scenes-dataset/algonauts_data/wds/{subj}_{abs_cnt}.tar")
    for i,idx in enumerate(range(total_cnt*preset_cnt,total_cnt*preset_cnt+total_cnt)):
        image = Image.fromarray(images[ordering[i]].astype('uint8').clip(0,255))
        with torch.no_grad():
            cur_imgs_tensor = torch.Tensor(images[ordering[i]]).permute(2,0,1).unsqueeze(0).to(device) / 255
            ib_inputs = {ModalityType.VISION: imagebind_transform(cur_imgs_tensor)}
            sink.write({
                "__key__": "sample%09d" % idx,
                "num_uniques.npy": np.array([num_uniques[i]]),
                "coco73k.npy": np.array([ordering[i]]),
                "vert.npy": all_betas[i],
                "trial.npy": np.array([ordered_imgs[i]]),
                "jpg": image,
                "clip_emb_hidden.npy": clip_extractor.embed_image(cur_imgs_tensor).detach().cpu().numpy(),
                "clip_emb_final.npy": clip_extractor_last.embed_image(cur_imgs_tensor).detach().cpu().numpy(),
                "imagebind_hidden.npy": imagebind_hidden(ib_inputs)['vision'].detach().cpu().numpy(),
                "imagebind_final.npy": imagebind(ib_inputs)['vision'].detach().cpu().numpy(),
                "fcn_maxpool.npy": fcn_model(cur_imgs_tensor)['backbone.maxpool'].detach().cpu().numpy(),
            })
    sink.close()
    
print(time.strftime("\nCurrent time: %H:%M:%S", time.localtime())) 