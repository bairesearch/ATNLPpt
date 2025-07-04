"""ATNLPpt_comparison.py

# Author:
Richard Bruce Baxter - Copyright (c) 2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ANNpt_main.py

# Usage:
see ANNpt_main.py

# Description:
ATNLPpt comparison

"""

from ANNpt_globalDefs import *
import torch
import torch.nn.functional as F
from typing import Sequence, Union

@torch.inference_mode()
def unit_similarity_vectors_nd(
    imgs: Union[Sequence[torch.Tensor], torch.Tensor],
    do_center: bool = False,          # subtract per-image mean before flattening
    eps: float = 1e-8,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """
    Args
    ----
    imgs       : - list / tuple of tensors *of identical shape*  
                 - or a single tensor shaped (B, *dims)
                 Any dtype accepted; will be cast to float32.
    do_center  : if True, subtract each image's mean value before similarity.
    eps        : numerical guard for division by 0.
    device     : 'cuda', 'cpu', or torch.device

    Returns
    -------
    sim_vecs   : (B, B) tensor where row i is the unit-L2 similarity vector
                 (cosine similarities) of image i to every other image.
                 ||row_i||_2 == 1.
    """
    device = torch.device(device)

    # --------- stack imgs into (B, *) float32 ---------------------------------
    if isinstance(imgs, torch.Tensor):
        x = imgs
        if x.ndim < 2:
            raise ValueError("Need at least a batch dimension plus data dims.")
    else:
        x = torch.stack(list(imgs), dim=0)
    x = x.to(device, dtype=torch.float32)

    # optional centering (helps when overall intensity shifts matter less)
    if do_center:
        x = x - x.mean(dim=tuple(range(1, x.ndim)), keepdim=True)

    # --------- flatten each image to a vector ---------------------------------
    B = x.shape[0]
    feats = x.reshape(B, -1)                    # (B, D)

    # --------- L2-normalise features (unit vector per image) ------------------
    feats = F.normalize(feats, p=2, dim=1, eps=eps)      # (B, D)

    # --------- cosine similarity matrix ---------------------------------------
    sim = feats @ feats.T                                # (B, B), range [-1 .. 1]

    # --------- convert each row to **unit similarity vector** -----------------
    sim_vecs = F.normalize(sim, p=2, dim=1, eps=eps)     # (B, B),  ||row_i||_2 == 1

    return sim_vecs


'''
# Suppose we have 10 tensors shaped (4, 16, 16) - e.g. 4-channel 1616 images
imgs = torch.rand(10, 4, 16, 16)          # replace with your actual data

sim_vecs = unit_similarity_vectors_nd(imgs, do_center=True, device="cpu")
print(sim_vecs.shape)      # torch.Size([10, 10])
print(sim_vecs[0])         # similarity profile of first image (unit norm)
'''
