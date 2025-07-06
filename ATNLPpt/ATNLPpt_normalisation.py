"""ATNLPpt_normalisation.py

# Author:
Richard Bruce Baxter - Copyright (c) 2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ANNpt_main.py

# Usage:
see ANNpt_main.py

# Description:
ATNLPpt normalisation

# Description
# -----------
# PyTorch re-write of ATNLP normalisation with upgrades 1-6:
#   (4) I/O tensor shapes
#		 in : (B1, C, L1)
#		 out: (B2, C, L2)  where B2 = B1*r	  (train)
#							  or  B1*r*(q-1)	(eval)
#   (5) No Python loops in the crop/resize - a single grid_sample call
#		 keypoint_idx is supplied as int tensor (B2, 2)
#   (6) Uses only PyTorch core - grid_sample provides the 'batch crop + resize'
# 

"""

from __future__ import annotations
from ANNpt_globalDefs import *
import torch
import torch.nn.functional as F
import spacy
import ATNLPpt_keypoints
from typing import List, Dict, Tuple, Literal
NL_MODEL = spacy.load("en_core_web_sm", disable=("ner", "parser", "lemmatizer"))


# ------------------------------------------------------------------------- #
# (4)(5)(6)  batch crop + resize with ONE grid_sample call (no loops)	   #
# ------------------------------------------------------------------------- #
@torch.no_grad()
def normalise_batch(
	seq_tensor: torch.Tensor,		  # (B1, C, L1)
	spacy_pos : torch.Tensor,		  # (B1, L1)
	spacy_offsets : torch.Tensor,		  # (B1, L1, 2)
	last_spacy_token_idx: int,
	mode : keypointModes,
	r : int,
	q : int,
	L2 : int,
	kp_indices_batch: List[List[int]],
	kp_meta_batch: List[List[Dict]],
	align_corners: bool = True,
) -> Tuple[torch.Tensor, List[List[Dict]]]:
	"""
	Main API - returns (B2, C, L2) where B2 depends on mode, plus key-point kp_meta.
	"""

	device = seq_tensor.device
	B1, C, L1 = seq_tensor.shape
	assert len(kp_indices_batch) == B1
	
	if(debugATNLPnormalisation):
		print("normalise_batch():")
		print("seq_tensor.shape = ", seq_tensor.shape)
		print("r = ", r)
		print("q = ", q)
		print("B1 = ", B1)
		print("C = ", C)
		print("L1 = ", L1)
		print("L2 = ", L2)

	# ---------- build key-point tensors for whole batch (loops allowed) ---- #
	all_pairs, all_valid, sample_idx = [], [], []

	for b in range(B1):
		
		# Each samples has a designated 'last spacy token' index (corresponding to the spacy token encapsulating the prediction target bert token or character)
		kp_indices, kp_meta = kp_indices_batch[b].copy(), kp_meta_batch[b].copy()
		ATNLPpt_keypoints.append_keypoints_last_token(last_spacy_token_idx[b], kp_indices, kp_meta)

		# ------------------------------------------------------------- #
		# Convert token-level key-points -> character-level key-points  
		#   currently adds the first delimiter token in the reference set (but not the last delimiter token)
		#   only build pairs from key-points strictly *before* the chosen last-token index
		# ------------------------------------------------------------- #
		character_offsets = spacy_offsets[b]		  # (Ls, 2) start/end
		kp_use = [character_offsets[idx][0] for idx in kp_indices if idx < last_spacy_token_idx[b]]	
		
		pairs, valid = ATNLPpt_keypoints.make_pairs(kp_use, mode, r, q)				 # upgrade 3
		
		if pairs.numel():							  # may be (0,2) in dev
			all_pairs.append(pairs)
			all_valid.append(valid)
			sample_idx.append(torch.full((pairs.size(0),), b, dtype=torch.long))

	if not all_pairs:								  # dev mode w/o pairs
		return torch.empty(0, C, L2)

	pairs   = torch.cat(all_pairs,  dim=0).to(device)		  # (B2, 2)
	valid   = torch.cat(all_valid,  dim=0).to(device)		  # (B2,)
	src_ids = torch.cat(sample_idx, dim=0).to(device)		  # (B2,)
	B2 = pairs.size(0)
	
	# ------------------  single pass crop + resize (no loops) -------------- #
	# 1. build a sampling grid in [-1,1] for grid_sample
	start = pairs[:, 0].unsqueeze(1).float()			# (B2,1)
	end   = pairs[:, 1].unsqueeze(1).float()			# (B2,1)
	lin   = torch.linspace(0, 1, L2, device=device).unsqueeze(0)  # (1,L2)
	real  = start + (end - 1 - start) * lin			 # (B2,L2) original idx
	grid_x = real / (L1 - 1) * 2 - 1
	grid_y = torch.zeros_like(grid_x)				   # dummy height (H=1)
	grid   = torch.stack((grid_y, grid_x), dim=-1)	   # (B2,L2,2)
	grid   = grid.unsqueeze(1)						  # (B2,1,L2,2)   H_out=1

	# 2. gather the relevant rows from the source batch
	src = seq_tensor[src_ids]						   # (B2,C,L1)
	src = src.unsqueeze(2)							  # (B2,C,1,L1)

	# 3. sample
	out = F.grid_sample(src, grid, mode="bilinear", align_corners=align_corners, padding_mode="zeros").squeeze(2)  # (B2,C,L2)

	# 4. zero-out the segments that were invalid (requirement 3 fallback)
	out[~valid] = 0

	if(debugATNLPnormalisation):
		print("B2 = ", B2)
		print("output shape :", tuple(out.shape))
				
	return out
