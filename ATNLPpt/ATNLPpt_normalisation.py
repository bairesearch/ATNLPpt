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
	last_token_idx: int,
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
		#ATNLPpt_keypoints.insert_keypoints_last_token(last_spacy_token_idx[b], kp_indices, kp_meta)

		# ------------------------------------------------------------- #
		# Convert token-level key-points -> character-level key-points  
		#   currently adds the first delimiter token in the reference set (but not the last delimiter token)
		#   only build pairs from key-points strictly *before* the chosen last-token index
		# ------------------------------------------------------------- #
		character_offsets = spacy_offsets[b]		  # (Ls, 2) start/end
		kp_use_spacy = [idx for idx in kp_indices if character_offsets[idx][0].item() < last_token_idx]	
		kp_use = [character_offsets[idx][0].item() for idx in kp_indices if character_offsets[idx][0].item() < last_token_idx]	
		
		ATNLPpt_keypoints.insert_keypoints_last_token(last_token_idx, kp_use)
		
		pairs, valid = ATNLPpt_keypoints.make_pairs(kp_use, mode, r, q)
		
		if(debugATNLPkeypoints):
			print("kp_use_spacy = ", kp_use_spacy)
			print("kp_use = ", kp_use)
			print("pairs = ", pairs)
			print("valid = ", valid)
		
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
	
	# ------------------  area-style crop + resize (no loops) --------------- #
	# 1. gather the relevant rows from the source batch
	src = seq_tensor[src_ids]							# (B2,C,L1)

	# 2. build a boolean mask for every crop span [start, end]
	pos	= torch.arange(L1, device=device).view(1,1,L1)		# (1,1,L1)
	start = pairs[:, 0].view(B2,1,1)						# (B2,1,1)
	end	= pairs[:, 1].view(B2,1,1)						# (B2,1,1)
	mask = (pos >= start) & (pos <= end)					# (B2,1,L1)
	mask = mask.expand(-1, C, -1)						# (B2,C,L1)

	# 3. zero-out everything outside the span
	seg = src.masked_fill(~mask, 0.0)					# (B2,C,L1)

	# 4. area-style down-sampling: max over L2 equal-width bins
	out = F.adaptive_avg_pool1d(seg, L2)				# (B2,C,L2)

	# 5. zero-out the snapshots that were marked invalid
	out[~valid] = 0

	if(debugATNLPnormalisation):
		print("B2 = ", B2)
		print("output shape :", tuple(out.shape))
				
	return out
