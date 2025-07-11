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

	last_spacy_token_idx = char_idx_to_spacy_idx(spacy_offsets, last_token_idx)

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
	all_keypointPairsCharIdx, all_keypointPairsValid, all_keypointPairsIndices, sample_idx = [], [], [], []

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
		kp_use = [character_offsets[idx][0].item() for idx in kp_indices if character_offsets[idx][0].item() < last_token_idx]
		ATNLPpt_keypoints.insert_keypoints_last_token(last_token_idx, kp_use)
		keypointPairsCharIdx, keypointPairsValid = ATNLPpt_keypoints.make_pairs(kp_use, mode, r, q)
		
		kp_use_spacy = [idx for idx in kp_indices if idx < last_spacy_token_idx[b]]		#if character_offsets[idx][0].item() < last_token_idx
		ATNLPpt_keypoints.insert_keypoints_last_token(last_spacy_token_idx[b], kp_use_spacy)
		keypointPairsIndices, keypointPairsValid = ATNLPpt_keypoints.make_pairs(kp_use_spacy, mode, r, q)
		#currently use keypointPairsValid from keypointPairsIndices (prevents intraReferenceSetDelimiter eg intraverb token prediction)

		if(debugATNLPkeypoints):
			print("kp_use_spacy = ", kp_use_spacy)
			print("kp_use = ", kp_use)
			print("keypointPairsCharIdx = ", keypointPairsCharIdx)
			print("keypointPairsValid = ", keypointPairsValid)
		
		#if keypointPairsCharIdx.numel():							  # may be (0,2) in dev
		all_keypointPairsCharIdx.append(keypointPairsCharIdx)
		all_keypointPairsValid.append(keypointPairsValid)
		all_keypointPairsIndices.append(keypointPairsIndices)
		sample_idx.append(torch.full((keypointPairsCharIdx.size(0),), b, dtype=torch.long))

	if not all_keypointPairsCharIdx:								  # dev mode w/o keypointPairsCharIdx
		return torch.empty(0, C, L2)

	keypointPairsIndices = torch.cat(all_keypointPairsIndices, dim=0).to(device)		  # (B2, 2)
	keypointPairsCharIdx = torch.cat(all_keypointPairsCharIdx, dim=0).to(device)		  # (B2, 2)
	keypointPairsValid = torch.cat(all_keypointPairsValid, dim=0).to(device)		  # (B2,)
	src_ids = torch.cat(sample_idx, dim=0).to(device)		  # (B2,)
	B2 = keypointPairsCharIdx.size(0)
	
	# ------------------  area-style crop + resize (no loops) --------------- #
	# 1. gather the relevant rows from the source batch
	src = seq_tensor[src_ids]							# (B2,C,L1)

	# 2. build a boolean mask for every crop span [start, end]
	pos	= torch.arange(L1, device=device).view(1,1,L1)		# (1,1,L1)
	start = keypointPairsCharIdx[:, 0].view(B2,1,1)						# (B2,1,1)
	end	= keypointPairsCharIdx[:, 1].view(B2,1,1)						# (B2,1,1)
	mask = (pos >= start) & (pos <= end)					# (B2,1,L1)
	mask = mask.expand(-1, C, -1)						# (B2,C,L1)

	# 3. zero-out everything outside the span
	seg = src.masked_fill(~mask, 0.0)					# (B2,C,L1)

	# 4. area-style down-sampling: max over L2 equal-width bins
	normalisedSnapshots = F.adaptive_avg_pool1d(seg, L2)				# (B2,C,L2)

	# 5. zero-out the snapshots that were marked invalid
	normalisedSnapshots[~keypointPairsValid] = 0

	S = B2 // B1
	normalisedSnapshots = normalisedSnapshots.reshape(B1, S, C, L2)	# (B1,S,C,L2)
		
	if(debugATNLPnormalisation):
		print("B2 = ", B2)
		print("normalisedSnapshots shape :", tuple(normalisedSnapshots.shape))
		
	keypointPairsIndices = keypointPairsIndices.reshape(B1, S, 2)	#(B1, S, 2)
	keypointPairsValid = keypointPairsValid.reshape(B1, S)	#(B1,S)
	
	return normalisedSnapshots, keypointPairsValid, keypointPairsIndices


def char_idx_to_spacy_idx(
	spacy_offsets: torch.Tensor,   # (B, Ls, 2)  [char_start, char_end)
	last_token_idx: torch.Tensor   # (B,) or scalar  character index
) -> torch.Tensor:				 # -> (B,)  token index within 0..Ls-1

	"""
	Parameters
	----------
	spacy_offsets  : (B, Ls, 2) long | int64
		Offsets produced by spaCy's `Token.idx` and `Token.idx + len(tok) - 1`.
		`spacy_offsets[b, t, 0]` \u2264 `spacy_offsets[b, t, 1]`.
	last_token_idx : int  |  shape (B,) tensor
		Character position that marks the 'current' / 'last' token.
		If a scalar is supplied, the same char-index is used for every batch item.

	Returns
	-------
	last_spacy_token_idx : (B,) long
		For each batch sample, the token index *t* whose span contains
		`last_token_idx[b]`, i.e.
		`spacy_offsets[b, t, 0] \u2264 last_token_idx[b] \u2264 spacy_offsets[b, t, 1]`.
		If no span matches (shouldn\u2019t happen in valid data) the result is 0.
	"""
	B, Ls, _ = spacy_offsets.shape
	device = spacy_offsets.device
	dtype = spacy_offsets.dtype

	# --- broadcast last_token_idx to shape (B,) ---------------------------- #
	last_char = torch.full((B,), last_token_idx, dtype=dtype, device=device)

	# --- vectorised containment test  (B, Ls) ------------------------------ #
	starts, ends = spacy_offsets[..., 0], spacy_offsets[..., 1]
	mask = (last_char[:, None] >= starts) & (last_char[:, None] <= ends)

	# --- convert boolean mask \u2192 index (first True along Ls) --------------- #
	# `mask` is guaranteed to have at least one True per row in valid data.
	last_spacy_token_idx = torch.argmax(mask.to(torch.int8), dim=1)

	# optional: assert validity during development
	# assert mask.any(dim=1).all(), "some last_token_idx not inside any token span"

	'''
	print("char_idx_to_spacy_idx():")
	print("last_token_idx = ", last_token_idx)
	print("spacy_offsets = ", spacy_offsets)
	print("last_spacy_token_idx = ", last_spacy_token_idx)
	'''

	return last_spacy_token_idx	# (B,) long

