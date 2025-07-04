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
#   (1) last token is always a key-point -> see _append_keypoints_last_token()
#   (2) key-points are obtained with spaCy; we store POS, start & end char
#   (3) generate key-point pairs:
#		 a) dev  : every ordered pair
#		 b) train: last r adjacent pairs
#		 c) eval : last r starts, span size 2->q		  -> see _make_pairs()
#	   Missing pairs are zero-filled as required
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
from typing import List, Dict, Tuple, Literal
NL_MODEL = spacy.load("en_core_web_sm", disable=("ner", "parser", "lemmatizer"))


def build_keypoints(
	spacy_pos : torch.Tensor,		  # (B1, L1)
	spacy_offsets : torch.Tensor,		  # (B1, L1, 2)
) -> Tuple[List[List[int]], List[List[Dict]]]:

	batchSize = spacy_pos.shape[0]
	kp_indices_batch = []
	kp_meta_batch = []
	
	for b in range(batchSize):
		kp_indices, kp_meta = _detect_keypoints(spacy_pos[b], spacy_offsets[b])
		kp_indices_batch.append(kp_indices)
		kp_meta_batch.append(kp_meta)

	return kp_indices_batch, kp_meta_batch

# ------------------------------------------------------------------------- #
# (1) + (2)  spaCy-based key-point detector that always includes last token #
# ------------------------------------------------------------------------- #
def _detect_keypoints(spacy_pos: torch.Tensor, spacy_offsets: torch.Tensor) -> Tuple[List[int], List[Dict]]:
	"""
	Parameters
	----------
	spacy_pos : torch.Tensor,		  # (L1)
	spacy_offsets : torch.Tensor,		  # (L1, 2)
	
	Return
	------
	kp_indices : token indices that are key-points
	kp_meta	   : [{token_idx, pos, char_start, char_end}, ...]
	"""
	kp_indices, kp_meta = [], []

	L1 = spacy_pos.shape[0]
	for i in range(L1):
		is_kp = spacy_pos[i] in referenceSetPosDelimiters or i == 0
		if is_kp:
			kp_indices.append(i)
		kp_meta.append({
			"token_idx": i,
			"pos": spacy_pos[i],
			"char_start": spacy_offsets[i][0],	#tok.idx,
			"char_end": spacy_offsets[i][1]	#tok.idx + len(tok)
		})

	return kp_indices, kp_meta


def _append_keypoints_last_token(
	last_token_idx: int,
	L1: int,
	kp_indices: List[int],
	kp_meta: List[Dict] 
):
	"""
	Parameters
	----------
	last_token_idx: int			#last token index at which to perform keypoint detection
	L1: int
	kp_indices : token indices that are key-points
	kp_meta	   : [{token_idx, pos, char_start, char_end}, ...]
	"""
	
	last_idx = min(last_token_idx, L1 - 1)   # clamp for safety
	if kp_indices[-1] != last_idx:
		kp_indices.append(last_idx)
		kp_meta[last_idx]["is_last_keypoint"] = True



# ------------------------------------------------------------------------- #
# (4)(5)(6)  batch crop + resize with ONE grid_sample call (no loops)	   #
# ------------------------------------------------------------------------- #
@torch.no_grad()
def normalise_batch(
	last_token_idx: int,
	seq_tensor: torch.Tensor,		  # (B1, C, L1)
	spacy_pos : torch.Tensor,		  # (B1, L1)
	spacy_offsets : torch.Tensor,		  # (B1, L1, 2)
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
		
		# All samples share the same designated 'last token' index.
		kp_indices, kp_meta = kp_indices_batch[b].copy(), kp_meta_batch[b].copy()
		_append_keypoints_last_token(last_token_idx, L1, kp_indices, kp_meta)

		#only build pairs from key-points strictly *before* the chosen last-token index
		kp_use = [idx for idx in kp_indices if idx < last_token_idx]

		pairs, valid = _make_pairs(kp_use, mode, r, q)				 # upgrade 3
		
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

# ------------------------------------------------------------------------- #
# (3)  build key-point pairs for dev | train | eval modes				   #
# ------------------------------------------------------------------------- #
def _make_pairs(kp: List[int], mode: keypointModes, r: int, q: int) -> Tuple[torch.Tensor, torch.Tensor]:
	"""
	Parameters
	----------
	kp   : sorted list of key-point token indices
	mode : "allKeypointCombinations" | "firstKeypointConsecutivePairs" | "firstKeypointPairs"
	r/q  : user parameters (see requirement 3)

	Return
	------
	pairs	: (N, 2)  long tensor of [i, j] with i<j; invalid -> 0,0
	valid_ms : (N,)	bool tensor, True where pair is valid
	"""
	if len(kp) < 2:
		# zero-fill for the expected number of rows so caller can reshape
		if mode == "firstKeypointConsecutivePairs":
			N = r
		elif mode == "firstKeypointPairs":
			N = r * (q - 1)
		elif mode == "allKeypointCombinations":		 # we let it be empty
			N = 0
		return torch.zeros(N, 2, dtype=torch.long), torch.zeros(N, dtype=torch.bool)

	# ------------  a) every ordered permutation -------------------- #
	if mode == "allKeypointCombinations":
		out = [(i, j) for idx_i, i in enumerate(kp) for j in kp[idx_i + 1:]]
		pairs = torch.as_tensor(out, dtype=torch.long)
		valid = torch.ones(len(out), dtype=torch.bool)
		return pairs, valid

	# ------------  b) last r adjacent pairs ---------------------- #
	if mode == "firstKeypointConsecutivePairs":
		adj = list(zip(kp[:-1], kp[1:]))			   # consecutive pairs
		adj = adj[-r:]								 # last r pairs
		valid_n = len(adj)
		# zero-padding to fixed length r
		pairs = torch.zeros(r, 2, dtype=torch.long)
		if valid_n:
			pairs[:valid_n] = torch.as_tensor(adj, dtype=torch.long)
		valid = torch.zeros(r, dtype=torch.bool)
		valid[:valid_n] = True
		return pairs, valid

	# ------------  c) last r starts, spans 2 -> q ------------------ #
	if mode == "firstKeypointPairs":
		out = []
		starts = kp[-r:]								# last r starting kps
		for s_idx, s in enumerate(starts):
			# pick up to q-1 subsequent key-points
			slice_end = len(kp) - (r - 1 - s_idx)	   # respect ordering
			next_kps = kp[s_idx + len(kp) - r + 1 : slice_end][:q - 1]
			if not next_kps:							# none available
				out.extend([(0, 0)] * (q - 1))
			else:
				# pad to q-1 so output length is constant
				for t in range(q - 1):
					if t < len(next_kps):
						out.append((s, next_kps[t]))
					else:
						out.append((0, 0))
		pairs = torch.as_tensor(out, dtype=torch.long)		  # (r*(q-1), 2)
		valid = ~(pairs[:, 0] == pairs[:, 1])
		return pairs, valid

	raise ValueError(f"Unknown mode {mode}")




