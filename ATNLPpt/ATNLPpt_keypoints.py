"""ATNLPpt_keypoints.py

# Author:
Richard Bruce Baxter - Copyright (c) 2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ANNpt_main.py

# Usage:
see ANNpt_main.py

# Description:
ATNLPpt keypoints

# Description
# -----------
# PyTorch re-write of ATNLP normalisation with upgrades 1-6:
#   (1) last token is always a key-point -> see _append_keypoints_last_token()
#   (2) key-points are obtained with spaCy; we store POS, start & end char
#   (3) generate key-point pairs:
#		 a) dev  : every ordered pair
#		 b) train: last R adjacent pairs
#		 c) eval : last R starts, span size 2->Q		  -> see _make_pairs()
#	   Missing pairs are zero-filled as required

"""

from ANNpt_globalDefs import *
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Literal
import ATNLPpt_pos


def build_keypoints(
	l : int,
	spacy_input_id : torch.Tensor,		  # (B1, L1)
	spacy_pos : torch.Tensor,		  # (B1, L1)
	spacy_tag : torch.Tensor,		  # (B1, L1)
	spacy_text : torch.Tensor,		  # (B1, L1)
	spacy_offsets : torch.Tensor,		  # (B1, L1, 2)
) -> Tuple[List[List[int]], List[List[Dict]]]:

	batchSize = spacy_tag.shape[0]
	kp_indices_batch = []
	kp_meta_batch = []
	kp_prev_level_used_batch = []
	for b in range(batchSize):
		kp_indices, kp_meta, kp_prev_level_used = detect_keypoints(l, spacy_input_id[b], spacy_pos[b], spacy_tag[b], spacy_text[b], spacy_offsets[b])
		kp_indices.reverse()
		kp_meta.reverse()
		kp_indices_batch.append(kp_indices)
		kp_meta_batch.append(kp_meta)
		kp_prev_level_used_batch.append(kp_prev_level_used)

	return kp_indices_batch, kp_meta_batch, kp_prev_level_used_batch

# ------------------------------------------------------------------------- #
# (1) + (2)  spaCy-based key-point detector that always includes last token #
# ------------------------------------------------------------------------- #
def detect_keypoints(	
	l : int,
	spacy_input_id : torch.Tensor,		  # (L1)
	spacy_pos : torch.Tensor,		  # (L1)
	spacy_tag : torch.Tensor,		  # (L1)
	spacy_text : torch.Tensor,		  # (L1)
	spacy_offsets : torch.Tensor		  # (L1, 2)
) -> Tuple[List[int], List[Dict]]:
	"""
	Parameters
	----------
	spacy_input_id : torch.Tensor,		  # (L1)
	spacy_pos : torch.Tensor,		  # (L1)
	spacy_tag : torch.Tensor,		  # (L1)
	spacy_text : torch.Tensor,		  # (L1)
	spacy_offsets : torch.Tensor,		  # (L1, 2)
	
	Return
	------
	kp_indices : token indices that are key-points
	kp_meta	   : [{token_idx, spacy_pos, spacy_tag, spacy_input_id, char_start, char_end}, ...]
	kp_prev_level_used: list of boolean lists of length [number of keypoints in previous level] - is keypoint of previous level also used in current level?
	"""
	kp_indices, kp_meta, kp_prev_level_used = [], [], []

	L1 = spacy_tag.shape[0]
	for i in range(L1):
		spacyInt = spacy_tag[i].item()
		spacyText = spacy_text[i]
		#print("i = ", i, ", spacyInt = ", spacyInt)
		
		is_kp = False
		if i == 0:
			is_kp = True	 #always treat first token in sequence as a keypoint
		else:
			for l2 in range(ATNLPmultiLevels):
				if spacyIntIsKeypoint(l2, spacyInt, spacyText):
					is_kp = True
		if is_kp:
			#print("is_kp")
			kp_indices.append(i)
		
		if(ATNLPuseMultiLevelTokenPrediction):
			if(l == 0):
				if is_kp:
					kp_prev_level_used.append(True)	#not used
			else:
				is_kp_prev = False
				for l2 in range(l-1, ATNLPmultiLevels):
					if spacyIntIsKeypoint(l2, spacyInt, spacyText):
						is_kp_prev = True
				if is_kp_prev:
					is_kp_curr = False
					for l2 in range(l, ATNLPmultiLevels):
						if spacyIntIsKeypoint(l2, spacyInt, spacyText):
							is_kp_curr = True
					if(is_kp_curr):
						kp_prev_level_used.append(True)
					else:
						kp_prev_level_used.append(False)
			if(debugATNLPkeypoints):
				print("create kp_prev_level_used = ", kp_prev_level_used)
			
		kp_meta.append({
			"token_idx": i,
			"spacy_pos": spacy_pos[i],
			"spacy_tag": spacy_tag[i],
			"spacy_input_id": spacy_input_id[i],
			"char_start": spacy_offsets[i][0],	#tok.idx,
			"char_end": spacy_offsets[i][1],	#tok.idx + len(tok)
		})

	return kp_indices, kp_meta, kp_prev_level_used

def spacyIntIsKeypoint(l, spacyInt, spacyText):
	result = False
	if spacyInt in ATNLPpt_pos.referenceSetPosDelimitersTagId[l]:
		result = True
	if spacyText in ATNLPpt_pos.referenceSetPosDelimitersText[l]:
		result = True
	return result

def generate_keypoint_pairs(
	B1 : int,
	R : int,
	Q : int,
	mode : keypointModes,
	device,
	spacy_offsets : torch.Tensor,		  # (B1, L1, 2)
	last_token_idx: int,
	kp_indices_batch: List[List[int]],
	kp_meta_batch: List[List[Dict]],
):
	assert len(kp_indices_batch) == B1

	last_spacy_token_idx = char_idx_to_spacy_idx(spacy_offsets, last_token_idx)

	# ---------- build key-point tensors for whole batch (loops allowed) ---- #
	all_keypointPairsCharIdx, all_keypointPairsValid, all_keypointPairsIndices, sample_idx = [], [], [], []
	
	for b in range(B1):
		
		# Each samples has a designated 'last spacy token' index (corresponding to the spacy token encapsulating the prediction target bert token or character)
		kp_indices, kp_meta = kp_indices_batch[b].copy(), kp_meta_batch[b].copy()
		#insert_keypoints_last_token(last_spacy_token_idx[b], kp_indices, kp_meta)

		# ------------------------------------------------------------- #
		# Convert token-level key-points -> character-level key-points  
		#   currently adds the first delimiter token in the reference set (but not the last delimiter token)
		#   only build pairs from key-points strictly *before* the chosen last-token index
		# ------------------------------------------------------------- #
		character_offsets = spacy_offsets[b]		  # (Ls, 2) start/end
		kp_use = [character_offsets[idx][0].item() for idx in kp_indices if character_offsets[idx][0].item() < last_token_idx]
		if(useSlidingWindow):
			insert_keypoints_last_token(last_token_idx, kp_use)
		keypointPairsCharIdx, keypointPairsValid = make_pairs(kp_use, mode, R, Q)
		
		kp_use_spacy = [idx for idx in kp_indices if idx < last_spacy_token_idx[b]]		#if character_offsets[idx][0].item() < last_token_idx
		if(useSlidingWindow):
			insert_keypoints_last_token(last_spacy_token_idx[b], kp_use_spacy)
		keypointPairsIndices, keypointPairsValid = make_pairs(kp_use_spacy, mode, R, Q)
		#currently use keypointPairsValid from keypointPairsIndices (prevents intraReferenceSetDelimiter eg intraverb token prediction)
			
		if(debugATNLPkeypoints):
			print("kp_use_spacy = ", kp_use_spacy)
			print("kp_use = ", kp_use)
			print("keypointPairsIndices = ", keypointPairsIndices)
			print("keypointPairsCharIdx = ", keypointPairsCharIdx)
			print("keypointPairsValid = ", keypointPairsValid)
		
		#if keypointPairsCharIdx.numel():							  # may be (0,2) in dev
		all_keypointPairsCharIdx.append(keypointPairsCharIdx)
		all_keypointPairsValid.append(keypointPairsValid)
		all_keypointPairsIndices.append(keypointPairsIndices)
		sample_idx.append(torch.full((keypointPairsCharIdx.size(0),), b, dtype=torch.long))

	keypointPairsIndices = torch.cat(all_keypointPairsIndices, dim=0).to(device)		  # (B2, 2)
	keypointPairsCharIdx = torch.cat(all_keypointPairsCharIdx, dim=0).to(device)		  # (B2, 2)
	keypointPairsValid = torch.cat(all_keypointPairsValid, dim=0).to(device)		  # (B2,)
	src_ids = torch.cat(sample_idx, dim=0).to(device)		  # (B2)

	#print("src_ids.shape = ", src_ids.shape)
	
	if all_keypointPairsCharIdx:								  # dev mode w/o keypointPairsCharIdx
		foundKeypointPairs = True
	else:
		foundKeypointPairs = False
		
	return foundKeypointPairs, keypointPairsIndices, keypointPairsCharIdx, keypointPairsValid, src_ids

def insert_keypoints_last_token(
	last_token_idx_sample: int,
	kp_use: List[int],
):
	"""
	Parameters
	----------
	last_token_idx_sample: int			#last spacy token index at which to perform keypoint detection]
	"""
	kp_use.insert(0, last_token_idx_sample)


def generate_keypoint_pairs_from_prev_level(
	l : int,
	mode : keypointModes,
	device,
	normalisedSequencePrevLevel : torch.Tensor,	#(B1*Q, R*L2, C)
	kp_prev_level_used_batch : List[bool]	#(B1prev, R)
):
	Rprev, Qprev, L2prev, Rcurr, Qcurr, L2curr = (R[l-1], Q[l-1], L2[l-1], R[l], Q[l], L2[l])
	
	B2 = normalisedSequencePrevLevel.shape[0]
	#assert normalisedSequencePrevLevel.shape[1] == Rprev
	
	# ---------- build key-point tensors for whole batch (loops allowed) ---- #
	all_keypointPairsCharIdx, all_keypointPairsValid, all_keypointPairsIndices, sample_idx = [], [], [], []
	#print("L2prev = ", L2prev)
	
	for b in range(B2):
		kp_prev_level_used = torch.tensor(kp_prev_level_used_batch[b], device=device)
		RprevReal = len(kp_prev_level_used_batch[b])
		aran = torch.arange(0, RprevReal, device=device, dtype=torch.long)
		kp_use = kp_prev_level_used.int()*aran
		if(not ATNLPuseSequenceLevelPredictionInput):
			kp_use = kp_use*L2prev	#generate keypoint indices by simply multiplying by number of tokens per normalised snapshot
		if(len(kp_prev_level_used) > 0):
			kp_use = kp_use[kp_prev_level_used]
			kp_use = kp_use.tolist()
		else:
			kp_use = []
		kp_use.reverse()	#make_pairs() expects kp_use to be reverse ordered as per build_keypoints()
		#print("kp_use = ", kp_use)
		
		keypointPairsCharIdx, keypointPairsValid = make_pairs(kp_use, mode, Rcurr, Qcurr)
		
		all_keypointPairsCharIdx.append(keypointPairsCharIdx)
		all_keypointPairsValid.append(keypointPairsValid)
		#all_keypointPairsIndices.append(keypointPairsIndices)
		sample_idx.append(torch.full((keypointPairsCharIdx.size(0),), b, dtype=torch.long))

	keypointPairsIndices = None	#keypointPairsIndices = torch.cat(all_keypointPairsIndices, dim=0).to(device)		  # (B2, 2)
	keypointPairsCharIdx = torch.cat(all_keypointPairsCharIdx, dim=0).to(device)		  # (B2, 2)
	keypointPairsValid = torch.cat(all_keypointPairsValid, dim=0).to(device)		  # (B2,)
	src_ids = torch.cat(sample_idx, dim=0).to(device)		  # (B2)
	
	if all_keypointPairsCharIdx:								  # dev mode w/o keypointPairsCharIdx
		foundKeypointPairs = True
	else:
		foundKeypointPairs = False
	
	if(debugATNLPkeypoints):
		print("foundKeypointPairs = ", foundKeypointPairs)
		print("keypointPairsCharIdx = ", keypointPairsCharIdx)
		print("keypointPairsValid = ", keypointPairsValid)
		print("src_ids = ", src_ids)

	return foundKeypointPairs, keypointPairsIndices, keypointPairsCharIdx, keypointPairsValid, src_ids
	
	

# ------------------------------------------------------------------------- #
# (3)  build key-point pairs for dev | train | eval modes				   #
# ------------------------------------------------------------------------- #
def make_pairs(kp: List[int], mode: keypointModes, R: int, Q: int) -> Tuple[torch.Tensor, torch.Tensor]:
	"""
	Parameters
	----------
	kp   : sorted list of key-point token indices
	mode : "allKeypointCombinations" | "firstKeypointConsecutivePairs" | "firstKeypointPairs"
	R/Q  : user parameters (see requirement 3)

	Return
	------
	pairs	: (N, 2)  long tensor of [i, j] with i<j; invalid -> 0,0
	valid_ms : (N,)	bool tensor, True where pair is valid
	"""
		
	if len(kp) < 2:
		# zero-fill for the expected number of rows so caller can reshape
		if mode == "firstKeypointConsecutivePairs":
			N = R
		elif mode == "firstKeypointPairs":
			N = R * Q
		elif mode == "allKeypointCombinations":		 # we let it be empty
			N = 0
		pairs = torch.zeros(N, 2, dtype=torch.long)
		valid = torch.zeros(N, dtype=torch.bool)
	else:
		# ------------  a) every ordered permutation -------------------- #
		if mode == "allKeypointCombinations":
			out = [(i, j) for idx_i, i in enumerate(kp) for j in kp[idx_i + 1:]]
			pairs = torch.as_tensor(out, dtype=torch.long)
			pairs = torch.sort(pairs, dim=1).values	# ensure i < j so downstream code always receives [start, end]
		# ------------  b) last R adjacent pairs ---------------------- #
		elif mode == "firstKeypointConsecutivePairs":
			adj = list(zip(kp[:-1], kp[1:]))			   # consecutive pairs
			adj = adj[:R]								 # keep the *first* R
			valid_n = len(adj)
			# zero-padding to fixed length R
			pairs = torch.zeros(R, 2, dtype=torch.long)
			if valid_n:
				pairs[:valid_n] = torch.as_tensor(adj, dtype=torch.long)
				pairs[:valid_n] = torch.sort(pairs[:valid_n], dim=1).values	# ensure i < j so downstream code always receives [start, end]
			#valid = torch.zeros(R, dtype=torch.bool)
		# ------------  c) first R starts, next Q key-points ------------------ #
		elif mode == "firstKeypointPairs":
			#orig unvectorised version
			out = []
			starts = kp[:R]								# first R starts
			for s_idx, s in enumerate(starts):
				# pick up to Q subsequent key-points
				next_kps = kp[s_idx + 1 : s_idx + 1 + (Q)]
				if not next_kps:							# none available
					out.extend([(0, 0)] * (Q))
				else:
					# pad to Q so output length is constant
					for t in range(Q):
						if t < len(next_kps):
							out.append((s, next_kps[t]))
						else:
							out.append((0, 0))
			pairs = torch.as_tensor(out, dtype=torch.long)		  # (R*Q, 2)
			pairs = torch.sort(pairs, dim=1).values	# ensure i < j so downstream code always receives [start, end]
			'''
			#vectorised version;
			# kp -> 1D tensor
			kp_t = torch.as_tensor(kp, dtype=torch.long, device=device)	# or kp.device if already tensor
			K = kp_t.numel()

			R = int(R)	# ensure plain ints in case they were tensors
			Q = int(Q)

			# indices of the first R starts
			start_vals = kp_t[:R]											# (R,)

			# build matrix of candidate indices into kp: row r -> r+1 ... r+Q
			row = torch.arange(R, device=kp_t.device).unsqueeze(1)			# (R,1)
			col = torch.arange(1, Q + 1, device=kp_t.device).unsqueeze(0)	# (1,Q)
			cand_idx = row + col												# (R,Q)

			valid = cand_idx < K												# (R,Q) mask
			# clamp to avoid OOB gather; we'll zero-out invalids afterward
			cand_idx_clamped = cand_idx.clamp_max(K - 1)

			next_vals = kp_t[cand_idx_clamped]								# (R,Q)
			next_vals = next_vals.masked_fill(~valid, 0)

			# repeat starts to align with next_vals
			start_repeat = start_vals.unsqueeze(1).expand(-1, Q)				# (R,Q)

			pairs = torch.stack((start_repeat, next_vals), dim=-1)			# (R,Q,2)
			pairs = pairs.view(-1, 2)										# (R*Q,2)
			pairs = torch.sort(pairs, dim=1).values							# ensure i<j
			'''
		else:
			raise ValueError(f"Unknown mode {mode}")
		valid = (pairs[:, 0] != pairs[:, 1])

		if reorderPairsToBeNotReversed and pairs.numel():
			_max = pairs.max().item() + 1
			order = torch.argsort(pairs[:, 0] * _max + pairs[:, 1])
			pairs = pairs[order]
			valid = valid[order]
	
	S = valid.shape[0]
	if(S < R*Q):
		#expand normalisedSnapshots with zeros if number keypoint pairs < R*Q
		#always assume Q is constant (already padded if necessary by make_pairs)
		Rcurrent = S//Q
		Rpad = R-Rcurrent
		if(Rpad > 0):
			Spad = Rpad*Q
			pairsPad = torch.zeros((Spad, 2), dtype=torch.long, device=pairs.device)
			validPad = torch.full((Spad,), False, dtype=torch.bool, device=valid.device)
			if(useSlidingWindow):
				#left pad;
				pairs = torch.cat((pairsPad, pairs), dim=0)
				valid = torch.cat((validPad, valid), dim=0)
			else:
				#right pad;
				pairs = torch.cat((pairs, pairsPad), dim=0)
				valid = torch.cat((valid, validPad), dim=0)

	return pairs, valid

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

