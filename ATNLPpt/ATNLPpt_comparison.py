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
from typing import Sequence, Union, Tuple, Optional

@torch.inference_mode()
def compare_1d_batches(
	candidates: torch.Tensor,		   # shape (B2, C, L)
	database: torch.Tensor,			 # shape (B3, C, L)   - B3 >> B2
	db_classes: torch.Tensor,		   # shape (B3,)  - int64 class targets
	B1: int,
	*,
	chunk: Optional[int] = None,		# split database along B3 for memory (None = no chunking)
	eps: float = 1e-8,
	device: str | torch.device = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	"""
	1-D cosine comparison of a batch of **candidates** against a large **database**.

	Returns
	-------
	unit_sim : (B2, B3)  - each row is the *unit-L2* similarity vector of one candidate
	top_cls  : (B2,)	 - class target of the single database item most similar to each candidate
	avg_top  : ()		- scalar, average of those top similarities across the whole candidate batch
	"""
	device = torch.device(device)
	candidates = candidates.to(device, dtype=torch.float32)
	database   = database.to(device, dtype=torch.float32)
	db_classes = db_classes.to(device)

	B2, C, L   = candidates.shape
	B3, C2, L2 = database.shape
	assert C == C2 and L == L2, "Candidates and database must have identical C and L"

	# ---------------------------------------------------------------------- #
	# 1. flatten (C, L) -> (D) and L2-normalise feature vectors			   #
	# ---------------------------------------------------------------------- #
	cand_feat = F.normalize(candidates.reshape(B2, -1), p=2, dim=1, eps=eps)  # (B2, D)
	db_feat = F.normalize(database.reshape(B3, -1),   p=2, dim=1, eps=eps)  # (B3, D)

	# ---------------------------------------------------------------------- #
	# 2. pairwise cosine similarities: (B2, D)  (D, B3) -> (B2, B3).		  #
	#	If B3 is huge, do it in manageable chunks to save RAM.			  #
	# ---------------------------------------------------------------------- #
	if chunk is None:
		sim = cand_feat @ db_feat.T							# (B2, B3)
	else:
		parts = []
		for s in range(0, B3, chunk):
			e = min(s + chunk, B3)
			parts.append(cand_feat @ db_feat[s:e].T)		   # (B2, e-s)
		sim = torch.cat(parts, dim=1)						  # (B2, B3)

	# ---------------------------------------------------------------------- #
	# 3. convert each row into a **unit similarity vector** (L2 = 1).		#
	# ---------------------------------------------------------------------- #
	unit_sim = F.normalize(sim, p=2, dim=1, eps=eps)		   # (B2, B3)

	'''
	# ---------------------------------------------------------------------- #
	# 4. find best-matching DB entry for every candidate.					#
	#	 top_idx  : argmax similarity along B3							 #
	#	 top_vals : corresponding similarity scores						#
	# ---------------------------------------------------------------------- #
	top_vals, top_idx = sim.max(dim=1)						 # (B2,), (B2,)

	# class targets of those best matches
	top_cls = db_classes[top_idx]							  # (B2,)

	# average similarity of the \u201cwinning\u201d matches (scalar)
	avg_top = top_vals.mean()								  # ()
	
	   return unit_sim, top_cls, avg_top
	'''
	
	# --- aggregate over S snapshots per logical sample -----------------------
	B2, B3 = sim.shape
	assert B2 % B1 == 0, "B2 must be a multiple of base_batch"
	S	  = B2 // B1								   # snapshots per sample

	mean_sim = sim.view(B1, S, B3).mean(dim=1)		  # (B1, B3)
	best_vals, best_idx = mean_sim.max(dim=1)		   # (B1,)

	top_cls = db_classes[best_idx]					  # (B1,)
	avg_sim = best_vals								 # (B1,)

	return unit_sim, top_cls, avg_sim
	



@torch.inference_mode()
def compare_1d_shift_invariant(
	candidates: torch.Tensor,	   # (B2, C, L)
	database:   torch.Tensor,	   # (B3, C, L)
	db_classes: torch.Tensor,	   # (B3,)
	B1: int,
	*,
	shiftInvariantPixels: int = None,   # None -> full (L-1)
	chunk: int | None = None,
	eps: float = 1e-8,
	device="cpu",
):
	"""
	FFT cross-correlation with optional K-pixel invariance.
	"""
	device = torch.device(device)
	candidates = candidates.to(device, dtype=torch.float32)
	database   = database.to(device,   dtype=torch.float32)
	db_classes = db_classes.to(device)

	B2, C, L   = candidates.shape
	B3		 = database.shape[0]
	K		  = (L - 1) if shiftInvariantPixels is None else int(shiftInvariantPixels)
	K		  = max(0, min(K, L - 1))				 # clamp

	fft_len	= 2 * L - 1							 # full linear corr length
	idx0	   = L - 1								 # zero-shift index
	low, high  = idx0 - K, idx0 + K + 1				# slice bounds

	# ----------------------------------------------------------- FFT once
	cand_fft  = torch.fft.rfft(candidates, n=fft_len)  # (B2, C, F)
	cand_nrm  = torch.linalg.vector_norm(candidates, dim=(1, 2))  # (B2,)

	# prep outputs
	sim_rows, best_vals, best_idx = [], [], []

	step = chunk or B3
	for s in range(0, B3, step):
		e = min(s + step, B3)

		db_fft  = torch.fft.rfft(database[s:e], n=fft_len)	 # (chunk, C, F)
		db_nrm  = torch.linalg.vector_norm(database[s:e], dim=(1, 2))  # (chunk,)

		# cross-spectra summed over channels
		prod = (cand_fft.unsqueeze(1) * db_fft.conj()).sum(dim=2)	   # (B2, chunk, F)
		corr = torch.fft.irfft(prod, n=fft_len, dim=2)				  # (B2, chunk, 2L-1)

		# ---- keep only lags in K ------------------------------------
		corr_window = corr[..., low:high]							   # (B2, chunk, 2K+1)
		maxcorr, shift_idx = corr_window.max(dim=2)					 # (B2, chunk)

		sim_block = maxcorr / (cand_nrm[:, None] * db_nrm[None, :] + eps)  # cosine-like

		sim_rows.append(sim_block)

		# best per candidate within this chunk
		top_vals_blk, top_idx_blk = sim_block.max(dim=1)
		best_vals.append(top_vals_blk)
		best_idx.append(top_idx_blk + s)

	sim = torch.cat(sim_rows, dim=1)									# (B2, B3)
	unit_sim = F.normalize(sim, p=2, dim=1, eps=eps)					# (B2, B3)

	'''
	# global best match for each candidate
	best_vals = torch.stack(best_vals, dim=1)						   # (B2, nChunks)
	best_idx  = torch.stack(best_idx,  dim=1)						   # (B2, nChunks)
	top_vals, col = best_vals.max(dim=1)								# (B2,)
	top_idx  = best_idx[torch.arange(B2), col]						  # (B2,)

	top_cls  = db_classes[top_idx]									  # (B2,)
	avg_top  = top_vals.mean()										  # scalar
	
	return unit_sim, top_cls, avg_top
	'''
	
	# --- aggregate over snapshots -------------------------------------------
	B2, B3 = sim.shape
	assert B2 % B1 == 0, "B2 must be a multiple of base_batch"
	S = B2 // B1

	mean_sim = sim.view(B1, S, B3).mean(dim=1)		  # (B1, B3)
	best_vals, best_idx = mean_sim.max(dim=1)		   # (B1,)

	top_cls  = db_classes[best_idx]					  # (B1,)
	avg_sim  = best_vals								 # (B1,)

	return unit_sim, top_cls, avg_sim




