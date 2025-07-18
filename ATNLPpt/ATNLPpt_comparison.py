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
from typing import Sequence, Union, Tuple, Optional, List, Dict
if(ATNLPsnapshotDatabaseDisk):
	import h5py
	import numpy as np
	import os
import ATNLPpt_database

if(not ATNLPcomparisonShiftInvariance):
	if(ATNLPnormalisedSnapshotsSparseTensors):
		def _cosine_block(
				c_blk: torch.Tensor,   # sparse  (B_c, D)
				d_blk: torch.Tensor,   # sparse  (B_d, D)
				*, eps: float = 1e-8
		) -> torch.Tensor:
			"""
			Fast cosine-similarity between two sparse blocks *without* per-pair loops.

			Returns
			-------
			sim : (B_c, B_d) dense
			"""
			assert c_blk.is_sparse and d_blk.is_sparse
			Bc, D = c_blk.shape
			Bd, _ = d_blk.shape

			# ---- densify only the (usually small) candidate block -----------------
			c_dense = c_blk.to_dense()								# (Bc, D)
			c_norm  = torch.linalg.vector_norm(c_dense, dim=1).clamp_min(eps)  # (Bc,)

			# ---- DB norms remain sparse-friendly ----------------------------------
			d_norm  = torch.sqrt(
				torch.sparse.sum(d_blk.pow(2), dim=1).to_dense()
			).clamp_min(eps)										# (Bd,)

			# ---- sparse.mm :  (Bd, D)[sparse]  (D, Bc)[dense]^T \u2192 (Bd, Bc) ------
			#   We feed the dense matrix transposed so the sparse one is first.
			sim = torch.sparse.mm(d_blk, c_dense.T)					# (Bd, Bc)
			sim = sim.T												# (Bc, Bd)

			# ---- cosine normalisation --------------------------------------------
			sim /= (c_norm[:, None] * d_norm[None, :])

			return sim
				
if(ATNLPsnapshotDatabaseDiskCompareChunks):
	if(ATNLPcomparisonShiftInvariance):
		# ===================================================================== #
		#  STREAMING, OUT-OF-CORE VARIANT \u2013 loads DB from HDF5 in sparse chunks #
		# ===================================================================== #
		def compare_1d_batches_stream_db(
			self,
			candidates: torch.Tensor,			# (B1, S, C, L)   *sparse*
			B1: int,
			keypointPairsIndices: torch.Tensor,	 # shape (B1, S)
			kp_meta_batch: List[List[Dict]],
			*,
			shiftInvariantPixels: int | None = None,
			chunk_nnz: int = ATNLPsnapshotDatabaseDiskCompareChunksSize,			# non-zeros per slice
			device: str | torch.device = "cpu",
			eps: float = 1e-8,
		) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
			"""
			Same outputs as compare_1d_shift_invariant(), but the database is
			read on-the-fly from **h5_path**, keeping <O(B2 + chunk_nnz)** RAM**.
			"""
			comparisonFound = False
			unit_sim, top_cls, avg_sim = (None, None, None)
			device = torch.device(device)

			B1, S, C, L = candidates.shape
			B2 = B1*S
			
			# candidate norms (dense 1-D, tiny)
			candidates = candidates.coalesce()
			cand_dense = candidates.to_dense()							# (B1, S, C, L)
			cand_dense = cand_dense.view(B2, C, L)		# (B2, C, L)
			
			# ---- shift window ----------------------------------------------------
			K = (L - 1) if shiftInvariantPixels is None else int(shiftInvariantPixels)
			K = max(0, min(K, L - 1))

			# ---- dense candidates once, pre-compute norms -----------------------
			D = C * L
			cand_dense_flat = cand_dense.reshape(B2, D)					# (B2, D)
			cand_norm = torch.linalg.vector_norm(cand_dense_flat, dim=1) + eps	# (B2,)

			# pre-build all (2K+1) shifted versions \u2192 list of dense (B2, D)
			shifted = []
			for s in range(-K, K + 1):
				if s >= 0:
					tmp = torch.zeros_like(cand_dense)
					tmp[:, :, s:] = cand_dense[:, :, :L - s]
				else:	# shift left
					tmp = torch.zeros_like(cand_dense)
					tmp[:, :, :L + s] = cand_dense[:, :, -s:]
				shifted.append(tmp.reshape(B2, D).T.contiguous())		# transpose for GEMM

			# ---- running outputs -------------------------------------------------
			best_val = torch.full((B2,), -float("inf"), device=device)	# (B2)
			best_idx = torch.full((B2,), -1, dtype=torch.long, device=device)	# (B2)
			best_class = torch.full((B2,), -1, dtype=torch.long, device=device,)	# (B2)
			sumsq = torch.zeros(B2, device=device)		# for L2-unit sim

			# ---- tiny helper: sparse max-corr with shift window -----------------
			def _max_corr_row(c_idx, c_val, d_idx, d_val):
				best = 0.0
				for s in range(-K, K + 1):
					acc = 0.0
					for ic, vc in zip(c_idx, c_val):
						j = ic + s
						if j < 0 or j >= L:
							continue
						pos = torch.searchsorted(d_idx, j)
						if pos < d_idx.numel() and d_idx[pos] == j:
							acc += vc * d_val[pos]
					if acc > best:
						best = acc
				return best

			for b1 in range(B1):
				for s in range(S):
					keypointPairsIndexFirst = keypointPairsIndices[b1, s, 0]
					referenceSetDelimiterID = self.getReferenceSetDelimiterID(keypointPairsIndexFirst, kp_meta_batch[b1])
					h5_path = ATNLPpt_database.getDatabaseName(referenceSetDelimiterID, s)
					b2 = b1*B1 + s
					
					# ---- open HDF5 once and stream by nnz slices ------------------------
					if os.path.isfile(h5_path):
						with h5py.File(h5_path, "R") as h5:
							ptr_ds	= h5["img_ptr"]			  # (B3+1,)
							class_ds = torch.from_numpy(h5["classes"][...]).to(torch.int64)
							val_ds	 = h5["values"]
							idx_ds	 = h5["indices"]			  # (3, nnz)
							B3 = class_ds.numel()

							start = 0
							while start < B3:
								end = start + 1
								while end < B3 and (ptr_ds[end] - ptr_ds[start]) < chunk_nnz:
									end += 1

								lo, hi = ptr_ds[start], ptr_ds[end]
								val_blk = torch.from_numpy(val_ds[lo:hi]).to(device)
								idx_blk = torch.from_numpy(idx_ds[:, lo:hi])		# (3, nnz_blk)
								idx_blk[0] -= start								# renum imgs

								# build CSR pointer *inside* the slice
								img_ptr_local = torch.from_numpy(ptr_ds[start:end] - lo)
								nnz_blk = hi - lo

								# -- build sparse DB block once (n, D) ----------------------
								n_img = end - start
								#db_blk = torch.sparse_coo_tensor(
								#	idx_blk, val_blk, size=(n_img, D),
								#	dtype=torch.float32, device=device,).coalesce()
								#db_norm = torch.sqrt(torch.sparse.sum(db_blk.pow(2), dim=1).to_dense()) + eps  # (n,)
								flat_idx = idx_blk[1] * L + idx_blk[2]				# (nnz_blk,)
								new_idx  = torch.stack((idx_blk[0], flat_idx), dim=0)	# (2, nnz_blk)
								db_blk = torch.sparse_coo_tensor(
									new_idx, val_blk, (n_img, D),
									dtype=torch.float32, device=device).coalesce()
								db_norm = torch.sqrt(torch.sparse.sum(db_blk.pow(2), dim=1).to_dense()) + eps  # (n,)

								# container that tracks max over shifts for whole block
								sim_max = torch.full((1, n_img), -float("inf"), device=device)

								# ---- vectorised over shifts (2K+1 small) -----------------
								for s_idx, cand_shift_T in enumerate(shifted[b2].unsqueeze(0)):		# (D, 1)
									# sim_blk =  (1, n_img)  via   (D, 1)^T  \u2190 (n,D)sparse
									sim_blk = torch.sparse.mm(db_blk, cand_shift_T).T	# (1, n_img)
									sim_blk /= (cand_norm[:, None] * db_norm[None, :])
									sim_max = torch.maximum(sim_max, sim_blk)

								# ---- accumulate results for this block -------------------
								sumsq[b2] += (sim_max ** 2).sum(dim=1).squeeze(0)					# (1,)

								cur_best, cur_idx = sim_max.max(dim=1)				# (1,)
								mask = cur_best[0] > best_val[b2]
								if(mask):
									best_val[b2] = cur_best[0]
									best_idx[b2] = cur_idx[0] + start
									best_class[b2] = class_ds[best_idx[b2]]
									
								start = end
								
								comparisonFound = True
					else:
						pass
			if(comparisonFound):
				unit_sim, top_cls, avg_sim = calculateTopCls(eps, class_ds, sumsq, best_val, best_idx, best_class, B1, B2, S)

			return comparisonFound, unit_sim, top_cls, avg_sim

	else:
		def compare_1d_batches_stream_db(
			self,
			candidates: torch.Tensor,			# (B1, S, C, L)  sparse
			B1: int,
			keypointPairsIndices: torch.Tensor,	 # shape (B1, S)
			kp_meta_batch: List[List[Dict]],
			*,
			shiftInvariantPixels: int | None = None,
			chunk_nnz: int = ATNLPsnapshotDatabaseDiskCompareChunksSize,			# how many non-zeros per slice
			device: str | torch.device = "cpu",
			eps: float = 1e-8,
		) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
			"""
			Exactly the same API/semantics as compare_1d_batches, but the *database*
			is read lazily from the HDF5 file in sparse chunks so nothing huge is
			held in RAM.
			"""
			comparisonFound = False
			unit_sim, top_cls, avg_sim = (None, None, None)
			device = torch.device(device)
			assert candidates.is_sparse

			# ---- move candidates to device & flatten ------------------------------
			print("candidates.shape = ", candidates.shape)
			B1, S, C, L = candidates.shape
			B2 = B1*S
			
			# candidate norms (dense 1-D, tiny)
			candidates = candidates.coalesce()
			cand_dense = candidates.to_dense()
			cand_dense = cand_dense.view(B2, C, L)		# (B2, C, L)

			D = C * L
			cand_norm = torch.sqrt(torch.sum(cand_dense.reshape(B2, D).pow(2), dim=1) + eps)		# (B2,)
			cand_flat = cand_dense.reshape(B2, D).to_sparse_coo()	# (B2, D)
			'''
			candidates = candidates.to(device).coalesce()
			B2, C, L = candidates.shape
			D = C * L
			# ---- flatten sparse (B2, C, L) -> (B2, D) --------------------
			idx_c, val_c = candidates.indices(), candidates.values()		# (3, nnz)
			flat_idx_c   = idx_c[1] * L + idx_c[2]							# channelL + pos
			new_idx_c	= torch.stack((idx_c[0], flat_idx_c), dim=0)		 # (2, nnz)
			cand_flat	= torch.sparse_coo_tensor(
				new_idx_c, val_c, size=(B2, D),
				dtype=torch.float32, device=device).coalesce()			 # (B2, D)
			# candidate norms (dense 1-D, tiny)
			cand_norm = torch.sqrt(torch.sparse.sum(cand_flat.pow(2), dim=1).to_dense() + eps)  # (B2,)
			'''

			# ---- prepare running outputs ------------------------------------------
			best_val = torch.full((B2,), -float("inf"), device=device)	# (B2)
			best_idx = torch.full((B2,), -1, dtype=torch.long, device=device)	# (B2)
			best_class = torch.full((B2,), -1, dtype=torch.long, device=device)	# (B2)
			sumsq = torch.zeros(B2, device=device)	 # accumulate sim for L2 later	# (B2)

			for b1 in range(B1):
				for s in range(S):
					keypointPairsIndexFirst = keypointPairsIndices[b1, s, 0]
					referenceSetDelimiterID = self.getReferenceSetDelimiterID(keypointPairsIndexFirst, kp_meta_batch[b1])
					h5_path = ATNLPpt_database.getDatabaseName(referenceSetDelimiterID, s)
					b2 = b1*B1 + s

					# ---- open HDF5 once ----------------------------------------------------
					if os.path.isfile(h5_path):
						with h5py.File(h5_path, "R") as h5:
							ptr_ds	= h5["img_ptr"]	  # (B3+1,)
							class_ds = torch.from_numpy(np.asarray(h5["classes"], dtype=np.int64))
							B3 = class_ds.numel()
							C_tot = C	 # same as candidates, but keep for clarity
							L_tot = L

							val_ds = h5["values"]
							idx_ds = h5["indices"]	# (3, nnz)

							# ---- iterate over slices whose NNZ \u2264 chunk_nnz ---------------------
							start = 0
							while start < B3:
								# grow 'end' until nnz(end) - nnz(start) \u2248 chunk_nnz
								end = start + 1
								while end < B3 and (ptr_ds[end] - ptr_ds[start]) < chunk_nnz:
									end += 1

								lo = ptr_ds[start]
								hi = ptr_ds[end]	  # one-past\u2010last
								nnz_blk = hi - lo

								# ---- slice raw arrays & build sparse block ------------------
								val_blk = torch.from_numpy(val_ds[lo:hi]).to(device)
								idx_blk = torch.from_numpy(idx_ds[:, lo:hi])	 # (3, nnz)

								# re-index image numbers within block to 0\u2026end-start-1
								idx_blk[0] -= start

								db_blk = torch.sparse_coo_tensor(
									idx_blk.to(torch.int64), val_blk,
									size=(end - start, C_tot, L_tot),
									dtype=torch.float32, device=device
								).coalesce()

								# flatten images in block
								#db_blk_flat = db_blk.reshape(end - start, -1)
								db_blk = torch.sparse_coo_tensor(
									idx_blk.to(torch.int64), val_blk,
									size=(end - start, C_tot, L_tot),
									dtype=torch.float32, device=device
								).coalesce()
								# ---- flatten sparse block to 2-D --------------------
								n_img = end - start
								flat_idx = idx_blk[1] * L_tot + idx_blk[2]				# (nnz_blk,)
								new_idx  = torch.stack((idx_blk[0], flat_idx), dim=0)		# (2, nnz_blk)
								db_blk_flat = torch.sparse_coo_tensor(
									new_idx, val_blk, (n_img, D),
									dtype=torch.float32, device=device).coalesce()		# (n_img, D)

								# ---- cosine similarities (dense (1, nBlkImgs)) -------------
								#sim_blk = _cosine_block(candidates.reshape(1, -1).coalesce(), db_blk_flat)												# (1, end-start)
								sim_blk = _cosine_block(cand_flat[b2].unsqueeze(0), db_blk_flat)		# (1, n_img)

								# ---- update running top-k & sum-of-squares -------------------
								sumsq[b2] += (sim_blk ** 2).sum(dim=1).squeeze(0)

								cur_best, cur_idx = sim_blk.max(dim=1)			# (1,)
								mask = cur_best[0] > best_val[b2]
								if(mask):
									best_val[b2] = cur_best[0]
									best_idx[b2] = cur_idx[0] + start
									best_class[b2] = class_ds[best_idx[b2]]

								start = end		# next slice
								
								comparisonFound = True
					else:
						pass
			if(comparisonFound):
				unit_sim, top_cls, avg_sim = calculateTopCls(eps, class_ds, sumsq, best_val, best_idx, best_class, B1, B2, S)

			return comparisonFound, unit_sim, top_cls, avg_sim		
else:
	if(ATNLPcomparisonShiftInvariance):
		def compare_1d_batches(
			self,
			candidates: torch.Tensor,	   # (B1, S, C, L)
			B1: int,
			keypointPairsIndices: torch.Tensor,	 # shape (B1, S)
			kp_meta_batch: List[List[Dict]],
			*,
			shiftInvariantPixels: int = None,   # None -> full (L-1)
			chunk: int | None = None,
			eps: float = 1e-8,
			device="cpu",
		):
			"""
			FFT cross-correlation with optional K-pixel invariance.
			"""
			
			comparisonFound = False
			unit_sim, top_cls, avg_sim = (None, None, None)
			B1, S, C, L = candidates.shape
			B2 = B1*S	
			device = torch.device(device)
			candidates = candidates.to(device, dtype=torch.float32)						   # snapshots per sample
		
			simList = []
			B2sim_valsList = []
			B2sim_idsList = []
			B2sim_classList = []
			for b1 in range(B1):
				for s in range(S):
					keypointPairsIndexFirst = keypointPairsIndices[b1, s, 0]
					referenceSetDelimiterID = self.getReferenceSetDelimiterID(keypointPairsIndexFirst, kp_meta_batch[b1])	
					if(self.database[referenceSetDelimiterID][s] is not None):
						candidate = candidates[b1, s].unsqueeze(0).coalesce()	# shape (1, C, L) 	#legacy code expects B2 dimension (set to 1)	
						database = self.database[referenceSetDelimiterID][s].to(ATNLPsnapshotDatabaseLoadDevice, dtype=torch.float32)	# shape (B3, C, L)   - B3 >> B2	
						if(ATNLPindexDatabaseByClassTarget):
							db_classes = torch.arange(0, numberOfClasses, dtype=torch.int64, device=device)	# shape (B3,)  - int64 class targets
							databaseOverloadNumber = self.db_classes[referenceSetDelimiterID][s].to(device)	# shape (B3)  - int64
							database = renormalise(database, databaseOverloadNumber)	#renormalise transformed snapshots by overload number
						else:
							db_classes = self.db_classes[referenceSetDelimiterID][s].to(device)	# shape (B3,)  - int64 class targets
						b2 = b1*B1 + s

						B3 = database.shape[0]
						K = (L - 1) if shiftInvariantPixels is None else int(shiftInvariantPixels)
						K = max(0, min(K, L - 1))				 # clamp

						fft_len	= 2 * L - 1							 # full linear corr length
						idx0 = L - 1								 # zero-shift index
						low, high = idx0 - K, idx0 + K + 1				# slice bounds

						# NOTE: torch.fft.* does **not** support sparse tensors.  Replace the
						# FFT-based pipeline with explicit index-matching cross-correlation.

						def _max_corr_row(c_idx, c_val, d_idx, d_val):
							best = 0.0
							for shift in range(-K, K + 1):
								acc = 0.0
								for ic, vc in zip(c_idx, c_val):
									j = ic + shift
									if j < 0 or j >= L:
										continue
									pos = torch.searchsorted(d_idx, j)
									if pos < d_idx.numel() and d_idx[pos] == j:
										acc += vc * d_val[pos]
								best = max(best, acc)
							return best

						# ---- flatten, then brute-force over sparse indices --------------------
						cand_nrm = torch.sqrt(torch.sparse.sum(candidate.pow(2), dim=(1, 2)).to_dense() + eps)	# (1,)	 # GPU
						db_nrm	= torch.sqrt(torch.sparse.sum(database.pow(2),   dim=(1, 2)).to_dense() + eps).to(device)	# (B3,)	# GPU view

						# -------- vectorised replacement --------------------------------
						#D = C * L
						#db_flat = database.reshape(B3, D)							# sparse (B3,D)
						#db_norm = torch.sqrt(torch.sparse.sum(db_flat.pow(2), dim=1).to_dense()) + eps  # (B3,)
						D = C * L
						idx, vals = database.indices(), database.values()			# (3, nnz)
						flat_idx  = idx[1] * L + idx[2]								# (nnz,)
						new_idx   = torch.stack((idx[0], flat_idx), dim=0)			# (2, nnz)
						db_flat   = torch.sparse_coo_tensor(
							new_idx, vals, size=(B3, D),
							dtype=torch.float32, device=device).coalesce()
						db_norm   = torch.sqrt(torch.sparse.sum(db_flat.pow(2), dim=1).to_dense()) + eps  # (B3,)

						# build all candidate shifts (same code as above)
						cand_dense = candidate.to_dense()
						cand_dense_flat = cand_dense.reshape(1, D)
						cand_norm = torch.linalg.vector_norm(cand_dense_flat, dim=1) + eps	# (1,)

						shifted = []
						for s in range(-K, K + 1):
							if s >= 0:
								tmp = torch.zeros_like(cand_dense)
								tmp[:, :, s:] = cand_dense[:, :, :L - s]
							else:
								tmp = torch.zeros_like(cand_dense)
								tmp[:, :, :L + s] = cand_dense[:, :, -s:]
							shifted.append(tmp.reshape(1, D).T.contiguous())		# (D, 1)

						# max\u2010over\u2010shift similarity matrix  (1, B3)
						sim = torch.full((1, B3), -float("inf"), device=device)
						for cand_shift_T in shifted:
							sim_blk = torch.sparse.mm(db_flat, cand_shift_T).T		# (1, B3)
							sim_blk /= (cand_norm[:, None] * db_norm[None, :])
							sim = torch.maximum(sim, sim_blk)
						
						sumsq = (sim ** 2).sum(dim=1)			# (1,)  for unit_sim later	#NOTUSED
						
						sim_val, sim_id = torch.max(sim, dim=1)	# (1)
						sim_class = db_classes[sim_id[0]].unsqueeze(0)
						comparisonFound = True
						print("comparisonFound")
					else:
						sim_val = torch.tensor([-1], device=device)	#CHECKTHIS
						sim_id = torch.tensor([0], device=device)	#CHECKTHIS
						sim_class = torch.tensor([-1], device=device)	#CHECKTHIS
						sim = None #torch.zeros((1, 1), device=device, device=device)
						#print("self.database[referenceSetDelimiterID][s] is None")
					
					simList.append(sim)
					B2sim_valsList.append(sim_val)
					B2sim_idsList.append(sim_id)
					B2sim_classList.append(sim_class)

					simList.append(sim)

			if(comparisonFound):
				sim = None	#B3 is no longer consistent across B2
				B2sim_vals = torch.cat(B2sim_valsList, dim=0)	 # (B2)
				B2sim_ids = torch.cat(B2sim_idsList, dim=0)	 # (B2)
				B2sim_class = torch.cat(B2sim_classList, dim=0)	 # (B2)
				unit_sim, top_cls, avg_sim = calculateTopCls(eps, db_classes, sim, B2sim_vals, B2sim_ids, B2sim_class, B1, B2, S)
			
			return comparisonFound, unit_sim, top_cls, avg_sim

	else:
		# ------------------------------------------------------------------ #
		#  helper: flatten (B, C, L) sparse \u2192 (B, D = CL) sparse            #
		# ------------------------------------------------------------------ #
		def _flatten_sparse_3d_to_2d(t: torch.Tensor) -> torch.Tensor:
			"""
			t : sparse COO (B, C, L)
			returns : sparse COO (B, CL)
			"""
			assert t.is_sparse
			B, C, L = t.shape
			idx, val = t.indices(), t.values()			# (3, nnz)
			flat_idx = idx[1] * L + idx[2]				# channelL + pos
			new_idx  = torch.stack((idx[0], flat_idx), dim=0)	# (2, nnz)
			return torch.sparse_coo_tensor(
				new_idx, val, (B, C * L),
				dtype=t.dtype, device=t.device,).coalesce()

		def compare_1d_batches(
			self,
			candidates: torch.Tensor,		   # shape (B1, S, C, L)
			B1: int,
			keypointPairsIndices: torch.Tensor,	 # shape (B1, S)
			kp_meta_batch: List[List[Dict]],
			*,
			shiftInvariantPixels: int | None = None,
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
			
			comparisonFound = False
			unit_sim, top_cls, avg_sim = (None, None, None)
			B1, S, C, L = candidates.shape
			B2 = B1*S
			device = torch.device(device)
			candidates = candidates.to(device, dtype=torch.float32)
			
			simList = []
			B2sim_valsList = []
			B2sim_idsList = []
			B2sim_classList = []
			for b1 in range(B1):
				for s in range(S):
					keypointPairsIndexFirst = keypointPairsIndices[b1, s, 0]
					referenceSetDelimiterID = self.getReferenceSetDelimiterID(keypointPairsIndexFirst, kp_meta_batch[b1])
					if(self.database[referenceSetDelimiterID][s] is not None):
						candidate = candidates[b1, s].unsqueeze(0).coalesce()	# shape (1, C, L) 	#legacy code expects B2 dimension (set to 1)	
						database = self.database[referenceSetDelimiterID][s].to(ATNLPsnapshotDatabaseLoadDevice, dtype=torch.float32)	# shape (B3, C, L)   - B3 >> B2
						if(ATNLPindexDatabaseByClassTarget):
							db_classes = torch.arange(0, numberOfClasses, dtype=torch.int64, device=device)	# shape (B3,)  - int64 class targets
							databaseOverloadNumber = self.db_classes[referenceSetDelimiterID][s].to(device)	# shape (B3)  - int64
							database = renormalise(database, databaseOverloadNumber)	#renormalise transformed snapshots by overload number
						else:
							db_classes = self.db_classes[referenceSetDelimiterID][s].to(device)	# shape (B3,)  - int64 class targets
						b2 = b1*B1 + s

						B3, C2, L2 = database.shape
						assert C == C2 and L == L2, "Candidates and database must have identical C and L"

						## ---- flatten to 2-D *sparse* feature matrices --------------------------
						#cand_feat = candidate.reshape(1, -1).coalesce()		# (1, D) - sparse	# GPU
						#db_feat	= database.reshape(B3, -1).coalesce()		# (B3, D) - sparse	# CPU
						# ---- flatten to (B, D) sparse matrices -------------------------------
						cand_feat = _flatten_sparse_3d_to_2d(candidate).to(device)	 # (1, D)
						db_feat = _flatten_sparse_3d_to_2d(database).coalesce()	 # (B3, D) on CPU

						# ---- pre-compute L2 norms (dense 1-D vectors, tiny) ---------------------
						cand_norm = torch.sqrt(torch.sparse.sum(cand_feat.pow(2), dim=1).to_dense() + eps)	# (1,)
						db_norm	= torch.sqrt(torch.sparse.sum(db_feat.pow(2),  dim=1).to_dense() + eps)	# (B3,)

						if chunk is None:
							sim = _cosine_block(cand_feat, db_feat.to(device))				# (1, B3)
						else:
							parts = []
							for s in range(0, B3, chunk):
								e = min(s + chunk, B3)
								d_blk = db_feat[s:e].to(device, non_blocking=True)
								parts.append(_cosine_block(cand_feat, d_blk))	# (1, e-s)
							sim = torch.cat(parts, dim=1)						  # (1, B3)
						
						sim_val, sim_id = torch.max(sim, dim=1)	# (1)
						sim_class = db_classes[sim_id[0]].unsqueeze(0)
						comparisonFound = True
						print("comparisonFound")
					else:
						sim_val = torch.tensor([-1], device=device)	#CHECKTHIS
						sim_id = torch.tensor([0], device=device)	#CHECKTHIS
						sim_class = torch.tensor([-1], device=device)	#CHECKTHIS
						sim = None #torch.zeros((1, 1), device=device)
						#print("self.database[referenceSetDelimiterID][s] is None")
					
					simList.append(sim)
					B2sim_valsList.append(sim_val)
					B2sim_idsList.append(sim_id)
					B2sim_classList.append(sim_class)
					
			if(comparisonFound):
				sim = None	#B3 is no longer consistent across B2
				B2sim_vals = torch.cat(B2sim_valsList, dim=0)	 # (B2)
				B2sim_ids = torch.cat(B2sim_idsList, dim=0)	 # (B2)
				B2sim_class = torch.cat(B2sim_classList, dim=0)	 # (B2)
				unit_sim, top_cls, avg_sim = calculateTopCls(eps, db_classes, sim, B2sim_vals, B2sim_ids, B2sim_class, B1, B2, S)
			
			return comparisonFound, unit_sim, top_cls, avg_sim

def renormalise(database, databaseOverloadNumber):
	if(ATNLPrenormaliseTransformedSnapshots):
		#renormalise database transformed snapshots by overload number
		#database.shape = shape (B3, C, L) - sparse COO tensor
		#databaseOverloadNumber.shape = (B3) - dense tensor

		database = database.coalesce()            # unique-index form
		inds = database._indices()                # (3, nnz)
		vals = database._values()                 # (nnz,)

		scale = databaseOverloadNumber     # (B3,)
		vals.div_(scale.index_select(0, inds[0]) )                 # in-place

		database = torch.sparse_coo_tensor(inds, vals, database.size(), device=database.device)
	return database

def calculateTopCls(eps, db_classes, sim, B2sim_vals, B2sim_ids, B2sim_class, B1, B2, S):

	#B3 is no longer consistent across B2; sim not available
	
	B2sim_vals = B2sim_vals.view(B1, S)		# (B1, S)
	B2sim_ids = B2sim_ids.view(B1, S)		# (B1, S)
	B2sim_class = B2sim_class.view(B1, S)		# (B1, S)

	avg_sim = B2sim_vals.mean(dim=1)		# (B1)
	top_cls, _ = torch.mode(B2sim_class, dim=1)  # returns (values, indices)	#TODO: find a better method (prioritise B2sim_class[B1, 0] ie nearest reference set class prediction)

	if(ATNLPsnapshotDatabaseDiskCompareChunks):
		sumsq = sim
		#unit_sim = (sumsq / (sumsq.sum(dim=1, keepdim=True) + eps)).sqrt()
		unit_sim = torch.sqrt(sumsq + eps)		# shape (B2,)
	else:
		#B3 is no longer consistent across B2; sim not available
		#unit_sim = F.normalize(sim, p=2, dim=1, eps=eps)					# (B2, B3)
		unit_sim = None
	
	return unit_sim, top_cls, avg_sim
	
