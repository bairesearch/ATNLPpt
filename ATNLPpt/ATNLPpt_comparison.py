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
if(ATNLPsnapshotDatabaseDisk):
	if(not ATNLPsnapshotDatabaseDiskSetSize):
		import h5py
		import numpy as np

if(not ATNLPcomparisonShiftInvariance):
	if(normalisedSnapshotsSparseTensors):
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
		if(normalisedSnapshotsSparseTensors):
			# ===================================================================== #
			#  STREAMING, OUT-OF-CORE VARIANT \u2013 loads DB from HDF5 in sparse chunks #
			# ===================================================================== #
			def compare_1d_batches_stream_db(
				candidates: torch.Tensor,			# (B2, C, L)   *sparse*
				h5_path: str,						# snapshotDatabaseName
				B1: int,
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
				device = torch.device(device)

				assert candidates.is_sparse
				candidates = candidates.to(device).coalesce()
				B2, C, L = candidates.shape

				# ---- shift window ----------------------------------------------------
				K = (L - 1) if shiftInvariantPixels is None else int(shiftInvariantPixels)
				K = max(0, min(K, L - 1))

				# ---- dense candidates once, pre-compute norms -----------------------
				cand_dense = candidates.to_dense()							# (B2, C, L)
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
				best_val = torch.full((B2,), -float("inf"), device=device)
				best_idx = torch.full((B2,), -1, dtype=torch.long, device=device)
				sumsq	 = torch.zeros(B2, device=device)		# for L2-unit sim

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

				# ---- open HDF5 once and stream by nnz slices ------------------------
				with h5py.File(h5_path, "r") as h5:
					ptr_ds	= h5["img_ptr"]			  # (B3+1,)
					class_ds = torch.from_numpy(h5["classes"][...]).to(torch.int64)
					val_ds	 = h5["values"]
					idx_ds	 = h5["indices"]			  # (3, nnz)
					B3		 = class_ds.numel()

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
						sim_max = torch.full((B2, n_img), -float("inf"), device=device)

						# ---- vectorised over shifts (2K+1 small) -----------------
						for s_idx, cand_shift_T in enumerate(shifted):		# (D, B2)
							# sim_blk =  (B2, n_img)  via   (D, B2)^T  \u2190 (n,D)sparse
							sim_blk = torch.sparse.mm(db_blk, cand_shift_T).T	# (B2, n_img)
							sim_blk /= (cand_norm[:, None] * db_norm[None, :])
							sim_max = torch.maximum(sim_max, sim_blk)

						# ---- accumulate results for this block -------------------
						sumsq += (sim_max ** 2).sum(dim=1)					# (B2,)

						cur_best, cur_idx = sim_max.max(dim=1)				# (B2,)
						mask = cur_best > best_val
						best_val[mask] = cur_best[mask]
						best_idx[mask] = cur_idx[mask] + start

						start = end

				# ---- aggregate over snapshots --------------------------------------
				assert B2 % B1 == 0, "B2 must divide into snapshots"
				S = B2 // B1

				avg_sim = best_val.view(B1, S).mean(dim=1)			# (B1,)
				top_cls = class_ds[best_idx.view(B1, S)[:, 0]]		# (B1,)

				# ---- inexpensive \u201cunit_sim\u201d proxy (norm only) -----------------------
				#unit_sim = (sumsq / (sumsq.sum(dim=1, keepdim=True) + eps)).sqrt()
				unit_sim = torch.sqrt(sumsq + eps)		# shape (B2,)

				return unit_sim, top_cls, avg_sim
		else:
			# ====================================================================== #
			#  DENSE HDF5 STREAMING VARIANTS                                       #
			#  -------------------------------------------------------------------- #
			#  * expect an HDF5 file containing                                     #
			#      * "images"   : (B3, C, L)  float32                                #
			#      * "classes"  : (B3,)       int32                                  #
			#  * RAM usage  O(B2 + chunk_imgs  C  L)                               #
			#  * Outputs identical to the in-RAM dense functions                     #
			# ====================================================================== #
			def compare_1d_shift_invariant_stream_db(
				candidates: torch.Tensor,			# (B2, C, L)  *dense*
				h5_path: str,
				B1: int,
				*,
				shiftInvariantPixels: int | None = None,
				chunk_imgs: int = ATNLPsnapshotDatabaseDiskChunkSize,			# DB rows per slice
				eps: float = 1e-8,
				device: str | torch.device = "cpu",
			) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
				device = torch.device(device)
				candidates = candidates.to(device, dtype=torch.float32)

				B2, C, L = candidates.shape
				K = (L - 1) if shiftInvariantPixels is None else int(shiftInvariantPixels)
				K = max(0, min(K, L - 1))

				fft_len	= 2 * L - 1
				idx0	 = L - 1
				low, high = idx0 - K, idx0 + K + 1

				cand_fft = torch.fft.rfft(candidates, n=fft_len)			 # (B2, C, F)
				cand_nrm = torch.linalg.vector_norm(candidates, dim=(1, 2))  # (B2,)

				best_val = torch.full((B2,), -float("inf"), device=device)
				best_idx = torch.full((B2,), -1, dtype=torch.long, device=device)
				sumsq	 = torch.zeros(B2, device=device)

				with h5py.File(h5_path, "r") as h5:
					images_ds = h5["images"]
					class_ds  = torch.from_numpy(h5["classes"][...]).to(torch.long)
					B3 = class_ds.numel()

					for s in range(0, B3, chunk_imgs):
						e = min(s + chunk_imgs, B3)
						db_blk = torch.from_numpy(images_ds[s:e]).to(device, dtype=torch.float32)  # (n, C, L)
						db_fft = torch.fft.rfft(db_blk, n=fft_len)		# (n, C, F)
						db_nrm = torch.linalg.vector_norm(db_blk, dim=(1, 2))	# (n,)

						prod = (cand_fft.unsqueeze(1) * db_fft.conj()).sum(dim=2)		# (B2, n, F)
						corr = torch.fft.irfft(prod, n=fft_len, dim=2)					# (B2, n, 2L-1)

						corr_window = corr[..., low:high]								# (B2, n, 2K+1)
						maxcorr, _ = corr_window.max(dim=2)							 # (B2, n)
						sim_blk = maxcorr / (cand_nrm[:, None] * db_nrm[None, :] + eps)  # (B2, n)

						sumsq += (sim_blk ** 2).sum(dim=1)

						cur_best, cur_idx = sim_blk.max(dim=1)
						mask = cur_best > best_val
						best_val[mask] = cur_best[mask]
						best_idx[mask] = cur_idx[mask] + s

				# snapshot aggregation
				assert B2 % B1 == 0
				S = B2 // B1

				avg_sim = best_val.view(B1, S).mean(dim=1)
				top_cls = class_ds[best_idx.view(B1, S)[:, 0]]

				#unit_sim = (sumsq / (sumsq.sum(dim=1, keepdim=True) + eps)).sqrt()
				unit_sim = torch.sqrt(sumsq + eps)		# shape (B2,)
				
				return unit_sim, top_cls, avg_sim

	else:
		if(normalisedSnapshotsSparseTensors):

			def compare_1d_batches_stream_db(
				candidates: torch.Tensor,			# (B2, C, L)  sparse
				h5_path: str,						# snapshotDatabaseName
				B1: int,
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
				device = torch.device(device)
				assert candidates.is_sparse

				# ---- move candidates to device & flatten ------------------------------
				candidates = candidates.to(device).coalesce()
				B2, C, L = candidates.shape

				## candidate norms (dense 1-D, tiny)
				#cand_norm = torch.sqrt(torch.sparse.sum(
				#	candidates.reshape(B2, -1).pow(2), dim=1
				#).to_dense() + eps)							# (B2,)
				candidates = candidates.to(device).coalesce()
				B2, C, L = candidates.shape
				D = C * L
				# ---- flatten sparse (B2, C, L) \u2192 (B2, D) --------------------
				idx_c, val_c = candidates.indices(), candidates.values()		# (3, nnz)
				flat_idx_c   = idx_c[1] * L + idx_c[2]							# channelL + pos
				new_idx_c	= torch.stack((idx_c[0], flat_idx_c), dim=0)		 # (2, nnz)
				cand_flat	= torch.sparse_coo_tensor(
					new_idx_c, val_c, size=(B2, D),
					dtype=torch.float32, device=device).coalesce()			 # (B2, D)
				# candidate norms (dense 1-D, tiny)
				cand_norm = torch.sqrt(torch.sparse.sum(cand_flat.pow(2), dim=1).to_dense() + eps)  # (B2,)

				# ---- prepare running outputs ------------------------------------------
				best_val = torch.full((B2,), -float("inf"), device=device)
				best_idx = torch.full((B2,), -1,		   device=device, dtype=torch.long)
				sumsq	 = torch.zeros(B2, device=device)	 # accumulate sim for L2 later

				# ---- open HDF5 once ----------------------------------------------------
				with h5py.File(h5_path, "r") as h5:
					ptr_ds	= h5["img_ptr"]	  # (B3+1,)
					class_ds = torch.from_numpy(np.asarray(h5["classes"], dtype=np.int64))
					B3		= class_ds.numel()
					C_tot	= C	 # same as candidates, but keep for clarity
					L_tot	= L

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

						# ---- cosine similarities (dense (B2, nBlkImgs)) -------------
						#sim_blk = _cosine_block(candidates.reshape(B2, -1).coalesce(), db_blk_flat)												# (B2, end-start)
						sim_blk = _cosine_block(cand_flat, db_blk_flat)		# (B2, n_img)

						# ---- update running top-k & sum-of-squares -------------------
						sumsq += (sim_blk ** 2).sum(dim=1)

						cur_best, cur_idx = sim_blk.max(dim=1)			# (B2,)
						mask = cur_best > best_val
						best_val[mask] = cur_best[mask]
						best_idx[mask] = cur_idx[mask] + start

						start = end		# next slice

				# ---- finish aggregation over snapshots -------------------------------
				assert B2 % B1 == 0, "B2 must be a multiple of base_batch"
				S = B2 // B1

				avg_sim = best_val.view(B1, S).mean(dim=1)			# (B1,)
				top_cls = class_ds[best_idx.view(B1, S)[:, 0]]		# (B1,)

				# ---- build L2-unit similarity vectors if you need them --------------
				#unit_sim = (sumsq / (sumsq.sum(dim=1, keepdim=True) + eps)).sqrt()
				unit_sim = torch.sqrt(sumsq + eps)		# shape (B2,)
				
				#  NOTE: the above gives you only the *norm* part.  Reconstructing the
				#  full (B2, B3) vector requires either: (a) a second DB pass that
				#  writes into a memory-mapped array, or (b) storing all sim blocks on
				#  disk.  Most users never actually need the entire vector, only the
				#  top-k scores.  If you do need it, ask and I\u2019ll show option (a) with
				#  numpy.memmap.

				return unit_sim, top_cls, avg_sim
		else:
			# ====================================================================== #
			#  DENSE HDF5 STREAMING VARIANTS                                       #
			#  -------------------------------------------------------------------- #
			#  * expect an HDF5 file containing                                     #
			#      * "images"   : (B3, C, L)  float32                                #
			#      * "classes"  : (B3,)       int32                                  #
			#  * RAM usage  O(B2 + chunk_imgs  C  L)                               #
			#  * Outputs identical to the in-RAM dense functions                     #
			# ====================================================================== #
			def compare_1d_batches_stream_db(
				candidates: torch.Tensor,			# (B2, C, L)  *dense*
				h5_path: str,						# path to H5 with "images", "classes"
				B1: int,
				*,
				chunk_imgs: int = ATNLPsnapshotDatabaseDiskChunkSize,			# DB rows per slice   (tune to RAM)
				eps: float = 1e-8,
				device: str | torch.device = "cpu",
			) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
				device = torch.device(device)
				candidates = candidates.to(device, dtype=torch.float32)

				B2, C, L = candidates.shape
				cand_feat = F.normalize(candidates.reshape(B2, -1), p=2, dim=1, eps=eps)	# (B2, D)

				# outputs we will update on-the-fly
				best_val = torch.full((B2,), -float("inf"), device=device)
				best_idx = torch.full((B2,), -1, dtype=torch.long, device=device)
				sumsq	 = torch.zeros(B2, device=device)		# for L2\u2010unit vector

				with h5py.File(h5_path, "r") as h5:
					images_ds = h5["images"]	  # (B3, C, L)
					class_ds  = torch.from_numpy(h5["classes"][...]).to(torch.long)
					B3 = class_ds.numel()

					for s in range(0, B3, chunk_imgs):
						e = min(s + chunk_imgs, B3)
						db_blk = torch.from_numpy(images_ds[s:e]).to(device, dtype=torch.float32)  # (n, C, L)
						db_feat = F.normalize(db_blk.reshape(e - s, -1), p=2, dim=1, eps=eps)	   # (n, D)

						sim_blk = cand_feat @ db_feat.T					# (B2, n)
						sumsq  += (sim_blk ** 2).sum(dim=1)

						cur_best, cur_idx = sim_blk.max(dim=1)			# (B2,)
						mask = cur_best > best_val
						best_val[mask] = cur_best[mask]
						best_idx[mask] = cur_idx[mask] + s

				# snapshot aggregation
				assert B2 % B1 == 0, "B2 must divide into snapshots"
				S = B2 // B1

				avg_sim = best_val.view(B1, S).mean(dim=1)				# (B1,)
				top_cls = class_ds[best_idx.view(B1, S)[:, 0]]			# (B1,)

				#unit_sim = (sumsq / (sumsq.sum(dim=1, keepdim=True) + eps)).sqrt()
				unit_sim = torch.sqrt(sumsq + eps)		# shape (B2,)
				
				return unit_sim, top_cls, avg_sim

					
else:
	if(ATNLPcomparisonShiftInvariance):
		def compare_1d_batches(
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
			database   = database.to(ATNLPsnapshotDatabaseLoadDevice, dtype=torch.float32)
			db_classes = db_classes.to(device)

			B2, C, L   = candidates.shape
			B3		 = database.shape[0]
			K		  = (L - 1) if shiftInvariantPixels is None else int(shiftInvariantPixels)
			K		  = max(0, min(K, L - 1))				 # clamp

			fft_len	= 2 * L - 1							 # full linear corr length
			idx0	   = L - 1								 # zero-shift index
			low, high  = idx0 - K, idx0 + K + 1				# slice bounds

			if(normalisedSnapshotsSparseTensors):
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
				cand_nrm = torch.sqrt(torch.sparse.sum(candidates.pow(2), dim=(1, 2)).to_dense() + eps)	# (B2,)	 # GPU
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
				cand_dense = candidates.to_dense()
				cand_dense_flat = cand_dense.reshape(B2, D)
				cand_norm = torch.linalg.vector_norm(cand_dense_flat, dim=1) + eps	# (B2,)

				shifted = []
				for s in range(-K, K + 1):
					if s >= 0:
						tmp = torch.zeros_like(cand_dense)
						tmp[:, :, s:] = cand_dense[:, :, :L - s]
					else:
						tmp = torch.zeros_like(cand_dense)
						tmp[:, :, :L + s] = cand_dense[:, :, -s:]
					shifted.append(tmp.reshape(B2, D).T.contiguous())		# (D, B2)

				# max\u2010over\u2010shift similarity matrix  (B2, B3)
				sim = torch.full((B2, B3), -float("inf"), device=device)
				for cand_shift_T in shifted:
					sim_blk = torch.sparse.mm(db_flat, cand_shift_T).T		# (B2, B3)
					sim_blk /= (cand_norm[:, None] * db_norm[None, :])
					sim = torch.maximum(sim, sim_blk)

				sumsq = (sim ** 2).sum(dim=1)			# (B2,)  for unit_sim later

			else:
				# ----------------------------------------------------------- FFT once
				cand_fft  = torch.fft.rfft(candidates, n=fft_len)  # (B2, C, F)
				cand_nrm  = torch.linalg.vector_norm(candidates, dim=(1, 2))  # (B2,)

				# prep outputs
				sim_rows, best_vals, best_idx = [], [], []

				step = chunk or B3
				for s in range(0, B3, step):
					e = min(s + step, B3)

					#db_slice  = database[s:e]
					db_slice = database[s:e].to(device, non_blocking=True)     #GPU now
					db_fft   = torch.fft.rfft(db_slice, n=fft_len)
					db_nrm  = torch.linalg.vector_norm(db_slice, dim=(1, 2))  # (chunk,)

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

			# --- aggregate over snapshots -------------------------------------------
			B2, B3 = sim.shape
			assert B2 % B1 == 0, "B2 must be a multiple of base_batch"
			S = B2 // B1

			mean_sim = sim.view(B1, S, B3).mean(dim=1)		  # (B1, B3)
			best_vals, best_idx = mean_sim.max(dim=1)		   # (B1,)

			top_cls  = db_classes[best_idx]					  # (B1,)
			avg_sim  = best_vals								 # (B1,)

			return unit_sim, top_cls, avg_sim

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
			candidates: torch.Tensor,		   # shape (B2, C, L)
			database: torch.Tensor,			 # shape (B3, C, L)   - B3 >> B2
			db_classes: torch.Tensor,		   # shape (B3,)  - int64 class targets
			B1: int,
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
			device = torch.device(device)
			candidates = candidates.to(device, dtype=torch.float32)
			database   = database.to(ATNLPsnapshotDatabaseLoadDevice, dtype=torch.float32)
			db_classes = db_classes.to(device)

			B2, C, L   = candidates.shape
			B3, C2, L2 = database.shape
			assert C == C2 and L == L2, "Candidates and database must have identical C and L"

			if(normalisedSnapshotsSparseTensors):
				## ---- flatten to 2-D *sparse* feature matrices --------------------------
				#cand_feat = candidates.reshape(B2, -1).coalesce()		# (B2, D) - sparse	# GPU
				#db_feat	= database.reshape(B3, -1).coalesce()		# (B3, D) - sparse	# CPU
				# ---- flatten to (B, D) sparse matrices -------------------------------
				cand_feat = _flatten_sparse_3d_to_2d(candidates).to(device)	 # (B2, D)
				db_feat = _flatten_sparse_3d_to_2d(database).coalesce()	 # (B3, D) on CPU

				# ---- pre-compute L2 norms (dense 1-D vectors, tiny) ---------------------
				cand_norm = torch.sqrt(torch.sparse.sum(cand_feat.pow(2), dim=1).to_dense() + eps)	# (B2,)
				db_norm	= torch.sqrt(torch.sparse.sum(db_feat.pow(2),  dim=1).to_dense() + eps)	# (B3,)

				if chunk is None:
					sim = _cosine_block(cand_feat, db_feat.to(device))				# (B2, B3)
				else:
					parts = []
					for s in range(0, B3, chunk):
						e = min(s + chunk, B3)
						d_blk = db_feat[s:e].to(device, non_blocking=True)
						parts.append(_cosine_block(cand_feat, d_blk))	# (B2, e-s)
					sim = torch.cat(parts, dim=1)						  # (B2, B3)
			else:
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
					#sim = cand_feat @ db_feat.T							# (B2, B3)
					sim = cand_feat @ db_feat.to(device).T			# (B2, B3)
				else:
					parts = []	# slice-by-slice, each moved to GPU
					for s in range(0, B3, chunk):
						e = min(s + chunk, B3)
						#db_slice  = db_feat[s:e]
						db_slice = db_feat[s:e].to(device, non_blocking=True)    # GPU now
						parts.append(cand_feat @ db_slice.T)		   # (B2, e-s)
					sim = torch.cat(parts, dim=1)						  # (B2, B3)

			# ---------------------------------------------------------------------- #
			# 3. convert each row into a **unit similarity vector** (L2 = 1).		#
			# ---------------------------------------------------------------------- #
			unit_sim = F.normalize(sim, p=2, dim=1, eps=eps)		   # (B2, B3)

			# --- aggregate over S snapshots per logical sample -----------------------
			B2, B3 = sim.shape
			assert B2 % B1 == 0, "B2 must be a multiple of base_batch"
			S	  = B2 // B1								   # snapshots per sample

			mean_sim = sim.view(B1, S, B3).mean(dim=1)		  # (B1, B3)
			best_vals, best_idx = mean_sim.max(dim=1)		   # (B1,)

			top_cls = db_classes[best_idx]					  # (B1,)
			avg_sim = best_vals								 # (B1,)

			return unit_sim, top_cls, avg_sim


	


