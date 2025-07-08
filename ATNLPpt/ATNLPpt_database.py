"""ATNLPpt_database.py

# Author:
Richard Bruce Baxter - Copyright (c) 2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ANNpt_main.py

# Usage:
see ANNpt_main.py

# Description:
ATNLPpt database

"""

from ANNpt_globalDefs import *
import torch
import numpy as np
if(ATNLPsnapshotDatabaseDisk):
	if(not ATNLPsnapshotDatabaseDiskSetSize):
		import h5py

if(ATNLPsnapshotDatabaseDisk):
	if(ATNLPnormalisedSnapshotsSparseTensors):
		# ------------------------------------------------------------------
		#  S P A R S E   H D F 5   D A T A B A S E   (disk snapshot)
		# ------------------------------------------------------------------
		if ATNLPsnapshotDatabaseDisk:		# one unified sparse path
			class H5DBWriter:
				"""
				Incrementally builds a CSR-style sparse COO store:
				  * values  : (nnz,)
				  * indices : (3, nnz)   [img, channel, pos]
				  * img_ptr : (num_imgs + 1,)  CSR pointer
				"""
				def __init__(self, C: int, L: int, path=snapshotDatabaseName, chunk_nnz: int = ATNLPsnapshotDatabaseDiskChunkSize):

					self.C = C
					self.L = L
					self.h5 = h5py.File(path, "w")

					self.val_ds = self.h5.create_dataset("values",  shape=(0,), maxshape=(None,), dtype="float32", chunks=(chunk_nnz,))
					self.idx_ds = self.h5.create_dataset("indices", shape=(3, 0), maxshape=(3, None), dtype="int32",  chunks=(3, chunk_nnz))
					# CSR pointer: img_ptr[i] = start of i-th image in values/indices
					self.ptr_ds = self.h5.create_dataset("img_ptr", shape=(1,), maxshape=(None,), dtype="int64", chunks=(chunk_nnz//C,))
					self.ptr_ds[0] = 0

					self.cls_ds = self.h5.create_dataset("classes", shape=(0,), maxshape=(None,), dtype="int32", chunks=(chunk_nnz//C,))

					self.nnz = 0					  # running non-zero counter

				def _flatten_idx(self, coords):
					"""
					coords : (2, nnz_img)  [channel, pos]  with 0 \u2264 channel < C, 0 \u2264 pos < L
					returns : (nnz_img,) linear index	channel * L + pos
					"""
					return coords[0] * self.L + coords[1]

				def add_batch(self, images: torch.Tensor, classes: torch.Tensor):
					"""
					images : (B1, C, L) *sparse* or *dense*
					classes: (B1,)
					"""
					B1 = images.size(0)
					if images.dim() != 3 or images.shape[1:] != (self.C, self.L):
						raise ValueError("shape mismatch")

					cls_np = classes.detach().to("cpu").to(torch.int32).numpy()
					# grow class dataset
					new_cls_size = self.cls_ds.shape[0] + B1
					self.cls_ds.resize((new_cls_size,))
					self.cls_ds[-B1:] = cls_np

					# process each image individually
					for b in range(B1):
						img = images[b].cpu()
						if not img.is_coalesced(): # guarantee a well-formed COO before .indices()
							img = img.coalesce()
						coords = img.indices()		# (2, nnz_img)  [channel, pos]
						vals   = img.values()		 # (nnz_img,)

						nnz_img = vals.size(0)
						if nnz_img == 0:
							# still need a pointer advance
							self.ptr_ds.resize((self.ptr_ds.shape[0] + 1,))
							self.ptr_ds[-1] = self.nnz
							continue

						# 1-D flatten channel & pos, prepend image id
						lin_idx = self._flatten_idx(coords).to(torch.int32)
						img_idx = torch.full_like(lin_idx, fill_value=b + (new_cls_size - B1))
						full_idx = torch.vstack((img_idx, coords))   # shape (3, nnz_img)

						# extend value / index datasets
						new_nnz = self.nnz + nnz_img
						self.val_ds.resize((new_nnz,))
						self.idx_ds.resize((3, new_nnz))
						self.val_ds[self.nnz:new_nnz] = vals.cpu().numpy().astype(np.float32)
						self.idx_ds[:, self.nnz:new_nnz] = full_idx.cpu().numpy()
						self.nnz = new_nnz

						# advance CSR pointer
						self.ptr_ds.resize((self.ptr_ds.shape[0] + 1,))
						self.ptr_ds[-1] = self.nnz

				def close(self):
					self.h5.flush()
					self.h5.close()

			# ------------------------------------------------------------------
			#  Finalise (load-back) - returns a *single* sparse tensor (B3, C, L)
			# ------------------------------------------------------------------
			def finaliseTrainedSnapshotDatabase(self):
				with h5py.File(snapshotDatabaseName, "r") as h5:
					values = torch.from_numpy(np.asarray(h5["values"],  dtype=np.float32))
					indices = torch.from_numpy(np.asarray(h5["indices"], dtype=np.int32))  # (3, nnz)
					cls	 = torch.from_numpy(np.asarray(h5["classes"], dtype=np.int32))

					B3  = cls.numel()
					C   = int(indices[1].max().item() + 1) if values.numel() else self.C
					L   = int(indices[2].max().item() + 1) if values.numel() else self.L

					size = (B3, C, L)
					self.database = torch.sparse_coo_tensor(indices.to(torch.int64), values, size=size, dtype=torch.float32, device="cpu",).coalesce()
					self.db_classes = cls
	else:
		if(ATNLPsnapshotDatabaseDiskSetSize):
			class DBWriter:
				"""
				Pre-allocates a memory-mapped file and sequentially fills it
				with (C, L) tensors and integer class targets generated during
				the training loop.
				"""
				def __init__(self, total_images: int, C: int, L: int, db_path=snapshotDatabaseNameFloat32, cls_path=snapshotDatabaseNameInt32):
					self.db = np.memmap(db_path, mode="w+", dtype="float32", shape=(total_images, C, L))
					self.cls = np.memmap(cls_path, mode="w+", dtype="int32", shape=(total_images,))
					self.ptr = 0						  # write pointer
					self.C, self.L = C, L
				def add_batch(self, images: torch.Tensor, classes: torch.Tensor):
					"""
					images : (B1, C, L)  float or uint8
					classes: (B1,)	   int64 / int32
					"""
					B1 = images.size(0)
					if images.dim() != 3 or images.shape[1:] != (self.C, self.L):
						raise ValueError("shape mismatch")

					if self.ptr + B1 > len(self.db):
						raise RuntimeError("pre-allocated DB is full")

					# move to CPU & dtype float32
					imgs_cpu = images.detach().cpu().to(torch.float32).numpy()
					cls_cpu  = classes.detach().cpu().to(torch.int32).numpy()

					self.db [self.ptr : self.ptr+B1] = imgs_cpu
					self.cls[self.ptr : self.ptr+B1] = cls_cpu
					self.ptr += B1				   # advance pointer
				def flush(self):
					self.db.flush(); self.cls.flush()
			def finaliseTrainedSnapshotDatabase(self):
				self.builder.flush()		 # final flush
				# (optional) call builder.flush() every N iterations

				# Mem-mapped variant
				self.database = torch.from_numpy(np.memmap(snapshotDatabaseNameFloat32, mode="r", dtype="float32", shape=(B3, C, L)))
				self.db_classes = torch.from_numpy(np.memmap(snapshotDatabaseNameInt32, mode="r", dtype="int32", shape=(B3,)))
		else:
			class H5DBWriter:
				def __init__(self, C, L, path=snapshotDatabaseName, chunks=(ATNLPsnapshotDatabaseDiskChunkSize, 1, 1)):
					self.h5 = h5py.File(path, "w")
					maxshape = (None, C, L)		  # unlimited in first dim
					self.img_ds = self.h5.create_dataset("images", shape=(0,  C, L),  maxshape=maxshape, dtype="float32", chunks=chunks)
					self.cls_ds = self.h5.create_dataset("classes", shape=(0,), maxshape=(None,), dtype="int32", chunks=(chunks[0],))

				def add_batch(self, images, classes):
					B1 = images.size(0)
					new_size = self.img_ds.shape[0] + B1
					self.img_ds.resize(new_size, axis=0)
					self.cls_ds.resize(new_size, axis=0)

					self.img_ds[-B1:] = images.detach().cpu().to(torch.float32).numpy()
					self.cls_ds[-B1:] = classes.detach().cpu().to(torch.int32).numpy()

				def close(self): 
					self.h5.close()	
			def finaliseTrainedSnapshotDatabase(self):
				with h5py.File(snapshotDatabaseName, "r") as h5:
					# 1 get HDF5 dataset handles
					img_ds = h5["images"]	 # shape (B3, C, L), float32
					cls_ds = h5["classes"]	# shape (B3,),	 int32

					# 2 expose as NumPy array views (zero-copy)
					np_imgs = np.asarray(img_ds, dtype=np.float32)   # still backed by the file
					np_cls  = np.asarray(cls_ds, dtype=np.int32)

				# 3  convert to Torch tensors that share the same memory*
				self.database = torch.from_numpy(np_imgs)   # (B3, C, L)
				self.db_classes = torch.from_numpy(np_cls)	# (B3,)

elif(ATNLPsnapshotDatabaseRamStatic):
	def finaliseTrainedSnapshotDatabase(self):
		if(ATNLPnormalisedSnapshotsSparseTensors):
			# Channel/length are constant across images (each list item is a single (C,L) snapshot)
			C, L = self.imgs_list[0].shape

			indices_cat, values_cat = [], []

			for b, img in enumerate(self.imgs_list):			# one snapshot per list item

				assert img.shape == (C, L), "shape mismatch in imgs_list"

				idx2d = img.indices()							# (2, nnz)
				val2d = img.values()							# (nnz,)

				batch_row = torch.full((1, idx2d.size(1)), b, dtype=idx2d.dtype, device=idx2d.device)
				idx3d = torch.cat([batch_row, idx2d], dim=0)	# (3, nnz)

				indices_cat.append(idx3d)
				values_cat.append(val2d)

			# Assemble final sparse (B3,C,L)
			indices_all = torch.cat(indices_cat, dim=1)
			values_all  = torch.cat(values_cat, dim=0)

			self.database = torch.sparse_coo_tensor(
				indices_all, values_all,
				size=(len(self.imgs_list), C, L),
				dtype=values_all.dtype, device=values_all.device
			).coalesce()
		else:
			self.database = torch.stack(self.imgs_list)		# (B3, C, L)
		self.db_classes = torch.stack(self.cls_list)		 # (B3,)
		self.database = self.database.to(ATNLPsnapshotDatabaseLoadDevice)
		self.db_classes = self.db_classes.to(ATNLPsnapshotDatabaseLoadDevice)
elif(ATNLPsnapshotDatabaseRamDynamic):
	def finaliseTrainedSnapshotDatabase(self):
		pass
