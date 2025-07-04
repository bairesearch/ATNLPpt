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
				imgs_cpu = images.detach().to(torch.float32, device="cpu").numpy()
				cls_cpu  = classes.detach().to(torch.int32,  device="cpu").numpy()

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
			def __init__(self, C, L, path=snapshotDatabaseName, chunks=(256, 1, 1)):
				self.h5 = h5py.File(path, "w")
				maxshape = (None, C, L)		  # unlimited in first dim
				self.img_ds = self.h5.create_dataset("images", shape=(0,  C, L),  maxshape=maxshape, dtype="float32", chunks=chunks)
				self.cls_ds = self.h5.create_dataset("classes", shape=(0,), maxshape=(None,), dtype="int32", chunks=(chunks[0],))

			def add_batch(self, images, classes):
				B1 = images.size(0)
				new_size = self.img_ds.shape[0] + B1
				self.img_ds.resize(new_size, axis=0)
				self.cls_ds.resize(new_size, axis=0)

				self.img_ds[-B1:] = images.detach().to(torch.float32, device="cpu").numpy()
				self.cls_ds[-B1:] = classes.detach().to(torch.int32,  device="cpu").numpy()

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
		self.database = torch.cat(self.imgs_list)		# (B3, C, L)
		self.db_classes = torch.cat(self.cls_list)		 # (B3,)
elif(ATNLPsnapshotDatabaseRamDynamic):
	def finaliseTrainedSnapshotDatabase(self):
		pass
