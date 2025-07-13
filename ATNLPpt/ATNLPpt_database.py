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
	if(ATNLPsnapshotDatabaseDiskCompareChunks):
		import h5py
if(ATNLPindexDatabaseByClassTarget):		
	import ATNLPpt_sparseTensors

if(ATNLPindexDatabaseByClassTarget):
	def createSnapshotDatabaseIndex(self, referenceSetDelimiterID, s):
		self.database[referenceSetDelimiterID][s] = ATNLPpt_sparseTensors.createEmptySparseTensor((numberOfClasses, C, L2))
		self.db_classes[referenceSetDelimiterID][s] = torch.empty((numberOfClasses), dtype=torch.int32, device=device)

	def add_batch(self, referenceSetDelimiterID, s, normalisedSnapshot: torch.Tensor, classTarget: int):
		snapshotDatabaseIndex = self.database[referenceSetDelimiterID][s]
		snapshotDatabaseClassesIndex = self.db_classes[referenceSetDelimiterID][s]
		#print("snapshotDatabaseIndex.device = ", snapshotDatabaseIndex.device)
		#print("normalisedSnapshot.device = ", normalisedSnapshot.device)
		ATNLPpt_sparseTensors.addSparseTensorToFirstDimIndex(snapshotDatabaseIndex, normalisedSnapshot, classTarget) 	#replaceAllSparseTensorElementsAtFirstDimIndex
		snapshotDatabaseClassesIndex[classTarget] = classTarget

	if(ATNLPsnapshotDatabaseDisk):

		def saveSnapshotDatabaseIndex(self, referenceSetDelimiterID, s):
			snapshotDatabaseName = getDatabaseName(referenceSetDelimiterID, s)
			snapshotDatabaseClassName = getDatabaseClassName(referenceSetDelimiterID, s)
			saveTensor(self.database[referenceSetDelimiterID][s], databaseFolderName, snapshotDatabaseName)
			saveTensor(self.db_classes[referenceSetDelimiterID][s], databaseFolderName, snapshotDatabaseClassName)
			self.database[referenceSetDelimiterID][s] = None
			self.db_classes[referenceSetDelimiterID][s] = None
			
		def loadSnapshotDatabaseIndex(self, referenceSetDelimiterID, s):
			snapshotDatabaseName = getDatabaseName(referenceSetDelimiterID, s)
			snapshotDatabaseClassName = getDatabaseClassName(referenceSetDelimiterID, s)
			self.database[referenceSetDelimiterID][s] = loadTensor(databaseFolderName, snapshotDatabaseName)
			self.db_classes[referenceSetDelimiterID][s] = loadTensor(databaseFolderName, snapshotDatabaseClassName)

		def databaseFileExists(folderName, fileName):
			pathName = os.path.join(folderName, fileName)
			return pathExists(pathName)
		
		def pathExists(pathName):
			if(os.path.exists(pathName)):
				return True
			else:
				return False
				
		def saveTensor(tensor, folderName, fileName):
			pathName = os.path.join(folderName, fileName)
			pt.save(tensor, pathName)

		def loadTensor(folderName, fileName):
			pathName = os.path.join(folderName, fileName)
			tensor = pt.load(pathName)
			tensor = tensor.to(device)
			return tensor
		
		def loadSnapshotDatabaseIndices(self, normalisedSnapshots, keypointPairsValid, keypointPairsIndices, kp_meta_batch):
			B1 = normalisedSnapshots.shape[0]
			S = normalisedSnapshots.shape[1]
			for b1 in range(B1):
				for s in range(S):
					if(keypointPairsValid[b1, s]):
						keypointPairsIndexFirst = keypointPairsIndices[b1, s, 0]
						referenceSetDelimiterID = self.getReferenceSetDelimiterID(keypointPairsIndexFirst, kp_meta_batch[b1])
						snapshotDatabaseName = getDatabaseName(referenceSetDelimiterID, s)
						if(self.database[referenceSetDelimiterID][s] == None):
							if(databaseFileExists(databaseFolderName, snapshotDatabaseName)):
								loadSnapshotDatabaseIndex(self, referenceSetDelimiterID, s)
							else:
								createSnapshotDatabaseIndex(self, referenceSetDelimiterID, s)
							
else:			
	if(ATNLPsnapshotDatabaseDisk):
		if(ATNLPsnapshotDatabaseDiskCompareChunks):
			# ------------------------------------------------------------------
			#  S P A R S E   H D F 5   D A T A B A S E   (disk snapshot)
			# ------------------------------------------------------------------
			class H5DBWriter:
				"""
				Incrementally builds a CSR-style sparse COO store:
				  * values  : (nnz,)
				  * indices : (3, nnz)   [img, channel, pos]
				  * img_ptr : (num_imgs + 1,)  CSR pointer
				"""
				def __init__(self, path: str, C: int, L: int, chunk_nnz: int = ATNLPsnapshotDatabaseDiskChunkSize):

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
	elif(ATNLPsnapshotDatabaseRam):
		def finaliseTrainedSnapshotDatabase(self):
			referenceSetDelimiterIDmax = self.getReferenceSetDelimiterIDmax()
			for referenceSetDelimiterID in range(referenceSetDelimiterIDmax):
				for s in range(S):
					if(len(self.normalisedSnapshot_list[referenceSetDelimiterID][s]) > 0):
						normalisedSnapshot_list = self.normalisedSnapshot_list[referenceSetDelimiterID][s]
						classTarget_list = self.classTarget_list[referenceSetDelimiterID][s]

						# Channel/length are constant across images (each list item is a single (C,L) snapshot)
						#print("normalisedSnapshot_list = ", normalisedSnapshot_list)
						C, L = normalisedSnapshot_list[0].shape
						indices_cat, values_cat = [], []
						for b, img in enumerate(normalisedSnapshot_list):			# one snapshot per list item
							assert img.shape == (C, L), "shape mismatch in normalisedSnapshot_list"
							img = img.coalesce()

							idx2d = img.indices()							# (2, nnz)
							val2d = img.values()							# (nnz,)

							batch_row = torch.full((1, idx2d.size(1)), b, dtype=idx2d.dtype, device=idx2d.device)
							idx3d = torch.cat([batch_row, idx2d], dim=0)	# (3, nnz)

							indices_cat.append(idx3d)
							values_cat.append(val2d)

						# Assemble final sparse (B3,C,L)
						indices_all = torch.cat(indices_cat, dim=1)
						values_all  = torch.cat(values_cat, dim=0)

						self.database[referenceSetDelimiterID][s] = torch.sparse_coo_tensor(
							indices_all, values_all,
							size=(len(normalisedSnapshot_list), C, L),
							dtype=values_all.dtype, device=values_all.device
						).coalesce().to(ATNLPsnapshotDatabaseLoadDevice)
						self.db_classes[referenceSetDelimiterID][s] = torch.stack(classTarget_list).to(ATNLPsnapshotDatabaseLoadDevice)		 # (B3,)

def getDatabaseName(referenceSetDelimiterID, s):
	snapshotDatabaseName = snapshotDatabaseNamePrepend + "refSetDelID" + str(referenceSetDelimiterID) + "S" + str(s) + snapshotDatabaseNameExtension
	return snapshotDatabaseName

def getDatabaseClassName(referenceSetDelimiterID, s):
	snapshotDatabaseClassName = snapshotDatabaseNamePrependClass + "refSetDelID" + str(referenceSetDelimiterID) + "S" + str(s) + snapshotDatabaseNameExtension
	return snapshotDatabaseClassName


