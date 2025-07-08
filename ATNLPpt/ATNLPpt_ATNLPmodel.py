"""ATNLPpt_ATNLPmodel.py

# Author:
Richard Bruce Baxter - Copyright (c) 2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ANNpt_main.py

# Usage:
see ANNpt_main.py

# Description:
ATNLPpt Axis Transformation Natural Language Processing (ATNLP) network model

"""

import torch
from torch import nn
from typing import List, Optional, Tuple
from ANNpt_globalDefs import *
import ATNLPpt_ATNLPmodelContinuousVarEncoding
import ATNLPpt_keypoints
import ATNLPpt_normalisation
import ATNLPpt_comparison
import ATNLPpt_database
import hashlib
from collections import defaultdict

	
class ATNLPconfig():
	def __init__(self, batchSize, numberOfLayers, hiddenLayerSize, inputLayerSize, outputLayerSize, numberOfFeatures, numberOfClasses, fieldTypeList):
		self.batchSize = batchSize
		self.numberOfLayers = numberOfLayers
		self.hiddenLayerSize = hiddenLayerSize
		self.inputLayerSize = inputLayerSize
		self.outputLayerSize = outputLayerSize
		self.numberOfFeatures = numberOfFeatures
		self.numberOfClasses = numberOfClasses
		self.fieldTypeList = fieldTypeList

class Loss:
	def __init__(self, value=0.0):
		self._value = value

	def item(self):
		return self._value

if(ATNLPsnapshotDatabaseDisk):
	def initialiseSnapshotDatabaseWriter():
		global snapshotDatabaseWriter
		if(ATNLPsnapshotDatabaseDiskSetSize):
			snapshotDatabaseWriter = ATNLPpt_database.DBWriter(datasetTrainRows*B2train, C, L2)
		else:
			snapshotDatabaseWriter = ATNLPpt_database.H5DBWriter(C, L2)
	
# -------------------------------------------------------------
# Core network module
# -------------------------------------------------------------

class ATNLPmodel(nn.Module):
	"""Custom neural network implementing the ATNLP specification."""

	def __init__(self, config: ATNLPconfig) -> None:
		super().__init__()

		# -----------------------------
		# Public config
		# -----------------------------
		self.config = config

		# -----------------------------
		# database declaration
		# -----------------------------
		if(ATNLPsnapshotDatabaseDisk):
			initialiseSnapshotDatabaseWriter()
		elif(ATNLPsnapshotDatabaseRamDynamic):
			self.databaseRamDynamicInitialised = False
		elif(ATNLPsnapshotDatabaseRamStatic):
			self.imgs_list, self.cls_list = [], []
			
	def deriveCurrentBatchSize(self, batch):
		(x, y) = batch
		if useNLPDatasetMultipleTokenisation:
			if useNLPcharacterInput:
				currentBatchSize = x["char_input_ids"].shape[0]
			else:
				currentBatchSize = x["bert_input_ids"].shape[0]
		else:
			currentBatchSize = x.shape[0]
		return currentBatchSize
		
	# ---------------------------------------------------------
	# Forward pass
	# ---------------------------------------------------------
		
	#@torch.no_grad()
	@torch.inference_mode()
	def forward(self, trainOrTest: bool, x: torch.Tensor, y: Optional[torch.Tensor] = None, optim=None, l=None, batchIndex=None, fieldTypeList=None) -> Tuple[torch.Tensor, torch.Tensor]:
		"""Forward pass.
		
		Args:
			trainOrTest: True=> train mode; False=> inference.
			x: dict of different data types;
				each is a (batch, length) feature tensor.
				if(useNLPcharacterInput):
					"char_input_ids" : (B, Lc),
				else:
					"bert_input_ids" : (B, Lb),
					"bert_offsets"   : (B, Lb, 2),
				"spacy_input_ids": (B, Ls),
				"spacy_pos"	  : (B, Ls),
				"spacy_offsets"  : (B, Ls, 2),
			y: None
				dynamically generated; (batch,) int64 labels when trainOrTest==True.
				
		Returns:
			predictions, outputActivations (both shape (batch, classes)).
		"""
		
		seq_input = self.generateSequenceInput(x)			
		B1 = self.batchSize = seq_input.shape[0]
		device = seq_input.device

		if(useSlidingWindow):
			if(useNLPDatasetPaddingMask):	
				non_pad	= (seq_input != NLPcharacterInputPadTokenID)		# [B1, L1]
				if not pt.any(non_pad):
					return
				lengths	= non_pad.sum(-1)							# [B1]
				max_len = int(lengths.max().item())
				numSubsamples = max(1, max_len - 1)				# predict *next* token only
				extra = contextSizeMax - lengths					# [B1]
			else:
				numSubsamples = contextSizeMax
		else:
			numSubsamples = 1
	
		# -----------------------------
		# Continuous var encoding as bits
		# -----------------------------
		seq_input_encoded = ATNLPpt_ATNLPmodelContinuousVarEncoding.encodeContinuousVarsAsBits(self, seq_input, ATNLPcontinuousVarEncodingNumBits).float()	#NLPcharacterInputSetLen	#[batchSize, sequenceLength*numBits]
		#seq_token_tensor = encodeContinuousVarsAsBits(self, x['bert_input_ids'], bertNumberTokenTypes).to(torch.int8)	#[batchSize, sequenceLength*numBits]

		seq_input_encoded = seq_input_encoded.reshape(B1, L1, C)	 #shape (B1, L1, C)
		seq_input_encoded = seq_input_encoded.permute(0, 2, 1)	 #shape (B1, C, L1)

		spacy_pos = x['spacy_pos']
		spacy_offsets = x['spacy_offsets']
		kp_indices_batch, kp_meta_batch = ATNLPpt_keypoints.build_keypoints(spacy_pos, spacy_offsets)
		if(debugATNLPkeypoints):
			print("kp_indices_batch = ", kp_indices_batch)
		
		numSubsamplesWithKeypoints = 0
		accuracyAllWindows = 0
		#print("numSubsamples = ", numSubsamples)
		for slidingWindowIndex in range(numSubsamples):
			if(debugSequentialLoops):
				print("\n************************** slidingWindowIndex = ", slidingWindowIndex)

			# -----------------------------
			# Transformation (normalisation)
			# -----------------------------
			if(trainOrTest):
				mode=keypointModeTrain
			else:
				mode=keypointModeTest
			
			last_token_idx = slidingWindowIndex
			normalisedSnapshots = ATNLPpt_normalisation.normalise_batch(seq_input_encoded, x['spacy_pos'], x['spacy_offsets'], last_token_idx, mode=mode, r=r, q=q, L2=L2, kp_indices_batch=kp_indices_batch, kp_meta_batch=kp_meta_batch)
			if(debugATNLPnormalisation):
				print("seq_input_encoded = ", seq_input_encoded)	
				print("normalisedSnapshots = ", normalisedSnapshots)
				print("normalisedSnapshots.count_nonzero() = ", normalisedSnapshots.count_nonzero())
			
			normalisedSnapshots, validSnapshotFound = self.removeInvalidNormalisedSnapshots(normalisedSnapshots, B1)
			if(validSnapshotFound):
				#print("validSnapshotFound")
				numSubsamplesWithKeypoints += 1
			else:
				continue
			
			if(ATNLPnormalisedSnapshotsSparseTensors):
				normalisedSnapshots = normalisedSnapshots.to_sparse_coo()
				normalisedSnapshots = normalisedSnapshots.coalesce()	

			#sanity checks;
			if(debugATNLPsnapshotDuplicates):
				if(ATNLPsnapshotDatabaseRamStatic):
					for dsnap in self.imgs_list:
						#print("dsnap.shape = ", dsnap.shape)
						#print("normalisedSnapshots[0].shape = ", normalisedSnapshots[0].shape)
						if(torch.equal(dsnap, normalisedSnapshots[0])):
							print("exact duplicate snapshot being added to database")
							dsnapSparse = dsnap.to_sparse_coo()
							normalisedSnapshotSparse = normalisedSnapshots[0].to_sparse_coo()
							print("dsnapSparse = ", dsnapSparse)
							print("normalisedSnapshotSparse = ", normalisedSnapshotSparse)

			y, classTargets = self.generateClassTargets(slidingWindowIndex, normalisedSnapshots, B1, seq_input)
			
			if(trainOrTest and not generateConnectionsAfterPropagating):	#debug only
				self.addNormalisedSnapshotToDatabase(normalisedSnapshots, classTargets)
			
			#sanity checks;
			if(debugATNLPsnapshotDuplicates):
				if(ATNLPsnapshotDatabaseRamDynamic and self.databaseRamDynamicInitialised):
					self.count_exact_duplicates()

			# -----------------------------
			# Prediction
			# -----------------------------
			if(not trainOrTest or (ATNLPsnapshotDatabaseRamDynamic and self.databaseRamDynamicInitialised)):
				if(ATNLPsnapshotDatabaseDiskCompareChunks):
					unit_sim, top_cls, avg_sim = ATNLPpt_comparison.compare_1d_batches_stream_db(normalisedSnapshots, snapshotDatabaseName, B1, chunk_nnz=ATNLPsnapshotDatabaseDiskCompareChunksSize, device=device, shiftInvariantPixels=ATNLPcomparisonShiftInvariancePixels)
				else:
					#print("self.database = ", self.database)	#does not support useLovelyTensors
					#print("self.db_classes = ", self.db_classes)	#does not support useLovelyTensors
					unit_sim, top_cls, avg_sim = ATNLPpt_comparison.compare_1d_batches(normalisedSnapshots, self.database, self.db_classes, B1, chunk=ATNLPsnapshotCompareChunkSize, device=device, shiftInvariantPixels=ATNLPcomparisonShiftInvariancePixels)
				predictions = top_cls
				
				# count how many are exactly correct
				if(useNLPDatasetPaddingMask):
					#print("y = ", y)
					valid_mask = (y != NLPcharacterInputPadTokenID)
					valid_count = valid_mask.sum().item()
					if valid_count > 0:
						correct = ((predictions == y) & valid_mask).sum().item()
						accuracy = correct / valid_count
					else:
						accuracy = 0.0
				else:
					correct = (predictions == y).sum().item()
					accuracy = correct / y.size(0)
				accuracyAllWindows += accuracy
				
				#if(debugATNLPnormalisation):
				print("accuracy = ", accuracy)
					
			# -----------------------------
			# Train
			# -----------------------------
			if(trainOrTest and generateConnectionsAfterPropagating):
				self.addNormalisedSnapshotToDatabase(normalisedSnapshots, classTargets)
		
		accuracy = accuracyAllWindows / numSubsamplesWithKeypoints
		loss = Loss(0.0)
		
		return loss, accuracy

	def removeInvalidNormalisedSnapshots(self, normalisedSnapshots, B1):
		#remove invalid normalisedSnapshots (without keypoints);
		B2 = normalisedSnapshots.shape[0]
		S = B2 // B1
		normalisedSnapshots = normalisedSnapshots.reshape(B1, S, C, L2)
		normalisedSnapshots = normalisedSnapshots.permute(1, 0, 2, 3)        # now (S, B1, C, L2)
		SnonZero = torch.count_nonzero(normalisedSnapshots, dim=(1,2,3))   # shape (S,)
		mask = SnonZero > 0        # (S,) boolean
		Snew = mask.sum().item()
		#print("Snew = ", Snew)
		normalisedSnapshots = normalisedSnapshots[mask]         # shape (S', B1, C, L2)
		normalisedSnapshots = normalisedSnapshots.permute(1, 0, 2, 3)        # now (B1, S', C, L2)
		normalisedSnapshots = normalisedSnapshots.reshape(B1*Snew, C, L2)
		if(normalisedSnapshots.count_nonzero() == 0):
			validSnapshotFound = False
		else:
			validSnapshotFound = True
		return normalisedSnapshots, validSnapshotFound
				
	def addNormalisedSnapshotToDatabase(self, normalisedSnapshots, classTargets):
		if(ATNLPsnapshotDatabaseDisk):
			snapshotDatabaseWriter.add_batch(normalisedSnapshots, classTargets)
		elif(ATNLPsnapshotDatabaseRamDynamic):
			if(self.databaseRamDynamicInitialised):
				self.database = torch.cat((self.database, normalisedSnapshots.to(ATNLPsnapshotDatabaseLoadDevice)), dim=0)		# (B3, C, L)
				self.db_classes = torch.cat((self.db_classes, classTargets.to(ATNLPsnapshotDatabaseLoadDevice)), dim=0)		 # (B3,)
			else:
				self.databaseRamDynamicInitialised = True
				self.database = normalisedSnapshots.to(ATNLPsnapshotDatabaseLoadDevice)
				self.db_classes = classTargets.to(ATNLPsnapshotDatabaseLoadDevice)
			if(ATNLPnormalisedSnapshotsSparseTensors):
				self.database = self.database.coalesce()
		elif(ATNLPsnapshotDatabaseRamStatic):
			S = normalisedSnapshots.shape[0]
			for s in range(S):
				normalisedSnapshot = normalisedSnapshots[s]
				normalisedSnapshot = normalisedSnapshot.coalesce()
				classTarget = classTargets[s]
				self.imgs_list.append(normalisedSnapshot)
				self.cls_list.append(classTarget)
					
	def generateClassTargets(self, slidingWindowIndex, normalisedSnapshots, B1, seq_input):
		#for now just predict final character in sequence window
		#FUTURE: predict Bert tokens rather than individual characters)
		
		B2 = normalisedSnapshots.shape[0]
		y = seq_input[:, slidingWindowIndex] #shape = B1	
		reps_per_elem = B2 // B1
		#print("B2 = ", B2)
		#print("B1 = ", B1)
		classTargets = y.repeat_interleave(reps_per_elem) #shape = B2
		return y, classTargets
				
	def finaliseTrainedSnapshotDatabase(self):
		if(ATNLPsnapshotDatabaseDiskCompareChunks):
			pass	#h5 database is read directly during comparison
		else:	
			ATNLPpt_database.finaliseTrainedSnapshotDatabase(self)	#generate self.database tensor from database for comparison
	
	def generateSequenceInput(self, x):
		if(useNLPcharacterInput):
			seq_input = x['char_input_ids']	# Tensor [batchSize, sequenceLength]
		else:
			"""
			Expand word-piece IDs so every character position gets the ID of the token that covers it.

			Returns
				ids_per_char : (B, Lc)  int64
			"""
			
			bert_input_ids = x['bert_input_ids']	# (B, Lb)	 int64
			bert_offsets = x['bert_offsets']	#(B, Lb, 2)  int64
			Lc = contextSizeMax
			pad_id = NLPcharacterInputPadTokenID
			
			# ------------------------------------------------------------------
			B, Lb = bert_input_ids.shape
			device = bert_input_ids.device

			# 1) Boolean mask saying \u201ctoken j covers char position c\u201d
			pos = pt.arange(Lc, device=device).view(1, Lc, 1)	  # (1,Lc,1)
			start, end = bert_offsets[..., 0], bert_offsets[..., 1]	   # (B,Lb)
			mask = (pos >= start.unsqueeze(1)) & (pos < end.unsqueeze(1))  # (B,Lc,Lb)

			# zero-length spans (CLS/SEP/PAD) have end==start \u2192 never True
			# ------------------------------------------------------------------
			# 2) For each character pick the first (and only) covering token
			token_idx = mask.float().argmax(dim=2)					   # (B,Lc), int64

			# 3) Detect char positions not covered by any token
			no_cover = ~mask.any(dim=2)								 # (B,Lc)
			token_idx[no_cover] = 0										  # safe filler

			# 4) Gather the BERT IDs and pad the gaps
			ids_per_char = bert_input_ids.gather(1, token_idx)			  # (B,Lc)
			ids_per_char[no_cover] = pad_id
			seq_input = ids_per_char
			
			#print("seq_input.shape = ", seq_input.shape)
	
		return seq_input

	if(debugATNLPsnapshotDuplicates):
		def count_exact_duplicates(self) -> int:
			"""
			Returns the number of database rows that share the *same* feature vector
			but carry *different* class labels.

			Parameters
			----------
			db		  : (B3, C, L)  dense (Strided) **or** sparse COO tensor
			db_classes  : (B3,)	   int64 / int32
			"""
			db = self.database
			db_classes = self.db_classes

			B3 = db.shape[0]

			# -------- 1. build a stable hash per row --------------------------------
			row_hashes = [""] * B3

			if db.is_sparse:
				# Flatten once for indexing
				flat = db.coalesce()
				idx, val = flat.indices(), flat.values()	  # (3, nnz), (nnz,)

				for b in range(B3):
					mask = idx[0] == b
					h = hashlib.md5()
					h.update(idx[1:, mask].t().contiguous().cpu().numpy().tobytes())
					h.update(val[mask].cpu().numpy().tobytes())
					row_hashes[b] = h.hexdigest()
			else:  # dense (\u201cStrided\u201d) layout
				for b in range(B3):
					h = hashlib.md5(db[b].cpu().numpy().tobytes()).hexdigest()
					row_hashes[b] = h

			# -------- 2. group rows by hash and look for label conflicts ------------
			buckets = defaultdict(list)
			for i, h in enumerate(row_hashes):
				buckets[h].append(i)

			conflicts = 0
			for rows in buckets.values():
				if len(rows) > 1:
					labels = db_classes[rows]
					if labels.unique().numel() > 1:
						conflicts += len(rows)

			return conflicts



