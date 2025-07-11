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
import ANNpt_data
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
			self.initialiseSnapshotDatabaseWriter()
		elif(ATNLPsnapshotDatabaseRam):
			referenceSetDelimiterIDmax = self.getReferenceSetDelimiterIDmax()
			print("referenceSetDelimiterIDmax = ", referenceSetDelimiterIDmax)
			print("S = ", S)
			self.normalisedSnapshot_list = [[[] for _ in range(S)] for _ in range(referenceSetDelimiterIDmax)]
			self.classTarget_list = [[[] for _ in range(S)] for _ in range(referenceSetDelimiterIDmax)]
			self.database = [[None for _ in range(S)] for _ in range(referenceSetDelimiterIDmax)]
			self.db_classes = [[None for _ in range(S)] for _ in range(referenceSetDelimiterIDmax)]
		
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

		kp_indices_batch, kp_meta_batch = ATNLPpt_keypoints.build_keypoints(x['spacy_input_ids'], x['spacy_pos'], x['spacy_tag'], x['spacy_offsets'])
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
			last_token_idx = slidingWindowIndex
			normalisedSnapshots, keypointPairsValid, keypointPairsIndices = ATNLPpt_normalisation.normalise_batch(seq_input_encoded, x['spacy_pos'], x['spacy_offsets'], last_token_idx, mode=keypointMode, r=r, q=q, L2=L2, kp_indices_batch=kp_indices_batch, kp_meta_batch=kp_meta_batch)
			if(debugATNLPnormalisation):
				print("seq_input_encoded = ", seq_input_encoded)	
				print("normalisedSnapshots = ", normalisedSnapshots)
				print("normalisedSnapshots.count_nonzero() = ", normalisedSnapshots.count_nonzero())
			
			if(self.detectValidNormalisedSnapshot(normalisedSnapshots)):
				numSubsamplesWithKeypoints += 1
			else:
				continue
			
			normalisedSnapshots = normalisedSnapshots.to_sparse_coo()
			normalisedSnapshots = normalisedSnapshots.coalesce()

			y, classTargets = self.generateClassTargets(slidingWindowIndex, normalisedSnapshots, B1, seq_input)
			
			if(trainOrTest and not generateConnectionsAfterPropagating):	#debug only
				self.addNormalisedSnapshotToDatabase(normalisedSnapshots, classTargets, keypointPairsValid, keypointPairsIndices, kp_meta_batch)
				if(ATNLPsnapshotDatabaseRamDynamic):
					ATNLPpt_database.finaliseTrainedSnapshotDatabase(self)
				
			# -----------------------------
			# Prediction
			# -----------------------------
			if(not trainOrTest or ATNLPsnapshotDatabaseRamDynamic):
				if(ATNLPsnapshotDatabaseDiskCompareChunks):
					comparisonFound, unit_sim, top_cls, avg_sim = ATNLPpt_comparison.compare_1d_batches_stream_db(self, normalisedSnapshots, B1, keypointPairsIndices, kp_meta_batch, chunk_nnz=ATNLPsnapshotDatabaseDiskCompareChunksSize, device=device, shiftInvariantPixels=ATNLPcomparisonShiftInvariancePixels)
				else:
					comparisonFound, unit_sim, top_cls, avg_sim = ATNLPpt_comparison.compare_1d_batches(self, normalisedSnapshots, B1, keypointPairsIndices, kp_meta_batch, chunk=ATNLPsnapshotCompareChunkSize, device=device, shiftInvariantPixels=ATNLPcomparisonShiftInvariancePixels)
				
				if(comparisonFound):
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
				self.addNormalisedSnapshotToDatabase(normalisedSnapshots, classTargets, keypointPairsValid, keypointPairsIndices, kp_meta_batch)
		
		if(ATNLPsnapshotDatabaseDisk and trainOrTest):
			self.closeSnapshotDatabaseWriter()
	
		accuracy = accuracyAllWindows / numSubsamplesWithKeypoints
		loss = Loss(0.0)
		
		return loss, accuracy
	
	if(ATNLPsnapshotDatabaseDisk):
		def initialiseSnapshotDatabaseWriter(self):
			global snapshotDatabaseWriters
			referenceSetDelimiterIDmax = self.getReferenceSetDelimiterIDmax()
			snapshotDatabaseWriters = [[None for _ in range(S)] for _ in range(referenceSetDelimiterIDmax)]

		def closeSnapshotDatabaseWriter(self):
			#TODO: make more efficient (use dict);
			global snapshotDatabaseWriters
			referenceSetDelimiterIDmax = self.getReferenceSetDelimiterIDmax()
			for referenceSetDelimiterID in range(referenceSetDelimiterIDmax):
				for s in range(S):
					if(snapshotDatabaseWriters[referenceSetDelimiterID][s] is not None):
						snapshotDatabaseWriters[referenceSetDelimiterID][s].close()
						snapshotDatabaseWriters[referenceSetDelimiterID][s] = None
						#print("snapshotDatabaseWriter close")
									
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
		
	def getReferenceSetDelimiterIDmax(self):
		referenceSetDelimiterIDmax = 0
		referenceSetDelimiterIDmax += 1	#"."
		referenceSetDelimiterIDmax += len(ATNLPpt_keypoints.verb_dict)
		#referenceSetDelimiterIDmax += len(ATNLPpt_keypoints.prep_dict)
		return referenceSetDelimiterIDmax

	def getReferenceSetDelimiterID(self, keypointPairsIndexFirst, kp_meta_batch):
		#sync with referenceSetPosDelimitersTagStr
		#print("keypointPairsIndexFirst = ", keypointPairsIndexFirst)
		keypointPairsIndexFirst = keypointPairsIndexFirst.item()
		spacy_input_id = kp_meta_batch[keypointPairsIndexFirst]['spacy_input_id'].item()
		spacy_pos = kp_meta_batch[keypointPairsIndexFirst]['spacy_pos'].item()
		spacy_input_id = ANNpt_data.to_uint64(spacy_input_id)	#convert back to its original unsigned form (to_int64 was used for dataloader)
		
		referenceSetDelimiterID = 0
		#print("spacy_input_id = ", spacy_input_id)
		#print("spacy_pos = ", spacy_pos)
		#print("ATNLPpt_keypoints.punctPosId = ", ATNLPpt_keypoints.punctPosId)
		#print("ATNLPpt_keypoints.verbPosId = ", ATNLPpt_keypoints.verbPosId)
		if(spacy_pos == ATNLPpt_keypoints.punctPosId):
			referenceSetsFirstDelimiterString = lexIntToLexString(ATNLPpt_keypoints.nlp, spacy_input_id)
			print("punctPosId: referenceSetsFirstDelimiterString = ", referenceSetsFirstDelimiterString)
			if(referenceSetsFirstDelimiterString == "."):
				referenceSetDelimiterID += 1	#"."
		elif(spacy_pos == ATNLPpt_keypoints.verbPosId):
			referenceSetsFirstDelimiterString = lexIntToLexString(ATNLPpt_keypoints.nlp, spacy_input_id)
			print("verbPosId: referenceSetsFirstDelimiterString = ", referenceSetsFirstDelimiterString)
			referenceSetDelimiterID += 1
			if referenceSetsFirstDelimiterString in ATNLPpt_keypoints.verb_dict:
				verbIndex = ATNLPpt_keypoints.verb_dict[referenceSetsFirstDelimiterString]
				#print("verbIndex = ", verbIndex)
				referenceSetDelimiterID += verbIndex
			else:
				printe("referenceSetsFirstDelimiterString not in verb_dict, referenceSetsFirstDelimiterString = " + referenceSetsFirstDelimiterString)
		'''
		elif(spacy_pos == ATNLPpt_keypoints.prepositionPosId):
			referenceSetsFirstDelimiterString = posIntToPosString(ATNLPpt_keypoints.nlp, spacy_input_id)
			print("referenceSetsFirstDelimiterString = ", referenceSetsFirstDelimiterString)
			referenceSetDelimiterID += 1
			referenceSetDelimiterID += len(ATNLPpt_keypoints.verb_dict)
			if referenceSetsFirstDelimiterString in ATNLPpt_keypoints.prep_dict:
				referenceSetDelimiterID += prep_dict[referenceSetsFirstDelimiterString]
			else:
				printe("referenceSetsFirstDelimiterString not in prep_dict, referenceSetsFirstDelimiterString = " + referenceSetsFirstDelimiterString)
		'''

		return referenceSetDelimiterID

	def detectValidNormalisedSnapshot(self, normalisedSnapshot):
		if(normalisedSnapshot.count_nonzero() == 0):
			validSnapshotFound = False
		else:
			validSnapshotFound = True
		return validSnapshotFound
			
	def addNormalisedSnapshotToDatabase(self, normalisedSnapshots, classTargets, keypointPairsValid, keypointPairsIndices, kp_meta_batch):
		B1 = normalisedSnapshots.shape[0]
		S = normalisedSnapshots.shape[1]
		if(ATNLPsnapshotDatabaseDisk):
			global snapshotDatabaseWriters
			for b1 in range(B1):
				for s in range(S):
					if(keypointPairsValid[b1, s]):
						keypointPairsIndexFirst = keypointPairsIndices[b1, s, 0]
						referenceSetDelimiterID = self.getReferenceSetDelimiterID(keypointPairsIndexFirst, kp_meta_batch[b1])
						if(snapshotDatabaseWriters[referenceSetDelimiterID][s] == None):
							snapshotDatabaseName = ATNLPpt_database.getDatabaseName(referenceSetDelimiterID, s)
							snapshotDatabaseWriters[referenceSetDelimiterID][s] = ATNLPpt_database.H5DBWriter(snapshotDatabaseName, C, L2)
						snapshotDatabaseWriter = snapshotDatabaseWriters[referenceSetDelimiterID][s]
						normalisedSnapshot = normalisedSnapshots[b1][s].unsqueeze(0)
						classTarget = classTargets[b1][s].unsqueeze(0)
						snapshotDatabaseWriter.add_batch(normalisedSnapshot, classTarget)
		elif(ATNLPsnapshotDatabaseRam):
			for b1 in range(B1):
				for s in range(S):
					if(keypointPairsValid[b1, s]):
						keypointPairsIndexFirst = keypointPairsIndices[b1, s, 0]
						referenceSetDelimiterID = self.getReferenceSetDelimiterID(keypointPairsIndexFirst, kp_meta_batch[b1])
						normalisedSnapshot = normalisedSnapshots[b1][s]
						classTarget = classTargets[b1][s]
						self.normalisedSnapshot_list[referenceSetDelimiterID][s].append(normalisedSnapshot)
						self.classTarget_list[referenceSetDelimiterID][s].append(classTarget)
		
	def generateClassTargets(self, slidingWindowIndex, normalisedSnapshots, B1, seq_input):
		#for now just predict final character in sequence window
		#FUTURE: predict Bert tokens rather than individual characters)
		
		B1 = normalisedSnapshots.shape[0]
		S = normalisedSnapshots.shape[1]
		y = seq_input[:, slidingWindowIndex] #shape = B1	
		B2 = B1*S
		classTargets = y.unsqueeze(1).repeat(1, S)	#shape = (B1, S)
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




