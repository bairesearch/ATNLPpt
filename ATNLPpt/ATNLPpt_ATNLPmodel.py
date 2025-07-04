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
import ATNLPpt_normalisation
#import ATNLPpt_comparison	#FUTURE
#import ATNLPpt_ATNLPmodelOutput	#FUTURE

		
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

	# ---------------------------------------------------------
	# Forward pass
	# ---------------------------------------------------------
		
	@torch.no_grad()
	def forward(self, trainOrTest: bool, x: torch.Tensor, y: Optional[torch.Tensor] = None, optim=None, l=None, batchIndex=None, fieldTypeList=None) -> Tuple[torch.Tensor, torch.Tensor]:
		"""Forward pass.
		
		Args:
			trainOrTest: True=> train mode; False=> inference.
			x: dict of different data types;
				each is a (batch, length) feature tensor.
				"char_input_ids" : (B, Lc),
				"bert_input_ids" : (B, Lb),
				"bert_offsets"   : (B, Lb, 2),
				"spacy_input_ids": (B, Ls),
				"spacy_pos"      : (B, Ls),
				"spacy_offsets"  : (B, Ls, 2),
			y: None
				dynamically generated; (batch,) int64 labels when trainOrTest==True.
				
		Returns:
			predictions, outputActivations (both shape (batch, classes)).
		"""
		seq_char = x['char_input_ids']	# Tensor [batchSize, sequenceLength]
		B1 = self.batchSize = seq_char.shape[0]
		#assert (self.batchSize == self.config.batchSize), "Batch size must match config.batchSize"
		device = seq_char.device

		if(useSlidingWindow):			
			non_pad	= (seq_char != NLPcharacterInputPadTokenID)		# [B1, L1]
			if not pt.any(non_pad):
				return
			lengths	= non_pad.sum(-1)							# [B1]
			max_len = int(lengths.max().item())
			numSubsamples = max(1, max_len - 1)				# predict *next* token only
			extra = contextSizeMax - lengths					# [B1]
		else:
			numSubsamples = 1
	
		# -----------------------------
		# Continuous var encoding as bits
		# -----------------------------
		seq_char_tensor = ATNLPpt_ATNLPmodelContinuousVarEncoding.encodeContinuousVarsAsBits(self, seq_char, ATNLPcontinuousVarEncodingNumBits).float()	#NLPcharacterInputSetLen	#[batchSize, sequenceLength*numBits]
		#seq_token_tensor = encodeContinuousVarsAsBits(self, x['bert_input_ids'], bertNumberTokenTypes).to(torch.int8)	#[batchSize, sequenceLength*numBits]

		seq_char_tensor = seq_char_tensor.reshape(B1, L1, C)     #shape (B1, L1, C)
		seq_char_tensor = seq_char_tensor.permute(0, 2, 1)     #shape (B1, C, L1)

		spacy_pos = x['spacy_pos']
		spacy_offsets = x['spacy_offsets']
		kp_indices_batch, kp_meta_batch = ATNLPpt_normalisation.build_keypoints(spacy_pos, spacy_offsets)
	
		accuracyAllWindows = 0
		#print("numSubsamples = ", numSubsamples)
		for slidingWindowIndex in range(numSubsamples):
			if(debugSequentialLoops):
				print("\n************************** slidingWindowIndex = ", slidingWindowIndex)

			# -----------------------------
			# Normalisation
			# -----------------------------
			if(trainOrTest):
				mode="firstKeypointConsecutivePairs"	 #out shape = (B1*r, C, L2)
			else:
				mode="firstKeypointPairs"	 	#out shape = (B1*r*(q-1), C, L2)		
			out = ATNLPpt_normalisation.normalise_batch(slidingWindowIndex, seq_char_tensor, x['spacy_pos'], x['spacy_offsets'], mode=mode, r=r, q=q, L2=L2, kp_indices_batch=kp_indices_batch, kp_meta_batch=kp_meta_batch)
	
			# -----------------------------
			# Output layer [FUTURE]
			# -----------------------------
			#predictions = EISANIpt_EISANImodelOutput.calculateOutputLayer(self, trainOrTest, layerActivations, y)

			'''
			# count how many are exactly correct
			if(useNLPDataset):
				valid_mask = (y != NLPcharacterInputPadTokenID)
				valid_count = valid_mask.sum().item()
				if valid_count > 0:
					correct = ((predictions == y) & valid_mask).sum().item()
					accuracyAllWindows += correct / valid_count
				else:
					accuracyAllWindows += 0.0
			else:
				correct = (predictions == y).sum().item()
				accuracy = correct / y.size(0)
				accuracyAllWindows += accuracy
			'''
		
		accuracy = accuracyAllWindows / numSubsamples
		loss = Loss(0.0)
		
		return loss, accuracy


