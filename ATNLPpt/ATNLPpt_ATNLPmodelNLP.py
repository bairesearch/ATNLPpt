"""ATNLPpt_ATNLPmodelNLP.py

# Author:
Richard Bruce Baxter - Copyright (c) 2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ANNpt_main.py

# Usage:
see ANNpt_main.py

# Description:
ATNLPpt model NLP

"""

import torch
from ANNpt_globalDefs import *
from transformers import AutoModel

if(useTokenEmbedding):
	model = AutoModel.from_pretrained(bertModelName).eval().to(device)   # embeddingSize-dim vectors

def getEncodedFeatureSize():
	if(useSequentialSANI):
		encodedFeatureSize = sequenceLength*EISANINLPcontinuousVarEncodingNumBits
	else:
		if(useTokenEmbedding):
			encodedFeatureSize = sequenceLength*embeddingSize*EISANINLPcontinuousVarEncodingNumBits
		else:
			encodedFeatureSize = sequenceLength*EISANINLPcontinuousVarEncodingNumBits
	return encodedFeatureSize
	
def encodeTokensInputIDs(self, x):
	input_ids = x
	if(useTokenEmbedding):
		token_type_ids = torch.zeros_like(input_ids)
		with torch.no_grad():
			embeds = model.embeddings(input_ids=input_ids, token_type_ids=token_type_ids)	 # [batchSize, sequenceLength, embeddingSize]
		embeds = normalize_to_unit_range(embeds)
		embeds = normalize_unit_range_to_0_to_1(embeds)
	else:
		embeds = input_ids	#will apply one-hot encoding directly to integers
	return embeds

if(useTokenEmbedding):
	def normalize_to_unit_range(embeds: torch.Tensor, per_token: bool = True, eps: float = 1e-12,) -> torch.Tensor:
		"""
		Scale each embedding vector so every element is in [-1, +1].

		Parameters
		----------
		embeds : FloatTensor  [B, L, H]
			Output of `model.embeddings(...)` (or any float tensor).
		per_token : bool, default True
			* True  -> scale each token independently  
			* False -> scale across the entire batch (rarely useful)
		eps : float
			Tiny constant to avoid division by zero.

		Returns
		-------
		norm : FloatTensor  [B, L, H]
			Same shape, values guaranteed in [-1, +1].
		"""
		if per_token:
			# max over hidden dim only \u2192 each token handled separately
			max_abs = embeds.abs().amax(dim=-1, keepdim=True)
		else:
			# max over every element in the tensor
			max_abs = embeds.abs().amax()

		norm = embeds / (max_abs + eps)
		norm = norm.clamp_(-1.0, 1.0)	  # numerical safety (optional)
		return norm

	def normalize_unit_range_to_0_to_1(x: torch.Tensor) -> torch.Tensor:
		"""
		Affine rescale from [-1, +1] -> [0, 1].

		Parameters
		----------
		x : FloatTensor
			Any tensor whose values are already in the closed interval [-1, +1].

		Returns
		-------
		y : FloatTensor
			Same shape as `x`, but with every element mapped so:
				-1 \u2192 0
				 0 \u2192 0.5
				+1 \u2192 1
		"""
		y = (x + 1.0) * 0.5		# (x + 1) / 2
		y = y.clamp_(0.0, 1.0)  # numerical safety (optional)
		return y
