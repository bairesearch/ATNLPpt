"""ATNLPpt_ATNLPmodelContinuousVarEncoding.py

# Author:
Richard Bruce Baxter - Copyright (c) 2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ANNpt_main.py

# Usage:
see ANNpt_main.py

# Description:
ATNLPpt model Continuous Var Encoding

"""

import torch
from ANNpt_globalDefs import *
import torch.nn.functional as F
if(useNLPDataset):
	import ATNLPpt_ATNLPmodelNLP

# -------------------------------------------------------------
# Continuous var encoding as bits
# -------------------------------------------------------------

def continuousVarEncoding(self, x):
	if useTabularDataset:
		numBits = EISANITABcontinuousVarEncodingNumBits
		encoded = encodeContinuousVarsAsBits(self, x, numBits)
		initActivation = encoded.to(torch.int8)
		numSubsamples = 1
	elif useImageDataset:
		numBits = EISANICNNcontinuousVarEncodingNumBits
		encoded = encodeContinuousVarsAsBits(self, x)
		initActivation = EISANIpt_EISANImodelCNN.propagate_conv_layers(self, encoded, numBits)	# (batch, encodedFeatureSize) int8
	elif useNLPDataset:
		numBits = EISANINLPcontinuousVarEncodingNumBits
		#print("x.shape = ", x.shape)
		#print("x = ", x)
		embedding = ATNLPpt_ATNLPmodelNLP.encodeTokensInputIDs(self, x)	# (batch, sequenceLength, embeddingSize) float32
		#print("embedding.shape = ", embedding.shape)
		encoded = encodeContinuousVarsAsBits(self, embedding, numBits)	#int8	#[batchSize, sequenceLength, embeddingSize*EISANINLPcontinuousVarEncodingNumBits]
		#print("encoded = ", encoded)
		initActivation = encoded.to(torch.int8)
		#encodedFeatureSize = EISANIpt_EISANImodelNLP.getEncodedFeatureSize()	#sequenceLength*embeddingSize*EISANINLPcontinuousVarEncodingNumBits
	return initActivation

def encodeContinuousVarsAsBits(self, x: torch.Tensor, numBits: int) -> torch.Tensor:
	if(useTabularDataset):
		pass
	if(useImageDataset):
		if numBits == 1:
			return x
		B, C, H, W = x.shape
		x = x.view(B, C * H * W)  # Flatten pixel dimensions
	elif(useNLPDataset):
		if(useTokenEmbedding):
			B, L, E = x.shape
			x = x.view(B, L * E)
		else:
			B, L = x.shape
			x = x.view(B, L)

	if useContinuousVarEncodeMethod=="grayCode":
		encoded_bits_list = gray_code_encode(self, x, numBits, continuousVarMin, continuousVarMax, self.config.fieldTypeList)
	elif useContinuousVarEncodeMethod=="thermometer":
		encoded_bits_list = thermometer_encode(self, x, numBits, continuousVarMin, continuousVarMax, self.config.fieldTypeList)
	elif useContinuousVarEncodeMethod=="onehot":
		encoded_bits = F.one_hot(x, num_classes=numBits)	# (B, L, numBits)

	if(useTabularDataset):
		code = torch.cat(encoded_bits_list, dim=1) # Concatenate along the feature/bit dimension	#[B, nCont*numBits]
	elif(useImageDataset):
		code = torch.stack(encoded_bits_list, dim=2) 	#[B, C*H*W, EISANICNNcontinuousVarEncodingNumBits]
		code = code.view(B, C, H, W, numBits)	#unflatten pixel dimensions
		code = code.permute(0, 1, 4, 2, 3)  # Rearrange dimensions to [B, C, numBits, H, W]
		code = code.reshape(B, C*numBits, H, W)
	elif(useNLPDataset):
		if useContinuousVarEncodeMethod=="onehot":
			code = encoded_bits.view(B, L, numBits) 	#[B, L*numBits]
			#FUTURE: update EISANIpt_EISANImodel and EISANIpt_EISANImodelDynamic to support useNLPDataset with dim [batchSize sequenceLength, EISANINLPcontinuousVarEncodingNumBits]
			code = code.reshape(B, L*numBits)
		else:
			code = torch.stack(encoded_bits_list, dim=1) 	#[B, L, numBits]
			#code = code.reshape(B, L, E*numBits)		#FUTURE: update EISANIpt_EISANImodel and EISANIpt_EISANImodelDynamic to support useNLPDataset with dim [batchSize sequenceLength, embeddingSize*EISANINLPcontinuousVarEncodingNumBits]
			code = code.reshape(B, L*E*numBits)

	return code

def gray_code_encode(self, x: torch.Tensor, numBits: int, minVal: float, maxVal: float, fieldTypeList: list) -> list[torch.Tensor]:
	"""Vectorised Gray-code encoding (batch, features, bits) with no costly Python loop."""
	batch_size, num_features = x.shape
	device = x.device

	# 1. Identify Boolean columns once
	isBool = torch.zeros(num_features, dtype=torch.bool, device=device)
	if encodeDatasetBoolValuesAs1Bit and fieldTypeList:
		limit = min(num_features, len(fieldTypeList))
		isBool[:limit] = torch.tensor([ft == 'bool' for ft in fieldTypeList[:limit]], device=device)

	# 2. Boolean bits (1-bit, already 0/1)
	boolBits = x[:, isBool].float()											# (batch, nBool)

	# 3. Continuous bits - quantise - Gray - bit-slice
	if (~isBool).any():
		xCont = x[:, ~isBool]												# (batch, nCont)
		intLevels = continuous_to_int(self, xCont, numBits, minVal, maxVal)		# (batch, nCont)
		grayLevels = intLevels ^ (intLevels >> 1)
		bitPos = torch.arange(numBits, device=device)						# (numBits,)
		contBits = ((grayLevels.unsqueeze(-1) >> bitPos) & 1).float()		# (batch, nCont, numBits)
	else:
		contBits = x.new_empty(batch_size, 0, numBits)

	# 4. Re-interleave (cheap Python, negligible cost)
	encoded_bits_list: list[torch.Tensor] = []
	bool_iter = iter(boolBits.unbind(dim=1))									# each - (batch,)
	cont_iter = iter(contBits.unbind(dim=1))								# each - (batch, numBits)
	for flag in isBool:														# trivial loop
		encoded_bits_list.append(next(bool_iter) if flag else next(cont_iter))
	return encoded_bits_list

def thermometer_encode(self, x: torch.Tensor, numBits: int, minVal: float, maxVal: float, fieldTypeList: list) -> list[torch.Tensor]:
	"""Vectorised thermometer encoding (batch, features, bits) without per-feature Python work."""
	batch_size, num_features = x.shape
	device = x.device

	isBool = torch.zeros(num_features, dtype=torch.bool, device=device)
	if encodeDatasetBoolValuesAs1Bit and fieldTypeList:
		limit = min(num_features, len(fieldTypeList))
		isBool[:limit] = torch.tensor([ft == 'bool' for ft in fieldTypeList[:limit]], device=device)

	boolBits = x[:, isBool].float()											# (batch, nBool)

	if (~isBool).any():
		xCont = x[:, ~isBool]												# (batch, nCont)
		intLevels = continuous_to_int(self, xCont, numBits, minVal, maxVal)		# (batch, nCont)
		thresholds = torch.arange(numBits, device=device)					# (numBits,)
		contBits = (intLevels.unsqueeze(-1) >= thresholds).float()			# (batch, nCont, numBits)
	else:
		contBits = x.new_empty(batch_size, 0, numBits)

	encoded_bits_list: list[torch.Tensor] = []
	bool_iter = iter(boolBits.unbind(dim=1))
	cont_iter = iter(contBits.unbind(dim=1))
	for flag in isBool:
		encoded_bits_list.append(next(bool_iter) if flag else next(cont_iter))
	return encoded_bits_list

def continuous_to_int(self, x: torch.Tensor, numBits: int, minVal: float, maxVal: float) -> torch.Tensor:
	"""Map (batch, features) continuous tensor to integer levels [0, 2**bits-1]."""
	xClamped = torch.clamp(x, minVal, maxVal)
	norm = (xClamped - minVal) / (maxVal - minVal)
	scaled = norm * (2 ** numBits - 1)
	return scaled.round().to(torch.long)
