"""ANNpt_linearSublayers.py

# Author:
Richard Bruce Baxter - Copyright (c) 2023-2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ANNpt_main.py

# Usage:
see ANNpt_main.py

# Description:
ANNpt linear sublayers

"""

import torch as pt
from torch import nn
from ANNpt_globalDefs import *
import numpy as np
import torch.nn.functional as F
import math

numpyVersion = 2

class ClippedReLU(nn.Module):
	def __init__(self, min_val=0, max_val=float('inf'), invertActivation=False):
		super(ClippedReLU, self).__init__()
		self.min_val = min_val
		self.max_val = max_val
		self.invertActivation = invertActivation	#not currently used (inversion is performed before activation function is applied)

	def forward(self, x):
		if(self.invertActivation):
			x = -pt.sign(x) * pt.abs(x)
		a = pt.clamp(x, min=self.min_val, max=self.max_val)
		return a

class Relu(nn.Module):	#not currently used
	def __init__(self, invertActivation=False):
		super(Relu, self).__init__()
		self.invertActivation = invertActivation

	def forward(self, x):
		if(self.invertActivation):
			x = -pt.sign(x) * pt.abs(x)
		a = torch.nn.functional.relu(x)
		return a
				
class LinearSegregated(nn.Module):
	def __init__(self, in_features, out_features, number_sublayers, bias=True, cnn=False):
		super().__init__()
		self.useCNNlayers = cnn
		if(self.useCNNlayers):
			self.segregatedLinear = nn.Conv2d(in_channels=in_features*number_sublayers, out_channels=out_features*number_sublayers, kernel_size=CNNkernelSize, stride=CNNstride, padding=CNNpadding, groups=number_sublayers, bias=bias)
		else:	
			self.segregatedLinear = nn.Conv1d(in_channels=in_features*number_sublayers, out_channels=out_features*number_sublayers, kernel_size=1, groups=number_sublayers, bias=bias)
		self.number_sublayers = number_sublayers
		
	def forward(self, x):
		#x.shape = batch_size, number_sublayers, in_features
		if(self.useCNNlayers):
			x = x.view(x.shape[0], x.shape[1]*x.shape[2], x.shape[3], x.shape[4])
		else:
			x = x.view(x.shape[0], x.shape[1]*x.shape[2], 1)
		x = self.segregatedLinear(x)
		if(self.useCNNlayers):
			x = x.view(x.shape[0], self.number_sublayers, x.shape[1]//self.number_sublayers, x.shape[2], x.shape[3])
		else:
			x = x.view(x.shape[0], self.number_sublayers, x.shape[1]//self.number_sublayers)
		#x.shape = batch_size, number_sublayers, out_features
		return x

def generateLinearLayerMatch(self, layerIndex, config, parallelStreams=False, sign=True, featuresFanIn=False, inFeaturesMatchHidden=False, inFeaturesMatchOutput=False, bias=True):
	if(inFeaturesMatchHidden):
		if(layerIndex == 0):
			in_features = config.hiddenLayerSize
			out_features = config.hiddenLayerSize
		else:
			printe("generateLinearLayer error: inFeaturesMatchHidden and layerIndex != 0")	
	elif(inFeaturesMatchOutput):
		if(layerIndex == config.numberOfLayers-1):
			in_features = config.outputLayerSize
			out_features = config.outputLayerSize
		else:
			printe("generateLinearLayer error: outFeaturesMatchInput and layerIndex != config.numberOfLayers-1")
	else:
		if(inputLayerInList and layerIndex == 0):
			in_features = config.inputLayerSize
		else:
			in_features = config.hiddenLayerSize
		if(outputLayerInList and layerIndex == config.numberOfLayers-1):
			out_features = config.outputLayerSize
		else:
			out_features = config.hiddenLayerSize
	if(featuresFanIn):
		in_features = in_features*2
	linearSublayersNumber = config.linearSublayersNumber
	return generateLinearLayer2(self, layerIndex, in_features, out_features, linearSublayersNumber, parallelStreams, sign, bias)

def generateLinearLayerCNN(self, layerIndex1, config, parallelStreams=False, forward=True, bias=True, layerIndex2=None):
	out_features, in_channels, out_channels, in_width, in_height, out_width, out_height, numberInputPaddedFeatures = getCNNproperties(self, layerIndex1)
	if(layerIndex2 == config.numberOfLayers):
		printe("generateLinearLayerCNN error: does not support layerIndex2 == config.numberOfLayers; use generateLinearLayerCNN instead")
	assert forward==True
	assert useLinearSublayers==False	#this CNN implementation wrapper does not currently support useLinearSublayers
	linearSublayersNumber = config.linearSublayersNumber
	return generateLinearLayer2(self, layerIndex1, in_channels, out_channels, linearSublayersNumber, parallelStreams, bias=bias, cnn=True)

def generateLinearLayer(self, layerIndex1, config, parallelStreams=False, forward=True, bias=True, layerIndex2=None):
	if(layerIndex2 is not None):
		if(layerIndex1 == 0):
			in_features = config.inputLayerSize
		elif(layerIndex1 == config.numberOfLayers):
			in_features = config.outputLayerSize
		else:
			in_features = config.hiddenLayerSize
		if(layerIndex2 == 0):
			out_features = config.inputLayerSize
		elif(layerIndex2 == config.numberOfLayers):
			out_features = config.outputLayerSize
		else:
			out_features = config.hiddenLayerSize
	if(not forward):
		#switch i/o layer features;
		temp = in_features
		in_features = out_features
		out_features = temp
	linearSublayersNumber = config.linearSublayersNumber
	return generateLinearLayer2(self, layerIndex1, in_features, out_features, linearSublayersNumber, parallelStreams, bias=bias, layerIndex2=layerIndex2)

def generateLinearLayer2(self, layerIndex1, in_features, out_features, linearSublayersNumber, parallelStreams=False, sign=True, bias=True, cnn=False, layerIndex2=None):
	if(useImageDataset and not cnn):
		if(layerIndex1 == 0):
			pass
		elif(layerIndex1 <= numberOfConvlayers):
			in_features = CNNhiddenLayerSize
		if(layerIndex2 <= numberOfConvlayers):
			out_features = CNNhiddenLayerSize
	if(getUseLinearSublayers(self, layerIndex1)):
		linear = LinearSegregated(in_features=in_features, out_features=out_features, number_sublayers=linearSublayersNumber, bias=bias)
	else:
		if(cnn):
			if(useCNNlayers2D):
				linear = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=CNNkernelSize, stride=CNNstride, padding=CNNpadding, bias=bias)
			else:
				linear = nn.Conv1d(in_channels=in_features, out_channels=out_features, kernel_size=CNNkernelSize, stride=CNNstride, padding=CNNpadding, bias=bias)
		else:
			if(parallelStreams):
				in_features = in_features*linearSublayersNumber
			linear = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)

	weightsSetLayer(self, layerIndex1, linear, sign)

	return linear

def generateActivationLayer(self, layerIndex, config, positive=True):
	return generateActivationFunction(activationFunctionType, positive)

class ReLUNeg(nn.Module):
    def forward(self, x):
        return pt.relu(-x)
		
def generateActivationFunction(activationFunctionType, positive=True):
	if(activationFunctionType=="softmax"):
		if(thresholdActivations):
			activation = OffsetSoftmax(thresholdActivationsMin)
		else:
			activation = nn.Softmax(dim=1)
	elif(activationFunctionType=="relu"):
		if(positive):
			if(thresholdActivations):
				activation = OffsetReLU(thresholdActivationsMin)
			else:
				activation = nn.ReLU()
		else:
			if(thresholdActivations):
				printe("trainThreshold==positive:activationFunctionType==relu generateActivationFunction error: !positive+thresholdActivations not yet coded")
			else:
				activation = ReLUNeg()	#sets positive values to zero, and converts negative values to positive
	elif(activationFunctionType=="clippedRelu"):
		activation = ClippedReLU(min_val=0, max_val=clippedReluMaxValue)
	elif(activationFunctionType=="sigmoid"):
		activation = nn.Sigmoid()
	elif(activationFunctionType=="none"):
		activation = None
	return activation

def getCNNproperties(self, layerIndex):

	numberOfHiddenLayers = self.config.numberOfLayers-1
	hiddenLayerSize = self.config.CNNhiddenLayerSize

	if(CNNinputPadding):
		#pad input layer to support CNN layers;
		if(CNNmaxInputPadding):	#pad input layer to support max number of CNN layers
			input_space_divisor = CNNinputSpaceDivisor**numberOfConvlayers
		else:	#pad input layer to support min number of CNN layers
			input_space_divisor = CNNinputSpaceDivisor
		remainder = self.config.inputLayerSize % input_space_divisor
		if(remainder == 0):
			numberInputPaddedFeatures = 0
		else:
			numberInputPaddedFeatures = (input_space_divisor-remainder)
			printe("getCNNproperties warning: numberInputPaddedFeatures != 0")
		inputLayerFeatures = self.config.inputLayerSize//numberInputImageChannels + numberInputPaddedFeatures
	else:
		numberInputPaddedFeatures = 0
		inputLayerFeatures = self.config.inputLayerSize//numberInputImageChannels
	
	if(debugCNN):
		print("\n\n getCNNproperties:")
		print("layerIndex = ", layerIndex)
		print("self.config.inputLayerSize = ", self.config.inputLayerSize)
		print("numberOfConvlayers = ", numberOfConvlayers)
		print("self.config.inputLayerSize = ", self.config.inputLayerSize)
		print("input_space_divisor = ", input_space_divisor)
		print("inputLayerFeatures = ", inputLayerFeatures)
		print("numberInputPaddedFeatures = ", numberInputPaddedFeatures)
	
	if(useCNNlayers2D):
		inputLayerWidth = math.sqrt(inputLayerFeatures)
		if not inputLayerWidth.is_integer():
			printe("executeLinearLayer error: useCNNlayers2D requires math.sqrt(inputLayerFeatures) is integer")
		inputLayerWidth = int(inputLayerWidth)
	else:
		inputLayerWidth = inputLayerFeatures
	
	if(layerIndex == 0):
		in_width_divisor = 1
	else:
		if(CNNconvergeEveryEvenLayer):
			in_width_divisor = CNNinputWidthDivisor**(layerIndex//2)
		else:
			in_width_divisor = CNNinputWidthDivisor**layerIndex
	if(CNNconvergeEveryEvenLayer):
		out_width_divisor = CNNinputWidthDivisor**((layerIndex+1)//2)
	else:
		out_width_divisor = CNNinputWidthDivisor**(layerIndex+1)
	
	if(inputLayerWidth%in_width_divisor == 0):
		in_width = inputLayerWidth//in_width_divisor
	else:
		#input layer does not support subdivision
		in_width = 1
	if(inputLayerWidth%out_width_divisor == 0):
		out_width = inputLayerWidth//out_width_divisor
		#print("inputLayerWidth = ", inputLayerWidth)
		#print("out_width_divisor = ", out_width_divisor)
	else:
		#input layer does not support subdivision
		out_width = 1

	if(debugCNN):
		print("in_width_divisor = ", in_width_divisor)
		print("out_width_divisor = ", out_width_divisor)
		print("in_width = ", in_width)
		print("out_width = ", out_width)
	
	if(layerIndex == 0):
		in_channels = numberInputImageChannels	#special number of channels for input layer
	else:
		if(in_width == 1):
			in_channels = hiddenLayerSize
		else:
			supportSubdivision = True
			if(useCNNlayers2D):
				if(hiddenLayerSize%(in_width**2) == 0):
					in_channels = hiddenLayerSize//(in_width**2)
					supportSubdivision = True
			else:
				if(hiddenLayerSize%in_width == 0):
					in_channels = hiddenLayerSize//in_width
					supportSubdivision = True
			if not supportSubdivision:
				#hidden layer does not support subdivision
				in_width = 1
				in_channels = hiddenLayerSize
	if(out_width == 1):
		out_channels = hiddenLayerSize
	else:	
		supportSubdivision = True
		if(useCNNlayers2D):
			if(hiddenLayerSize%(out_width**2) == 0):
				out_channels = hiddenLayerSize//(out_width**2)
				supportSubdivision = True
		else:
			if(hiddenLayerSize%out_width == 0):
				out_channels = hiddenLayerSize//out_width
				supportSubdivision = True
		if not supportSubdivision:
			#hidden layer does not support subdivision
			out_width = 1
			out_channels = hiddenLayerSize

	out_features = hiddenLayerSize
	in_height = in_width	#only used by useCNNlayers2D
	out_height = out_width	#only used by useCNNlayers2D
	
	if(debugCNN):
		print("out_features = ", out_features)
		print("in_channels = ", in_channels)
		print("out_channels = ", out_channels)
		print("in_width = ", in_width)
		print("out_width = ", out_width)
	
	return out_features, in_channels, out_channels, in_width, in_height, out_width, out_height, numberInputPaddedFeatures
		
def executeLinearLayer(self, layerIndex, x, linear, parallelStreams=False, sign=True, cnn=False):
	if(useSignedWeights):
		weightsFixLayer(self, layerIndex, linear, sign)	#otherwise need to constrain backprop weight update function to never set weights below 0
		
	if(cnn):		
		if(layerIndex == 0):
			out_features, in_channels, out_channels, in_width, in_height, out_width, out_height, numberInputPaddedFeatures = getCNNproperties(self, layerIndex)
			#always ensure input layer supports at least an initial subdivision (or if CNNmaxInputPadding, the number of subdivisions implied by numberOfHiddenLayers)
			x = F.pad(x, (0, numberInputPaddedFeatures), mode='constant', value=0)	#pad along dim=1 by numberPaddedFeatures with zeros
		x = convReshapeIn(self, x, layerIndex)
		x = linear(x)
		x = convReshapeOut(self, x, layerIndex)
	elif(getUseLinearSublayers(self, layerIndex)):
		#perform computation for each sublayer independently
		if(not parallelStreams):
			x = x.unsqueeze(dim=1).repeat(1, self.config.linearSublayersNumber, 1)
		x = linear(x)
	else:
		if(parallelStreams):
			x = pt.reshape(x, (x.shape[0], x.shape[1]*x.shape[2]))
		x = linear(x)
	return x

def executeActivationLayer(self, layerIndex, x, activationFunction, parallelStreams=False, executeActivationFunctionOverFeatures=True):
	if(normaliseActivationSparsity):
		x = nn.functional.layer_norm(x, x.shape[1:])   #normalized_shape does not include batchSize
	if(getUseLinearSublayers(self, layerIndex) and not simulatedDendriticBranches):
		if(activationFunctionType=="softmax"):
			if(executeActivationFunctionOverFeatures):
				numberOfSamples = x.shape[0]
				numberOfSublayers = x.shape[1]
				if(self.useCNNlayers):
					x = pt.reshape(x, (x.shape[0]*x.shape[1], x.shape[2], x.shape[3], x.shape[4]))
				else:
					x = pt.reshape(x, (x.shape[0]*x.shape[1], x.shape[2]))
			x = activationFunction(x)
			if(executeActivationFunctionOverFeatures):
				if(self.useCNNlayers):
					x = pt.reshape(x, (numberOfSamples, numberOfSublayers, x.shape[1], x.shape[2], x.shape[3]))
				else:
					x = pt.reshape(x, (numberOfSamples, numberOfSublayers, x.shape[1]))
		elif(activationFunctionType=="relu"):
			x = activationFunction(x)
		elif(activationFunctionType=="none"):
			pass
		if(not parallelStreams):
			if(self.useCNNlayers):
				x = pt.reshape(x, (x.shape[0], x.shape[1]*x.shape[2], x.shape[3], x.shape[4]))
			else:
				x = pt.reshape(x, (x.shape[0], x.shape[1]*x.shape[2]))
	else:
		if(activationFunctionType!="none"):
			x = activationFunction(x)
	return x
			
def getUseLinearSublayers(self, layerIndex):
	result = False
	if(useLinearSublayers):
		if(outputLayerInList):
			if(layerIndex != self.config.numberOfLayers-1):	#final layer does not useLinearSublayers
				result = True
		else:
			result = True
	return result

def weightsSetLayer(self, layerIndex, linear, sign=True):
	if(useSignedWeights):
		weightsSetSignLayer(self, layerIndex, linear, sign)
	if(useCustomWeightInitialisation):
		if(getUseLinearSublayers(self, layerIndex)):
			nn.init.normal_(linear.segregatedLinear.weight, mean=Wmean, std=WstdDev)
		else:
			nn.init.normal_(linear.weight, mean=Wmean, std=WstdDev)
	if(useCustomBiasInitialisation):
		if(getUseLinearSublayers(self, layerIndex)):
			nn.init.constant_(linear.segregatedLinear.bias, 0)
		else:
			nn.init.constant_(linear.bias, 0)
		if(useCustomBiasNoTrain):
			linear.bias.requires_grad = False

def weightsFixLayer(self, layerIndex, linear, sign=True):
	#if(not trainLastLayerOnly):
	if(not usePositiveWeightsClampModel):
		weightsSetSignLayer(self, layerIndex, linear, sign)
			
def weightsSetSignLayer(self, layerIndex, linear, sign=True):
	if(getUseLinearSublayers(self, layerIndex)):
		weights = linear.segregatedLinear.weight #only sign weights allowed
		weights = pt.abs(weights)
		if(not sign):
			weights = -weights
		linear.segregatedLinear.weight = pt.nn.Parameter(weights)
	else:
		weights = linear.weight #only sign weights allowed
		weights = pt.abs(weights)
		if(not sign):
			weights = -weights
		linear.weight = pt.nn.Parameter(weights)
	if(debugUsePositiveWeightsVerify):
		if(getUseLinearSublayers(self, layerIndex)):
			weights = linear.segregatedLinear.weight
			bias = linear.segregatedLinear.bias
		else:
			weights = linear.weight
			bias = linear.bias
		if(numpyVersion == 2):
			printLayerWeights("weight", weights)
			printLayerWeights("bias", bias)	
		else:
			print("weights = ", weights)
			print("bias = ", bias)

def printLayerWeights(name, weights):
	if(numpyVersion == 2):
		weights_numpy = weights.detach().cpu().numpy()
		np.set_printoptions(formatter={'all': lambda x: custom_formatter(x)})
		print(name, " = ", weights_numpy)
	else:
		print(name, " = ", weights)

def custom_formatter(array):
    return f"{array}"
		
def weightsSetPositiveModel(self):
	if(useSignedWeights):
		if(usePositiveWeightsClampModel):
			for p in self.parameters():
				p.data.clamp_(0)

class OffsetReLU(nn.Module):
	def __init__(self, offset):
		super(OffsetReLU, self).__init__()
		self.offset = offset

	def forward(self, x):
		if(debugPrintActivationOutput):
			print("OffsetReLU: x = ", x)
		#print("self.offset = ", self.offset)
		x = pt.max(pt.zeros_like(x), x - self.offset)
		return x

class OffsetSoftmax(nn.Module):
	def __init__(self, offset):
		super(OffsetSoftmax, self).__init__()
		self.offset = offset
		self.softmax = nn.Softmax(dim=1)

	def forward(self, x):
		if(debugPrintActivationOutput):
			print("OffsetSoftmax: x = ", x)
		#print("self.offset = ", self.offset)
		x = self.softmax(x)
		if(debugPrintActivationOutput):
			print("OffsetSoftmax: x after softmax = ", x)
		x = pt.max(pt.zeros_like(x), x - self.offset)
		return x

def executeResidual(self, l1, l2, Z, AprevLayer):
	if(supportSkipLayersResidual):
		# only do a residual add if shapes already match
		if AprevLayer.shape == Z.shape:
			Z = Z + AprevLayer
	return Z

def executeBatchNormLayer(self, layerIndex, x, batchNorm, batchNormFC):
	if(useCNNlayers):
		assert not getUseLinearSublayers(self, layerIndex)
		if(layerIndex < numberOfConvlayers):
			if(CNNmaxPool):
				x = convReshapeIntermediateMaxPool(self, x, layerIndex)
			elif(CNNbatchNorm):
				x = convReshapeIntermediate(self, x, layerIndex)
			if(CNNbatchNorm):
				x = self.batchNorm[layerIndex](x)
		else:
			if(batchNormFC):
				idx = layerIndex - numberOfConvlayers
				x = self.batchNormFC[idx](x)
	return x

def executeBatchNormLayer(self, layerIndex, x, batchNorm, batchNormFC):
	if(useCNNlayers):
		assert not getUseLinearSublayers(self, layerIndex)
		if(layerIndex < numberOfConvlayers):
			if(CNNbatchNorm):
				x = self.batchNorm[layerIndex](x)
		else:
			if(batchNormFC):
				idx = layerIndex - numberOfConvlayers
				x = self.batchNormFC[idx](x)
	return x

def executeDropoutLayer(self, layerIndex, x, batchNorm):
	if(dropout):
		x = self.dropout[layerIndex](x)
	return x

def executeMaxPoolLayer(self, layerIndex, x, maxPool):
	if(useCNNlayers):
		assert not getUseLinearSublayers(self, layerIndex)
		if(layerIndex < numberOfConvlayers):
			if(CNNmaxPool):
				if(not (CNNconvergeEveryEvenLayer and layerIndex%2 == 0)):
					x = self.maxPool(x)
	return x

def convReshapeIntermediate(self, layerIndex, x, supportMaxPool=True):
	if(useCNNlayers):
		assert not getUseLinearSublayers(self, layerIndex)
		if(layerIndex < numberOfConvlayers):
			if(CNNmaxPool and supportMaxPool):
				x = convReshapeIntermediateMaxPool(self, x, layerIndex)
			else:
				x = convReshapeIntermediateNoMaxPool(self, x, layerIndex)
	return x

def convReshapeIntermediateMaxPool(self, x, layerIndex):
	if(CNNconvergeEveryEvenLayer and layerIndex%2 == 0):
		return convReshapeIntermediateNoMaxPool(self, x, layerIndex)
	else:
		out_features, in_channels, out_channels, in_width, in_height, out_width, out_height, numberInputPaddedFeatures = getCNNproperties(self, layerIndex)
		batch_size = x.shape[0]
		if(useCNNlayers2D):
			x = x.reshape(batch_size, out_channels, out_height*CNNinputWidthDivisor, out_width*CNNinputWidthDivisor)
		else:
			x = x.reshape(batch_size, out_channels, out_width*CNNinputWidthDivisor)
	return x
	
def convReshapeIntermediateNoMaxPool(self, x, layerIndex):
	out_features, in_channels, out_channels, in_width, in_height, out_width, out_height, numberInputPaddedFeatures = getCNNproperties(self, layerIndex)
	batch_size = x.shape[0]
	if(useCNNlayers2D):
		x = x.reshape(batch_size, out_channels, out_height, out_width)
	else:
		x = x.reshape(batch_size, out_channels, out_width)
	return x

def convReshapeIn(self, x, layerIndex):
	out_features, in_channels, out_channels, in_width, in_height, out_width, out_height, numberInputPaddedFeatures = getCNNproperties(self, layerIndex)
	batch_size = x.shape[0]
	if(useCNNlayers2D):
		x = x.reshape(batch_size, in_channels, in_height, in_width)
	else:
		x = x.reshape(batch_size, in_channels, in_width)
	return x

def convReshapeFinal(self, layerIndex, x):
	if(useCNNlayers):
		assert not getUseLinearSublayers(self, layerIndex)
		if(layerIndex < numberOfConvlayers):
			x = convReshapeOut(self, x, layerIndex)
	return x

def convReshapeOut(self, x, layerIndex):
	out_features, in_channels, out_channels, in_width, in_height, out_width, out_height, numberInputPaddedFeatures = getCNNproperties(self, layerIndex)
	batch_size = x.shape[0]
	#model code always assumes data dimensions are flattened;
	if useImageDataset:
		x = x.reshape(batch_size, -1)
	else:
		x = x.reshape(batch_size, out_features)
	return x

