"""ATNLPpt_ATNLP.py

# Author:
Richard Bruce Baxter - Copyright (c) 2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ANNpt_main.py

# Usage:
see ANNpt_main.py

# Description:
ATNLPpt Axis Transformation Natural Language Processing (ATNLP)

"""

from ANNpt_globalDefs import *
from torchsummary import summary
import ATNLPpt_ATNLPmodel
import ANNpt_data


def createModel(dataset):
	datasetSize = ANNpt_data.getDatasetSize(dataset, printSize=False)
	numberOfFeatures = ANNpt_data.countNumberFeatures(dataset)	#note "numberOfFeatures" is the raw continuous var input (without x-bit encoding)	#not used
	numberOfClasses, numberOfClassSamples = ANNpt_data.countNumberClasses(dataset)
	fieldTypeList = ANNpt_data.createFieldTypeList(dataset)

	if(printATNLPmodelProperties):
		print("Creating new model:")
		print("\t ---")
		print("\t stateTrainDataset = ", stateTrainDataset)
		print("\t stateTestDataset = ", stateTestDataset)
		print("\t ---")
		print("\t datasetName = ", datasetName)
		print("\t datasetSize = ", datasetSize)
		print("\t datasetSizeSubsetName = ", datasetSizeSubsetName)
		print("\t datasetTrainRows = ", datasetTrainRows)
		print("\t datasetTestRows = ", datasetTestRows)
		print("\t trainNumberOfEpochs = ", trainNumberOfEpochs)
		print("\t ---")
		print("\t batchSize = ", batchSize)
		print("\t inputLayerSize (numberOfFeatures) = ", numberOfFeatures)
		print("\t outputLayerSize (numberOfClasses) = ", numberOfClasses)
		print("\t ---")
		print("\t ATNLPnormalisedSnapshotsSparseTensors = ", ATNLPnormalisedSnapshotsSparseTensors)
		print("\t ATNLPcomparisonShiftInvariance = ", ATNLPcomparisonShiftInvariance)
		print("\t ATNLPcomparisonShiftInvariancePixels = ", ATNLPcomparisonShiftInvariancePixels)
		print("\t ATNLPsnapshotDatabaseDisk = ", ATNLPsnapshotDatabaseDisk)
		print("\t ATNLPsnapshotDatabaseRamDynamic = ", ATNLPsnapshotDatabaseRamDynamic)
		print("\t ATNLPsnapshotDatabaseRamStatic = ", ATNLPsnapshotDatabaseRamStatic)
		print("\t ATNLPsnapshotDatabaseDiskCompareChunks = ", ATNLPsnapshotDatabaseDiskCompareChunks)
		print("\t ---")

	config = ATNLPpt_ATNLPmodel.ATNLPconfig(
		batchSize = batchSize,
		numberOfLayers = numberOfLayers,
		hiddenLayerSize = hiddenLayerSize,
		inputLayerSize = numberOfFeatures,
		outputLayerSize = numberOfClasses,
		numberOfFeatures = numberOfFeatures,
		numberOfClasses = numberOfClasses,
		fieldTypeList = fieldTypeList,
	)
	model = ATNLPpt_ATNLPmodel.ATNLPmodel(config)
	print(model)

	return model




