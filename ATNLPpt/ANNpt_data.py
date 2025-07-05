"""ANNpt_data.py

# Author:
Richard Bruce Baxter - Copyright (c) 2023-2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ANNpt_main.py

# Usage:
see ANNpt_main.py

# Description:
ANNpt data 

"""


import torch as pt
from datasets import load_dataset, Value
from ANNpt_globalDefs import *
import ANNpt_globalDefs
import numpy as np
import random
if(useTabularDataset):
	pass
elif(useImageDataset):
	import torchvision
	import torchvision.transforms as transforms
elif(useNLPDataset):
	from transformers import AutoTokenizer, DataCollatorWithPadding
	from torch.utils.data import IterableDataset as TorchIterableDataset, DataLoader
	from datasets import Dataset, IterableDataset as HFDIterable, get_dataset_config_info, DatasetDict
	bert_tokenizer, bert_pad_id = None, None
	import string
	import spacy
	from collections import defaultdict
	nlp = None                       # lazy-loaded spaCy pipeline
	
if(disableDatasetCache):
	import datasets
	from datasets import disable_caching
	disable_caching()
	datasets.config.IN_MEMORY_MAX_SIZE = 32 * 10**9	#in bytes

debugSaveRawDatasetToCSV = False	#output dataset to csv file for manual checks
debugSaveSplitDatasetToCSV = False	#output dataset to csv file for manual checks
debugSaveNormalisedDatasetToCSV = False	#output dataset to csv file for manual checks

def loadDataset():
	if(useTabularDataset):
		return loadDatasetTabular()
	elif(useImageDataset):
		return loadDatasetImage()
	elif(useNLPDataset):
		return loadDatasetNLP()

def saveDatasetToCSV(dataset):
	for split in dataset.keys():  # Usually 'train', 'test', etc.
		output_file = f"{datasetName}_{split}.csv"
		dataset[split].to_csv(output_file)
		print(f"Saved {split} split to {output_file}")


def countNumberClasses(dataset, printSize=False):
	numberOfClassSamples = {}
	datasetSize = getDatasetSize(dataset)
	if(useTabularDataset):
		numberOfClasses = 0
		for i in range(datasetSize):
			if useTabularDataset:
				# For Hugging Face datasets, the label is accessed via the classFieldName
				row = dataset[i]
				target = int(row[classFieldName])
			elif useImageDataset:
				# For PyTorch datasets (e.g., CIFAR10), the label is the second element of the dataset tuple
				_, target = dataset[i]
			else:
				raise AttributeError("Unsupported dataset type: Unable to count classes.")

			if(target in numberOfClassSamples):
				numberOfClassSamples[target] = numberOfClassSamples[target] + 1
			else:
				numberOfClassSamples[target] = 0

			#print("target = ", target)
			if(target > numberOfClasses):
				numberOfClasses = target
		numberOfClasses = numberOfClasses+1
	elif(useImageDataset):
		numberOfClasses = ANNpt_globalDefs.numberOfClasses
		numberOfClassSamples = None	#not used
	elif(useNLPDataset):
		numberOfClasses = ANNpt_globalDefs.numberOfClasses
		numberOfClassSamples = None	#not used

	#if(printSize):
	#print("numberOfClasses = ", numberOfClasses)
	return numberOfClasses, numberOfClassSamples

def countNumberFeatures(dataset, printSize=False):
	if useTabularDataset:
		# For tabular datasets, use the features attribute
		numberOfFeatures = len(dataset.features) - 1  # -1 to ignore class targets
	elif useImageDataset:
		# For image datasets, infer the number of features from the image shape
		sample_image, _ = dataset[0]  # Get the first sample (image, label)
		numberOfFeatures = sample_image.shape[0]*sample_image.shape[1]*sample_image.shape[2]
	elif useNLPDataset:
		if(useTokenEmbedding):
			numberOfFeatures = contextSizeMax * embeddingSize
		else:
			numberOfFeatures = contextSizeMax
	else:
		raise AttributeError("Unsupported dataset type: Unable to determine the number of features.")

	if(printSize):
		print("numberOfFeatures = ", numberOfFeatures)
	return numberOfFeatures

def getDatasetSize(dataset, printSize=False):
	if useTabularDataset:
		# Otherwise, assume it's a Hugging Face dataset
		datasetSize = dataset.num_rows
	elif useImageDataset:
		# Check if the dataset is a PyTorch dataset (e.g., CIFAR10)
		datasetSize = len(dataset)
	elif useNLPDataset:
		datasetSize = datasetSizeRecord
	else:
		raise AttributeError("Unsupported dataset type: Unable to determine the number of features.")

	if(printSize):
		print("datasetSize = ", datasetSize)
	return datasetSize

def createFieldTypeList(dataset):
	if(useTabularDataset):
		fieldTypeList = [fieldType.dtype for fieldType in dataset.features.values() if hasattr(fieldType, 'dtype')]
	else:
		fieldTypeList = None
	return fieldTypeList
	

if(useTabularDataset):

	def loadDatasetTabular():
		if(datasetLocalFile):
			trainFileNameFull = dataPathName + '/' + trainFileName
			if(datasetHasTestSplit):
				testFileNameFull = dataPathName + '/' +  testFileName
				dataset = load_dataset('csv', data_files={"train":trainFileNameFull, "test":testFileNameFull})
			else:
				dataset = load_dataset('csv', data_files={"train":trainFileNameFull})
		else:
			if(datasetSpecifyDataFiles):
				if(datasetHasTestSplit):
					dataset = load_dataset(datasetNameFull, data_files={"train":trainFileName, "test":testFileName})
				else:
					dataset = load_dataset(datasetNameFull, data_files={"train":trainFileName})
			elif(datasetHasSubsetType):
				dataset = load_dataset(datasetNameFull, datasetSubsetName)
			else:
				dataset = load_dataset(datasetNameFull)

		if(debugCullDatasetSamples):
			dataset['train'] = dataset['train'].select(range(100))
			if(datasetHasTestSplit):
				dataset['test'] = dataset['test'].select(range(100))

		if(debugSaveRawDatasetToCSV):
			saveDatasetToCSV(dataset)

		if(datasetConvertFeatureValues):
			dataset[datasetSplitNameTrain] = convertFeatureValues(dataset[datasetSplitNameTrain])
			if(datasetHasTestSplit):
				dataset[datasetSplitNameTest] = convertFeatureValues(dataset[datasetSplitNameTest])
		if(datasetConvertClassValues):
			dataset[datasetSplitNameTrain] = convertClassValues(dataset[datasetSplitNameTrain])
			if(datasetHasTestSplit):
				dataset[datasetSplitNameTest] = convertClassValues(dataset[datasetSplitNameTest])
		else:
			if(datasetConvertClassTargetColumnFloatToInt):
				dataset[datasetSplitNameTrain] = convertClassTargetColumnFloatToInt(dataset[datasetSplitNameTrain])
				if(datasetHasTestSplit):
					dataset[datasetSplitNameTest] = convertClassTargetColumnFloatToInt(dataset[datasetSplitNameTest])

		if(not datasetHasTestSplit):
			dataset = dataset[datasetSplitNameTrain].train_test_split(test_size=datasetTestSplitSize, shuffle=datasetTestSplitShuffle, seed=datasetTestSplitSeed)

		if(debugSaveSplitDatasetToCSV):
			saveDatasetToCSV(dataset)

		if(datasetEqualiseClassSamples):
			dataset[datasetSplitNameTrain] = equaliseClassSamples(dataset[datasetSplitNameTrain])
			if(datasetHasTestSplit and datasetEqualiseClassSamplesTest):
				dataset[datasetSplitNameTest] = equaliseClassSamples(dataset[datasetSplitNameTest])
		if(datasetNormalise):
			dataset[datasetSplitNameTrain] = normaliseDataset(dataset[datasetSplitNameTrain])
			dataset[datasetSplitNameTest] = normaliseDataset(dataset[datasetSplitNameTest])
		if(datasetRepeat):
			dataset[datasetSplitNameTrain] = repeatDataset(dataset[datasetSplitNameTrain])
			dataset[datasetSplitNameTest] = repeatDataset(dataset[datasetSplitNameTest])
		if(datasetShuffle):
			dataset[datasetSplitNameTrain] = shuffleDataset(dataset[datasetSplitNameTrain])
			dataset[datasetSplitNameTest] = shuffleDataset(dataset[datasetSplitNameTest])
		if(datasetOrderByClass):
			dataset[datasetSplitNameTrain] = orderDatasetByClass(dataset[datasetSplitNameTrain])
			dataset[datasetSplitNameTest] = orderDatasetByClass(dataset[datasetSplitNameTest])
			#dataset = orderDatasetByClass(dataset)

		dataset[datasetSplitNameTrain] = repositionClassFieldToLastColumn(dataset[datasetSplitNameTrain])
		dataset[datasetSplitNameTest] = repositionClassFieldToLastColumn(dataset[datasetSplitNameTest])

		if(debugSaveNormalisedDatasetToCSV):
			saveDatasetToCSV(dataset)

		return dataset

	def equaliseClassSamples(dataset):
		#Equalises the number of samples across each class by repeating class samples as necessary.

		_, numberOfClassSamples = countNumberClasses(dataset) #dict: {class_label: count}
		if not numberOfClassSamples: # Handles empty or single-class datasets gracefully
			printe("No classes found in the dataset.")

		max_samples = 0
		if numberOfClassSamples: # Ensure there's at least one class to avoid error with max() on empty sequence
			max_samples = max(numberOfClassSamples.values())
		if max_samples == 0: # All classes have 0 samples or no classes
			printe("No samples found in any class.")

		class_specific_indices = {class_val: [] for class_val in numberOfClassSamples.keys()}
		for i in range(getDatasetSize(dataset)):
			row = dataset[i]
			# Assuming classFieldName holds the key to the class label and it's convertible to int
			# This part might need adjustment based on how class labels are stored/accessed
			try:
				target = int(row[classFieldName])
				if target in class_specific_indices:
					class_specific_indices[target].append(i)
			except (KeyError, ValueError) as e:
				print(f"Warning: Could not process class label for row {i}: {e}")
				continue # Skip rows where class label is problematic

		all_new_indices = []
		for class_val, count in numberOfClassSamples.items():
			current_indices = class_specific_indices.get(class_val, [])
			all_new_indices.extend(current_indices) # Add existing samples

			num_to_add = max_samples - count
			if num_to_add > 0 and current_indices: # Check if samples need to be added and if source samples exist
				repeated_indices = random.choices(current_indices, k=num_to_add)
				all_new_indices.extend(repeated_indices)

		if all_new_indices:
			# Shuffle to mix original and repeated samples, then select.
			# random.shuffle(all_new_indices) # Optional: shuffle before select if desired, otherwise select preserves order then shuffleDataset handles it later
			dataset = dataset.select(all_new_indices)

		return dataset

	def repositionClassFieldToLastColumn(dataset):
		classDataList = dataset[classFieldName]
		dataset = dataset.remove_columns(classFieldName)
		dataset = dataset.add_column(classFieldName, classDataList)
		return dataset

	def convertClassTargetColumnFloatToInt(dataset):
		classDataList = dataset[classFieldName]
		classDataList = [int(value) for value in classDataList]
		dataset = dataset.remove_columns(classFieldName)
		dataset = dataset.add_column(classFieldName, classDataList)
		return dataset

	def normaliseDataset(dataset):
		print("normaliseDataset:  dataset.num_rows = ",  dataset.num_rows, ", len(dataset.features) = ", len(dataset.features))
		datasetSize = getDatasetSize(dataset)
		for featureIndex, featureName in enumerate(list(dataset.features)):
			#print("featureIndex = ", featureIndex)
			if(featureName != classFieldName):
				fieldType = dataset.features[featureName]
				if hasattr(fieldType, 'dtype') and fieldType.dtype == 'bool':
					continue #skip normalisation for boolean fields
				if(datasetCorrectMissingValues):
					featureDataList = []
					for i in range(datasetSize):
						row = dataset[i]
						featureCell = row[featureName]
						if(featureCell == None):
							featureCell = 0
						featureDataList.append(featureCell)
				else:
					featureDataList = dataset[featureName]
				featureData = np.array(featureDataList)
				#print("featureData = ", featureData)
				if(datasetNormaliseMinMax):
					featureMin = np.amin(featureData)
					featureMax = np.amax(featureData)
					#if(featureMax - featureMin == 0):
					#	print("warning: (featureMax - featureMin == 0)")
					featureData = (featureData - featureMin) / (featureMax - featureMin + 1e-8) #featureData/featureMax
				elif(datasetNormaliseStdAvg):
					featureMean = np.mean(featureData)
					featureStd = np.std(featureData)
					featureData = featureData-featureMean
					featureData = featureData/featureStd
				featureDataList = featureData.tolist()
				dataset = dataset.remove_columns(featureName)
				dataset = dataset.add_column(featureName, featureDataList)
		return dataset

	def repeatDataset(dataset):
		datasetSize = getDatasetSize(dataset)
		repeatIndices = list(range(datasetSize))
		repeatIndices = repeatIndices*datasetRepeatSize
		dataset = dataset.select(repeatIndices)
		return dataset

	def shuffleDataset(dataset):
		datasetSize = getDatasetSize(dataset)
		dataset = dataset.shuffle()
		return dataset

	def orderDatasetByClass(dataset):
		dataset = dataset.sort(classFieldName)
		return dataset

	def convertFeatureValues(dataset):
		print("convertFeatureValues:  dataset.num_rows = ",  dataset.num_rows, ", len(dataset.features) = ", len(dataset.features))
		for fieldName, fieldType in dataset.features.items():
			#print("convertFeatureValues: fieldName = ", fieldName)
			if fieldType.dtype == 'string':
				dataset = convertCategoricalFieldValues(dataset, fieldName, dataType=float)
			#elif fieldType.dtype == 'bool':
			#	dataset = dataset.cast_column(fieldName, Value('float32'))
		return dataset

	def bool_to_float(example):
		example[fieldName] = float(example[fieldName])
		return example

	def convertClassValues(dataset):
		return convertCategoricalFieldValues(dataset, classFieldName, dataType=int)

	def convertCategoricalFieldValues(dataset, fieldName, dataType=float):
		if(not (dataType==float or dataType==int)):
			printe("convertCategoricalFieldValues error: not (dataType==float or dataType==int)")

		#print("convertCategoricalFieldValues: fieldName = ", fieldName)
		fieldIndex = 0
		fieldIndexDict = {}
		fieldNew = []
		datasetSize = getDatasetSize(dataset)
		#print("datasetSize = ", datasetSize)
		numberOfClasses = 0

		for i in range(datasetSize):
			row = dataset[i]
			targetString = row[fieldName]
			if(targetString not in fieldIndexDict):
				fieldIndexDict[targetString] = fieldIndex
				fieldIndex = fieldIndex + 1		

		booleanCategoryDetected = False
		if(fieldIndex == 2):
			booleanCategoryDetected = True

		for i in range(datasetSize):
			row = dataset[i]
			targetString = row[fieldName]
			target = fieldIndexDict[targetString]
			if(dataType==int):	#always store class target as int (never bool)
				target = int(target)	#keep as int (redundant)
			elif(booleanCategoryDetected):
				target = bool(target)
			elif(dataType==float):
				target = float(target)
			fieldNew.append(target)

		dataset = dataset.remove_columns(fieldName)
		dataset = dataset.add_column(fieldName, fieldNew)

		return dataset

	def normaliseBooleanFieldValues(dataset, fieldName, dataType=float):
		if(not (dataType==float or dataType==int)):
			printe("normaliseBooleanFieldValues error: not (dataType==float or dataType==int)")

		fieldNew = []
		datasetSize = getDatasetSize(dataset)
		for i in range(datasetSize):
			row = dataset[i]
			print("i = ", i)
			targetBool = row[fieldName]
			if(targetBool == True):
				target = 1
			elif(targetBool == False):
				target = 0
			if(dataType==float):
				target = float(target)
			fieldNew.append(target)

		dataset = dataset.remove_columns(fieldName)
		dataset = dataset.add_column(fieldName, fieldNew)

		return dataset

	def createDataLoader(dataset):
		return createDataLoaderTabular(dataset)

	def createDataLoaderTabular(dataset):
		dataLoaderDataset = DataloaderDatasetTabular(dataset)	
		maintainEvenBatchSizes = True
		if(dataloaderRepeatSampler):
			numberOfSamples = getDatasetSize(dataset)*dataloaderRepeatSize
			if(dataloaderRepeatSamplerCustom):
				sampler = CustomRandomSampler(dataset, shuffle=True, num_samples=numberOfSamples)
			else:
				sampler = pt.utils.data.RandomSampler(dataset, replacement=True, num_samples=numberOfSamples)
			loader = pt.utils.data.DataLoader(dataset=dataLoaderDataset, batch_size=batchSize, drop_last=dataloaderMaintainBatchSize, sampler=sampler)
		else:
			loader = pt.utils.data.DataLoader(dataset=dataLoaderDataset, batch_size=batchSize, shuffle=dataloaderShuffle, drop_last=dataloaderMaintainBatchSize)
		return loader

	def createDataLoaderTabularPaired(dataset1, dataset2):
		dataLoaderDataset = DataloaderDatasetTabularPaired(dataset1, dataset2)	
		if(dataloaderRepeatSampler):
			numberOfSamples = getDatasetSize(dataset1)*dataloaderRepeatSize
			if(dataloaderRepeatSamplerCustom):
				sampler = CustomRandomSampler(dataset1, shuffle=True, num_samples=numberOfSamples)
			else:
				sampler = pt.utils.data.RandomSampler(dataset1, replacement=True, num_samples=numberOfSamples)
			loader = pt.utils.data.DataLoader(dataset=dataLoaderDataset, batch_size=batchSize, drop_last=dataloaderMaintainBatchSize, sampler=sampler)
		else:
			loader = pt.utils.data.DataLoader(dataset=dataLoaderDataset, batch_size=batchSize, shuffle=dataloaderShuffle, drop_last=dataloaderMaintainBatchSize)
		return loader

	class DataloaderDatasetTabular(pt.utils.data.Dataset):
		def __init__(self, dataset):
			self.datasetSize = getDatasetSize(dataset)
			self.dataset = dataset
			self.datasetIterator = iter(dataset)

		def __len__(self):
			return self.datasetSize

		def __getitem__(self, i):
			if(dataloaderRepeatSampler):
				#index = i % self.datasetSize
				#document = self.dataset[index]
				try:
					document = next(self.datasetIterator)	#does not support dataloaderShuffle
				except StopIteration:
					self.datasetIterator = iter(self.dataset)
					document = next(self.datasetIterator)	#does not support dataloaderShuffle
			else:
				 document = self.dataset[i]	
				 #document = next(self.datasetIterator) #does not support dataloaderShuffle
			documentList = list(document.values())
			if(datasetReplaceNoneValues):
				documentList = [x if x is not None else 0 for x in documentList]
			#print("documentList = ", documentList)
			x = documentList[0:-1]
			y = documentList[-1]
			x = pt.Tensor(x).float()
			batchSample = (x, y)
			return batchSample

	class DataloaderDatasetTabularPaired(pt.utils.data.Dataset):
		def __init__(self, dataset1, dataset2):
			self.datasetSize = getDatasetSize(dataset1)
			self.dataset1 = dataset1
			self.dataset2 = dataset2
			self.datasetIterator1 = iter(dataset1)
			self.datasetIterator2 = iter(dataset2)

		def __len__(self):
			return self.datasetSize

		def __getitem__(self, i):
			if(dataloaderRepeatSampler):
				#index = i % self.datasetSize
				#document1 = self.dataset1[index]
				#document2 = self.dataset2[index]
				try:
					document1 = next(self.datasetIterator1)	#does not support dataloaderShuffle
					document2 = next(self.datasetIterator2)	#does not support dataloaderShuffle
				except StopIteration:
					self.datasetIterator1 = iter(self.dataset1)
					self.datasetIterator2 = iter(self.dataset2)
					document1 = next(self.datasetIterator1)	#does not support dataloaderShuffle
					document2 = next(self.datasetIterator2)	#does not support dataloaderShuffle
			else:
				document1 = self.dataset1[i]
				document2 = self.dataset2[i]	
				#document1 = next(self.datasetIterator1)	#does not support dataloaderShuffle
				#document2 = next(self.datasetIterator2)	#does not support dataloaderShuffle
			documentList1 = list(document1.values())
			documentList2 = list(document2.values())
			if(datasetReplaceNoneValues):
				documentList1 = [x if x is not None else 0 for x in documentList1]
				documentList2 = [x if x is not None else 0 for x in documentList2]
			#print("documentList = ", documentList)
			x1 = documentList1[0:-1]
			x2 = documentList2[0:-1]
			x1 = pt.Tensor(x1).float()
			x2 = pt.Tensor(x2).float()
			x1 = pt.unsqueeze(x1, dim=0)
			x2 = pt.unsqueeze(x2, dim=0)
			x = pt.concat([x1, x2], dim=0)
			y1 = documentList1[-1]
			y2 = documentList2[-1]
			#print("y1 = ", y1, ", y2 = ", y2)	#verify they are equal
			y = y1
			batchSample = (x, y)
			return batchSample

	class CustomRandomSampler(pt.utils.data.Sampler):
		def __init__(self, dataset, shuffle, num_samples):
			self.dataset = dataset
			self.shuffle = shuffle
			self.num_samples = num_samples

		def __iter__(self):
			order = list(range((getDatasetSize(self.dataset))))
			idx = 0
			sampleIndex = 0
			while sampleIndex < self.num_samples:
				#print("idx = ", idx)
				#print("order[idx] = ", order[idx])
				yield order[idx]
				idx += 1
				if idx == len(order):
					if self.shuffle:
						random.shuffle(order)
					idx = 0
				sampleIndex += 1

elif(useImageDataset):

	def loadDatasetImage():
		# Load the CIFAR-10 dataset with optional augmentation
		if imageDatasetAugment:
			train_transform = transforms.Compose([
				transforms.RandomCrop(32, padding=4),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
				transforms.Lambda(lambda img: cutout(img))
			])
		else:
			train_transform = transforms.Compose([
				transforms.ToTensor(),
				transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
			])
		test_transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		])
		dataset = {}
		if(datasetName=="CIFAR10"):
			dataset[datasetSplitNameTrain] = torchvision.datasets.CIFAR10(root=dataPathName, train=True, download=True, transform=train_transform)
			dataset[datasetSplitNameTest] = torchvision.datasets.CIFAR10(root=dataPathName, train=False, download=True, transform=test_transform)
		else:
			printe("loadDatasetImage currently requires datasetName==CIFAR10")

		return dataset

	def createDataLoaderImage(dataset):
		loader = pt.utils.data.DataLoader(dataset, batch_size=batchSize, shuffle=True)
		return loader

	def cutout(img, n_holes=1, length=16):
		"""Apply cutout augmentation to a tensor image."""
		h, w = img.size(1), img.size(2)
		mask = np.ones((h, w), np.float32)
		for _ in range(n_holes):
			y = random.randrange(h)
			x = random.randrange(w)
			y1 = max(0, y - length//2)
			y2 = min(h, y + length//2)
			x1 = max(0, x - length//2)
			x2 = min(w, x + length//2)
			mask[y1:y2, x1:x2] = 0.
		mask = pt.from_numpy(mask)
		mask = mask.expand_as(img)
		return img * mask

elif(useNLPDataset):

	def loadDatasetNLP():
		info = get_dataset_config_info(datasetName, datasetCfg)  # tiny JSON download
		datasetSize = info.splits["train"].num_examples
		base_stream = load_dataset(datasetName, datasetCfg, split="train", streaming=True, trust_remote_code=True)

		if(stateTestDataset):
			assert datasetSizeSubset, "loadDatasetNLP error: if stateTestDataset, datasetSizeSubset is required"
			if(not stateTrainDataset):
				print("loadDatasetNLP warning: stateTestDataset and !stateTrainDataset: assume train rows already streamed and cached (else will take long time to download train data before can start streaming test data)")
			#train_rows = int(datasetSize*(1-datasetTestSplitSize))
			#eval_rows = int(datasetSize*datasetTestSplitSize)
			train_rows = datasetTrainRows
			eval_rows = datasetTestRows
			train_stream = base_stream.take(train_rows)
			test_stream = base_stream.skip(train_rows).take(eval_rows)
			datasetSize = train_rows
		else:
			if(datasetSizeSubset):
				train_rows = datasetTrainRows
				datasetSize = train_rows
			else:
				train_rows =  int(datasetSize)
			eval_rows = int(0)
			train_stream = base_stream
			test_stream = None

		print(f"Train size: {train_rows:,}")
		print(f"Eval size: {eval_rows:,}")
		global datasetSizeRecord
		datasetSizeRecord = datasetSize

		dataset = DatasetDict({datasetSplitNameTrain: train_stream, datasetSplitNameTest: test_stream})

		return dataset

	if(useNLPDatasetSelectTokenisation):
		def encode(batch):
			texts = batch["text"]
			if useNLPcharacterInput:
				out_ids = []
				for txt in texts:
					if useNLPcharacterInputBasic:
						txt = txt.lower()
					ids = []
					for ch in txt:
						if ch in _CHAR2ID:
							ids.append(_CHAR2ID[ch])
					out_ids.append(ids[:contextSizeMax])   # truncate, no pad yet
				return {"input_ids": out_ids}
			else:	
				global bert_tokenizer, bert_pad_id   # only used in BERT mode
				if bert_tokenizer is None:
					bert_tokenizer = AutoTokenizer.from_pretrained(bertModelName, use_fast=True)
					bert_pad_id	= bert_tokenizer.pad_token_id
					assert bert_pad_id == NLPcharacterInputPadTokenID
				enc = bert_tokenizer(texts, truncation=True, max_length=contextSizeMax, padding=False)
				if(debugOnlyPrintStreamedWikiArticleTitles):
					print(batch["title"])
				return enc   # already {"input_ids": [...], ...}

		def collate(batch):
			# batch_size==1 in your new set-up but keep it generic
			seqs   = [item for item in batch]
			pad_id = NLPcharacterInputPadTokenID
			max_L  = max(len(s) for s in seqs)
			padded = pt.full((len(seqs), max_L), pad_id, dtype=pt.long)
			for i, s in enumerate(seqs):
				padded[i, : len(s)] = s
			x = padded
			y = padded	 # no target yet (redundant)
			return x, y

		class RawSampleDataset(TorchIterableDataset):
			 """
			 Pass-through: yields one dict {'input_ids': Tensor[seq_len]} per article.
			 No left-shift / crop logic here any more.
			 """
			 def __init__(self, hf_iterable):
				 super().__init__()
				 self.hf_ds = hf_iterable

			 def __iter__(self):
				 for art in self.hf_ds:
					 ids = art["input_ids"]
					 if not isinstance(ids, pt.Tensor):
						 ids = pt.tensor(ids, dtype=pt.long)
					 yield ids
	else:
		def encode(batch):
			"""
			Returns per-text dictionaries with
	    		bert_input_ids  : List[int]
	    		bert_offsets    : List[(start,end)]      (char alignment)
	    		char_input_ids  : List[int]
	    		spacy_input_ids : List[int]              (orth IDs)
	    		spacy_pos       : List[int]              (POS enum)
	    		spacy_offsets   : List[(start,end)]
			"""
			global bert_tokenizer, bert_pad_id, nlp

			texts = batch["text"]
			if bert_tokenizer is None:
				bert_tokenizer = AutoTokenizer.from_pretrained(bertModelName, use_fast=True)
				bert_pad_id = bert_tokenizer.pad_token_id
			if nlp is None:
				nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "lemmatizer"])

			enc = bert_tokenizer(
				texts,
				truncation=True,
				max_length=contextSizeMax,
				padding=False,
				return_offsets_mapping=True,
			)

			out = defaultdict(list)
			for i, txt in enumerate(texts):
				# --- BERT ---
				out["bert_input_ids"].append(enc["input_ids"][i])
				out["bert_offsets"].append(enc["offset_mapping"][i])

				# --- characters ---
				char_ids = [ _CHAR2ID[ch] for ch in (txt.lower() if useNLPcharacterInputBasic else txt)
		            		 if ch in _CHAR2ID ][:contextSizeMax]
				out["char_input_ids"].append(char_ids)

				# --- spaCy ---
				doc = nlp(txt)

				# Fit into signed-64 range so torch.tensor() never overflows
				def to_int64(u):                                 # keep sign if already < 2^63
					return u if u < (1 << 63) else u - (1 << 64) # two\u2019s-complement wrap
				sp_ids = [to_int64(tok.lex_id) for tok in doc][:contextSizeMax]
				sp_pos = [to_int64(tok.tag)    for tok in doc][:contextSizeMax]

				sp_off = [ (tok.idx, tok.idx+len(tok)) for tok in doc ][:contextSizeMax]
				out["spacy_input_ids"].append(sp_ids)
				out["spacy_pos"].append(sp_pos)
				out["spacy_offsets"].append(sp_off)

			return out

		def collate(batch):
			B = len(batch)
			# helper -------------------------------------------------------------
			def pad1d(seqs, pad_id):
				L = max(len(s) for s in seqs)
				out = pt.full((B, L), pad_id, dtype=pt.long)
				for i, s in enumerate(seqs):
					out[i, :len(s)] = pt.tensor(s, dtype=pt.long)
				return out
			def pad2d(seqs):				# for offset pairs
				L = max(len(s) for s in seqs)
				out = pt.full((B, L, 2), -1, dtype=pt.long)
				for i, s in enumerate(seqs):
					out[i, :len(s)] = pt.tensor(s, dtype=pt.long)
				return out
			# -------------------------------------------------------------------
			char_ids   = pad1d([s["char_input_ids"]   for s in batch], NLPcharacterInputPadTokenID)
			bert_ids   = pad1d([s["bert_input_ids"]   for s in batch], bert_pad_id)
			bert_off   = pad2d([s["bert_offsets"]     for s in batch])
			spacy_ids  = pad1d([s["spacy_input_ids"]  for s in batch], 0)
			spacy_pos  = pad1d([s["spacy_pos"]        for s in batch], 0)
			spacy_off  = pad2d([s["spacy_offsets"]    for s in batch])

			x = {
				"char_input_ids" : char_ids,
				"bert_input_ids" : bert_ids,
				"bert_offsets"   : bert_off,
				"spacy_input_ids": spacy_ids,
				"spacy_pos"      : spacy_pos,
				"spacy_offsets"  : spacy_off,
			}
			y = None	#dynamically extracted from x
			return x, y

		class RawSampleDataset(TorchIterableDataset):
			"""
			Pass-through: yields one dict {'input_ids': Tensor[seq_len]} per article.
			No left-shift / crop logic here any more.
			"""
			def __init__(self, hf_iterable):
				super().__init__()
				self.hf_ds = hf_iterable

			def __iter__(self):
				for art in self.hf_ds:                          # art is already a dict from encode_multi
					# No heavy work here \u2013 just forward the fields your collate() needs.
					# If you prefer tensors early, uncomment the few lines below.

					# for key in ("char_input_ids", "bert_input_ids", "spacy_input_ids", "spacy_pos"):
					#     art[key] = pt.tensor(art[key], dtype=pt.long)
					# art["bert_offsets"]  = pt.tensor(art["bert_offsets"],  dtype=pt.long)
					# art["spacy_offsets"] = pt.tensor(art["spacy_offsets"], dtype=pt.long)

					yield art

			''' 
			"char_input_ids" : (B, Lc),
			"bert_input_ids" : (B, Lb),
			"bert_offsets"   : (B, Lb, 2),
			"spacy_input_ids": (B, Ls),
			"spacy_pos"      : (B, Ls),
			"spacy_offsets"  : (B, Ls, 2),
			'''

	def createDataLoaderNLP(dataset: "Dataset | HFDIterable"):
		"""Return DataLoader that yields (x, y) batches per the spec above."""

		ds_tok = dataset.map(encode, batched=True, remove_columns=dataset.column_names)
		ds_tok = ds_tok.with_format("torch")

		# If the result is map-style convert it, otherwise keep it as-is
		if isinstance(ds_tok, HFDIterable):
			ds_iter = ds_tok                      # already iterable -> nothing to do
		else:
			ds_iter = ds_tok.to_iterable_dataset()   # map-style -> convert

		ds = RawSampleDataset(ds_iter)

		loader = DataLoader(ds, batch_size=batchSize, collate_fn=collate, num_workers=numWorkers, pin_memory=pt.cuda.is_available())

		return loader

	def ascii_printable_with_whitespace() -> list[str]:
		"""
		Return ASCII chars 0-127 with all control codes removed
		except the standard whitespace set:
			e.g. space (32), TAB (9), LF (10), [CR (13), VT (11), FF (12)]
		The order is stable: whitespace first, then 32-126 printable.
		"""
		# whitelist the whitespace control chars you want to keep
		whitespace_keep = {' ', '\t', '\n'}		#{' ', '\t', '\n', '\r', '\v', '\f'}

		chars = []
		# 0-127 inclusive
		for code in range(128):
			ch = chr(code)
			# keep if printable or whitelisted whitespace
			if ch.isprintable() or ch in whitespace_keep:
				chars.append(ch)
		return chars

	def _build_char_tables():
		if useNLPcharacterInputBasic:
			table = {c: i+1 for i, c in enumerate(NLPcharacterInputBasicSet)}  # 0 reserved for PAD (NLPcharacterInputPadTokenID)
			rev   = {i+1: c for i, c in enumerate(NLPcharacterInputBasicSet)}
		else:
			# Drop all control codes (0-31, 127) but keep whitespace
			allowed = ascii_printable_with_whitespace()
			assert len(allowed) == NLPcharacterInputSetLen-1	# -1 explanation; 0 reserved for PAD (NLPcharacterInputPadTokenID)
			table = {c: idx+1 for idx, c in enumerate(allowed)}	 #0 reserved for PAD (NLPcharacterInputPadTokenID)
			rev   = {idx+1: c for idx, c in enumerate(allowed)}
		return table, rev

	_CHAR2ID, _ID2CHAR = _build_char_tables()
