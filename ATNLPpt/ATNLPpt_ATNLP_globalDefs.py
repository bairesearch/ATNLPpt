"""ATNLPpt_ATNLP_globalDefs.py

# Author:
Richard Bruce Baxter - Copyright (c) 2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ANNpt_main.py

# Usage:
see ANNpt_main.py

# Description:
ATNLPpt globalDefs

"""

import math
from typing import Literal
			
printATNLPmodelProperties = True
debugSequentialLoops = True
debugATNLPnormalisation = True

useNLPDataset = True	#mandatory

debugOnlyPrintStreamedWikiArticleTitles = False

useSlidingWindow = True	#enables sliding window	#mandatory

useNLPcharacterInputBasic = True	#if True: only use a basic lowercase+punctuation character set of 30 chars, else if False: use a full printable subset of ASCII-128
useContinuousVarEncodeMethod = "onehot"	#just convert character id directly to onehot vector
if(useNLPcharacterInputBasic):
	NLPcharacterInputBasicSet = [' ', 'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','.','(',')', ',']	#select 31 characters for normalcy
	NLPcharacterInputSetLen = len(NLPcharacterInputBasicSet)+1	#32	# 0 reserved for PAD (NLPcharacterInputPadTokenID)
else:
	NLPcharacterInputSetLen = 98	  # full printable subset of ASCII-128	# 0 reserved for PAD (NLPcharacterInputPadTokenID)
ATNLPcontinuousVarEncodingNumBits = NLPcharacterInputSetLen

bertModelName = "bert-base-uncased"	#bertModelName = "bert-large-uncased"
bertNumberTokenTypes = 30522	#tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")	print(len(tokenizer))
useTokenEmbedding = False	#use character embeddings for primary normalisation process (token embeddings are only used for prediction)

contextSizeMax = 128*4	#default: 512	#production: 512*4	#assume approx 4 characters per BERT token
numberOfClasses = bertNumberTokenTypes	#or NLPcharacterInputSetLen

sequenceLength = contextSizeMax
NLPcharacterInputPadTokenID = 0	#must be same as bert pad token id	#assert bert_tokenizer.pad_token_id == NLPcharacterInputPadTokenID

inputDataNames = ["char_input_ids", "bert_input_ids", "bert_offsets", "spacy_input_ids", "spacy_pos", "spacy_offsets"]	

#encoding vars;
C = ATNLPcontinuousVarEncodingNumBits	#vocabulary size

#sequence length vars;
L1 = sequenceLength
L2 = 10	#normalisation length for each reference set

#keypoint extraction vars;
keypointModes = Literal["allKeypointCombinations", "firstKeypointConsecutivePairs", "firstKeypointPairs"]
r = 3	#the last r (user defined) set of 2 consecutive keypoints in batch sequence
q = 4   #the last r (user defined) set of 2 keypoints (of distance 2->q) in batch sequence               
referenceSetPosDelimiters = {".", "CC", "IN", "TO", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", ",", ";"}       # identical to TF version

#batch size vars;
#B1 = batchSize
#B2 is encapsulates the number of normalisations (sets of 2 keypoints); either b) B1*r or c) B1*r*(q-1).


useNLPDatasetSelectTokenisation = False
useCustomLearningAlgorithm = True	#mandatory (disable all backprop optimisers)
trainLocal = True	#local learning rule	#required

#sublayer paramters:	
simulatedDendriticBranches = False	#optional	#performTopK selection of neurons based on local inhibition - equivalent to multiple independent fully connected weights per neuron (SDBANN)
useLinearSublayers = False

workingDrive = '/large/source/ANNpython/ATNLPpt/'
dataDrive = workingDrive	#'/datasets/'

modelName = 'modelATNLP'

