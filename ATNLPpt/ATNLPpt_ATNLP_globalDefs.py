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
debugATNLPnormalisation = False
debugATNLPcomparison = True
debugATNLPkeypoints = False
debugSkipFirstBatch = False	#skips the first dataset batch (sample) where batchSize=1 for debug, as this contains very few keypoints at start of sequence

useNLPDataset = True	#mandatory
useNLPDatasetPaddingMask = True	#default: True	#not strictly required for wikipedia dataset and small sequence lengths, as it extracts only the first L tokens from each article

enforceConfigBatchSize = True	#required such that (B1 and) B2 can be determined at initialisation (not dynamic)
debugOnlyPrintStreamedWikiArticleTitles = False

import torch as pt
if pt.cuda.is_available():
	deviceGPU = pt.device("cuda")
deviceCPU = pt.device("cpu")

ATNLPnormalisedSnapshotsSparseTensors = True	#mandatory	#required to perform image comparison against a database of any significant size at speed
ATNLPcomparisonShiftInvariance = False	#default: True	#orig: False	#add redundancy to support incomplete alignment between candidate and database normalised snapshots
ATNLPcomparisonShiftInvariancePixels = None
if(ATNLPcomparisonShiftInvariance):
	ATNLPcomparisonShiftInvariancePixels = 2

if(debugATNLPcomparison):
	generateConnectionsAfterPropagating = False
	debugSkipFirstBatch = True	#temp	#skips the first dataset batch (sample) where batchSize=1 for debug, as this contains very few keypoints at start of sequence
else:
	generateConnectionsAfterPropagating = True	#default: True	#relevant for ATNLPsnapshotDatabaseRamDynamic only	#set to False for debug only - adds the current normalisedSnapshots to the self.database before executing compare_1d_batches() 

ATNLPsnapshotDatabaseDisk = False	#slow and high capacity
ATNLPsnapshotDatabaseRam = True 	#fast and low capacity
if(ATNLPsnapshotDatabaseDisk):
	ATNLPsnapshotDatabaseDiskChunkSize = 1000000
	ATNLPsnapshotDatabaseDiskCompareChunksSize = ATNLPsnapshotDatabaseDiskChunkSize
	ATNLPsnapshotDatabaseLoadDevice = deviceCPU	#default: deviceCPU	
	snapshotDatabaseNamePrepend = "train_db"
	snapshotDatabaseNameExtension = ".h5"
	ATNLPsnapshotDatabaseRamDynamic = False	#mandatory: False
elif(ATNLPsnapshotDatabaseRam):
	ATNLPsnapshotCompareChunkSize = None	#chunking is advantageous only if the whole flattened database (plus the similarity matrix) cannot fit on your GPU; otherwise chunking just adds loop overhead.
	ATNLPsnapshotDatabaseLoadDevice = deviceGPU	#default: deviceGPU
	if(debugATNLPcomparison):
		ATNLPsnapshotDatabaseRamDynamic = True	#optional #very slow but enables train predictions	#continuously update database tensor (do not use intermediary python list)	#useful for debug (required for prediction performance during train)	#debug only
	else:
		ATNLPsnapshotDatabaseRamDynamic = False

useSlidingWindow = True	#enables sliding window	#mandatory

bertModelName = "bert-base-uncased"	#bertModelName = "bert-large-uncased"
bertNumberTokenTypes = 30522	#tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")	print(len(tokenizer))

useNLPcharacterInput = False
if(useNLPcharacterInput):
	useContinuousVarEncodeMethod = "onehot"	#just convert character id directly to onehot vector
	useTokenEmbedding = False	#token embeddings are not used (one-hot vectors instead)
	ATNLPcontinuousVarEncodingNumBits = NLPcharacterInputSetLen
else:
	useContinuousVarEncodeMethod = "onehot"
	useTokenEmbedding = False	#token embeddings are not used (one-hot vectors instead)
	ATNLPcontinuousVarEncodingNumBits = bertNumberTokenTypes
contextSizeMax = 128*4	#default: 512	#production: 512*4	#specified in characters	#assume approx 4 characters per BERT token
contextSizeMaxCharacters = contextSizeMax	
contextSizeMaxBertTokens = contextSizeMax//2	#safe only (max)	#average: //4	- wikipedia average token length
contextSizeMaxSpacyTokens = contextSizeMax//2	#safe only (max)	#average: //6	- wikipedia average word length

numberOfClasses = ATNLPcontinuousVarEncodingNumBits

sequenceLength = contextSizeMax
NLPcharacterInputPadTokenID = 0	#must be same as bert pad token id	#assert bert_tokenizer.pad_token_id == NLPcharacterInputPadTokenID

inputDataNames = ["char_input_ids", "bert_input_ids", "bert_offsets", "spacy_input_ids", "spacy_pos", "spacy_tag", "spacy_offsets"]	

#encoding vars;
C = ATNLPcontinuousVarEncodingNumBits	#vocabulary size

#sequence length vars;
L1 = sequenceLength
L2 = 10	#default: 10	#normalisation length for each reference set

#keypoint extraction vars;
keypointModes = Literal["allKeypointCombinations", "firstKeypointConsecutivePairs", "firstKeypointPairs"]
r = 3	#the last r (user defined) set of 2 consecutive keypoints in batch sequence
q = 4   #the last r (user defined) set of 2 keypoints (of distance 2->q) in batch sequence               

#referenceSetPosDelimitersPosStr = {"PUNCT", "VERB", "ADP"}
#referenceSetPosDelimitersPosStr = {"PUNCT", "VERB"}
##referenceSetPosDelimitersTagStr = {".", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "IN", "TO", "CC", ",", ";"}       # identical to TF version
referenceSetPosDelimitersTagStr = {".", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"}	#verbs only	#TODO: collapse auxiliary verbs (eg tagged as VBZ/VBD) with adjacent VBN into single ref set delimiter; eg has [VBZ] run [VBN], had [VBD] gone [VBN] -> has_run [VBZ], had_gone [VBD]
keypointMode="firstKeypointConsecutivePairs"	 #out shape = (B1*r, C, L2)
#keypointMode="firstKeypointPairs"	 	#out shape = (B1*r*(q-1), C, L2)	#default (requires testing)
if(keypointMode == "firstKeypointPairs"):
	S = r
elif(keypointMode == "firstKeypointConsecutivePairs"):
	S = r*(q-1)
		
useNLPDatasetMultipleTokenisation = True	#mandatory: True	#required for spacy tokenisation
if(useNLPDatasetMultipleTokenisation):
	useNLPDatasetMultipleTokenisationSpacy = True	#mandatory: True
	if(useNLPcharacterInput):
		useNLPDatasetMultipleTokenisationChar = True	#mandatory: True
		useNLPDatasetMultipleTokenisationBert = False	#optional
	else:
		useNLPDatasetMultipleTokenisationChar = False	#optional
		useNLPDatasetMultipleTokenisationBert = True	#mandatory: True
	if(useNLPDatasetMultipleTokenisationChar): 
		useNLPcharacterInputBasic = True	#if True: only use a basic lowercase+punctuation character set of 30 chars, else if False: use a full printable subset of ASCII-128
		if(useNLPcharacterInputBasic):
			NLPcharacterInputBasicSet = [' ', 'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','.','(',')', ',']	#select 31 characters for normalcy
			NLPcharacterInputSetLen = len(NLPcharacterInputBasicSet)+1	#32	# 0 reserved for PAD (NLPcharacterInputPadTokenID)
		else:
			NLPcharacterInputSetLen = 98	  # full printable subset of ASCII-128	# 0 reserved for PAD (NLPcharacterInputPadTokenID)

useCustomLearningAlgorithm = True	#mandatory (disable all backprop optimisers)
trainLocal = True	#local learning rule	#required

#sublayer paramters:	
simulatedDendriticBranches = False	#optional	#performTopK selection of neurons based on local inhibition - equivalent to multiple independent fully connected weights per neuron (SDBANN)
useLinearSublayers = False

workingDrive = '/large/source/ANNpython/ATNLPpt/'
dataDrive = workingDrive	#'/datasets/'

modelName = 'modelATNLP'

def lexIntToLexString(nlp, lexInt):
	if lexInt in nlp.vocab.strings:
		return nlp.vocab.strings[lexInt]
	else:
		return ''

def posIntToPosString(nlp, posInt):
	if posInt in nlp.vocab.strings:
		return nlp.vocab[posInt].text
	else:
		return ''
		
def posStringToPosInt(nlp, posString):
	return nlp.vocab.strings[posString]


