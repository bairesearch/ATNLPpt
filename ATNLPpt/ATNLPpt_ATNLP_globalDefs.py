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

debugSequentialLoops = False
debugATNLPnormalisation = False
debugATNLPcomparison = False
debugATNLPkeypoints = False
debugSkipFirstBatch = False	#skips the first dataset batch (sample) where batchSize=1 for debug, as this contains very few keypoints at start of sequence

useNLPDataset = True	#mandatory
useNLPDatasetPaddingMask = True	#default: True	#not strictly required for wikipedia dataset and small sequence lengths, as it extracts only the first L tokens from each article

enforceConfigBatchSize = True	#required such that (B1 and) B2 can be determined at initialisation (not dynamic)
debugOnlyPrintStreamedWikiArticleTitles = False

ATNLPcompareUntransformedTokenPredictionStrict = False	#dependent var (initialisation only)

import torch as pt
if pt.cuda.is_available():
	deviceGPU = pt.device("cuda")
else:
	deviceGPU = pt.device("cpu")
deviceCPU = pt.device("cpu")

referenceSetPosDelimiterTypes = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]	#prep: "IN", "TO"	#conj: "CC", ",", ";"
sentenceCharDelimiterTypes = [".", "?", "!"]
paragraphCharDelimiterTypes = ["\n"]

ATNLPusePredictionHead = True	#use ML model (transformer/wavenet) as a next token prediction head
if(ATNLPusePredictionHead):
	ATNLPtiePredictionHeadEncoderDecoderWeights = False
	ATNLPcompareUntransformedTokenPrediction = False	#default: False	#train a predictive network with untransformed token prediction (ie standard transformer implementation)	#dev only
	if(ATNLPcompareUntransformedTokenPrediction):
		ATNLPcompareUntransformedTokenPredictionStrict = True	#generateSequenceInput does not expand (bert) tokens to characters
		ATNLPuseMultiLevelTokenPrediction = False
		ATNLPmultiLevelOnlyPredictLastLevel = False
		ATNLPmultiLevels = 1
	else:
		ATNLPuseMultiLevelTokenPrediction = True	#optional	#predicts char/subword (bert), subsentence (reference set), sentence, paragraph tokens
		if(ATNLPuseMultiLevelTokenPrediction):
			ATNLPmultiLevelOnlyPredictLastLevel = False	#default: False	#only perform prediction across last level of token generation
			ATNLPmultiLevels = 3
			ATNLPmultiLevelTokensDelimiterNames = ['pos', 'eos', 'eop']
			ATNLPmultiLevelTokensDelimiterTypes = ['pos', 'char', 'char']
			ATNLPmultiLevelTokens = ['referenceSets', 'sentences', 'paragraphs']	#if !ATNLPuseSequenceLevelPrediction, prediction targets = ['subwords', 'referenceSets', 'sentences']; or if ATNLPuseSequenceLevelPrediction: prediction targets = ['referenceSets', 'sentences', 'paragraphs']
			ATNLPmultiLevelTokensDelimiters = [referenceSetPosDelimiterTypes, sentenceCharDelimiterTypes, paragraphCharDelimiterTypes]
		else:
			ATNLPmultiLevelOnlyPredictLastLevel = False
			ATNLPmultiLevels = 1
	ATNLPuseSequenceLevelPrediction = False	#optional	#predicts sequences (eg reference sets) rather than normalised tokens	#if !ATNLPuseSequenceLevelPrediction, prediction target = 'subwords'; or if ATNLPuseSequenceLevelPrediction: prediction target = 'referenceSets'
	backboneType = "transformer" #"transformer", "wavenet"
	reorderPairsToBeNotReversed = True	#default: True - prediction head may expect ordered normalised snapshots as input 
	optimiserAdamW = True
	useCustomLearningAlgorithm = False
	trainLocalIndividialLayers = False	#train once per prediction head model execution
	d_model = 128	#normalised snapshot token encoding size
	useSlidingWindow = False	#does not use sliding window during training
else:
	ATNLPuseMultiLevelTokenPrediction = False	#mandatory: False
	ATNLPmultiLevels = 1
	ATNLPuseSequenceLevelPrediction = False	#mandatory: False
	backboneType = "none"
	useCustomLearningAlgorithm = True	#mandatory (disable all backprop optimisers)
	reorderPairsToBeNotReversed = False	#default: False (process last normalised snapshot first as this is special; it is only defined by 1 reference set delimiter keypoint)
	useSlidingWindow = True		#mandatory
	ATNLPcompareUntransformedTokenPrediction = False
trainLocal = True	#local learning rule	#required
	
ATNLPindexDatabaseByClassTarget = True	#optional	#overload normalised snapshots with same class target	#orig: False
ATNLPindexDatabaseByReferenceSetIndex = True	#mandatory: True	#orig: False
ATNLPindexDatabaseByReferenceSetDelimiterToken = True	#mandatory: True	#orig: False
if(ATNLPindexDatabaseByClassTarget):
	ATNLPrenormaliseTransformedSnapshots = False	#default: True	- incomplete (requires suitable renormalisation function) #currently renormalises transformed snapshots by overload number	#explore other renomalisation functions, eg tanh (this has problems), changing all normalised snapshots to binary, etc
	
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
	if(ATNLPindexDatabaseByClassTarget):
		ATNLPsnapshotDatabaseDiskCompareChunks = False	#mandatory: False
	else:
		ATNLPsnapshotDatabaseDiskCompareChunks = True	#mandatory: True
	if(ATNLPsnapshotDatabaseDiskCompareChunks):
		ATNLPsnapshotDatabaseDiskChunkSize = 1000000
		ATNLPsnapshotDatabaseDiskCompareChunksSize = ATNLPsnapshotDatabaseDiskChunkSize
		ATNLPsnapshotDatabaseLoadDevice = deviceCPU	#default: deviceCPU	
		snapshotDatabaseNamePrepend = "train_db"
		snapshotDatabaseNameExtension = ".h5"
	else:
		ATNLPsnapshotCompareChunkSize = None
		ATNLPsnapshotDatabaseLoadDevice = deviceGPU #default: deviceGPU	
		snapshotDatabaseNamePrepend = "train_db"
		snapshotDatabaseNamePrependNumber = "train_db_number"
		snapshotDatabaseNameExtension = ".pkl"
	ATNLPsnapshotDatabaseRamDynamic = False	#mandatory: False
elif(ATNLPsnapshotDatabaseRam):
	ATNLPsnapshotDatabaseDiskCompareChunks = False	#mandatory: False
	ATNLPsnapshotCompareChunkSize = None	#chunking is advantageous only if the whole flattened database (plus the similarity matrix) cannot fit on your GPU; otherwise chunking just adds loop overhead.
	ATNLPsnapshotDatabaseLoadDevice = deviceGPU	#default: deviceGPU
	ATNLPsnapshotDatabaseRamDynamic = True	#optional #very slow but enables train predictions	#if(!ATNLPindexDatabaseByClassTarget): continuously update database tensor (do not use intermediary python list)	#useful for debug (required for prediction performance during train)	#debug only
	saveAndLoadModel = False	#self.database can be large so do not save it to disk
deviceSparse = ATNLPsnapshotDatabaseLoadDevice
	
bertModelName = "bert-base-uncased"	#bertModelName = "bert-large-uncased"
bertNumberTokenTypes = 30522	#tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")	print(len(tokenizer))

useNLPcharacterInput = False		#default: False, recommended for ATNLPcompareUntransformedTokenPrediction (discrete token prediction comparison)
if(useNLPcharacterInput):
	if(ATNLPcompareUntransformedTokenPrediction):
		 ATNLPcompareUntransformedTokenPredictionStrict = True	#char input never involves expansion to bert tokens
	
useNLPDatasetMultipleTokenisation = True	#mandatory: True	#required for spacy tokenisation
if(useNLPDatasetMultipleTokenisation):
	if(ATNLPcompareUntransformedTokenPrediction):
		useNLPDatasetMultipleTokenisationSpacy = False
	else:
		useNLPDatasetMultipleTokenisationSpacy = True
	if(useNLPcharacterInput):
		useNLPDatasetMultipleTokenisationChar = True	#mandatory: True
		useNLPDatasetMultipleTokenisationBert = False	#optional
	else:
		useNLPDatasetMultipleTokenisationChar = False	#optional
		useNLPDatasetMultipleTokenisationBert = True	#mandatory: True
	if(useNLPDatasetMultipleTokenisationChar): 
		useNLPcharacterInputBasic = True	#if True: only use a basic lowercase+punctuation character set of 30 chars, else if False: use a full printable subset of ASCII-128
		if(useNLPcharacterInputBasic):
			NLPcharacterInputBasicSet = [' ', 'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','Q','R','s','t','u','v','w','x','y','z','.','(',')', ',']	#select 31 characters for normalcy
			NLPcharacterInputSetLen = len(NLPcharacterInputBasicSet)+1	#32	# 0 reserved for PAD (NLPpadTokenID)
		else:
			NLPcharacterInputSetLen = 98	  # full printable subset of ASCII-128	# 0 reserved for PAD (NLPpadTokenID)

if(useNLPcharacterInput):
	useContinuousVarEncodeMethod = "onehot"	#just convert character id directly to onehot vector
	useTokenEmbedding = False	#token embeddings are not used (one-hot vectors instead)
	ATNLPcontinuousVarEncodingNumBits = NLPcharacterInputSetLen
else:
	useContinuousVarEncodeMethod = "onehot"
	useTokenEmbedding = False	#token embeddings are not used (one-hot vectors instead)
	ATNLPcontinuousVarEncodingNumBits = bertNumberTokenTypes
if(useNLPcharacterInput):
	contextSizeMax = 512*4	#default: 2048	#useNLPcharacterInput requires less memory
else:
	contextSizeMax = 128*4	#default: 512	#production: 512*4	#specified in characters	#assume approx 4 characters per BERT token
contextSizeMaxCharacters = contextSizeMax	
contextSizeMaxBertTokens = contextSizeMax//2	#safe only (max)	#average: //4	- wikipedia average token length
contextSizeMaxSpacyTokens = contextSizeMax//4	#safe only (max)	#average: //6	- wikipedia average word length

numberOfClasses = ATNLPcontinuousVarEncodingNumBits

sequenceLength = contextSizeMax
NLPpadTokenID = 0		#must be same as bert pad token id	#assert bert_tokenizer.pad_token_id == NLPpadTokenID
NLPmaskTokenID = 103	#must be same as bert mask token id	#assert bert_tokenizer.mask_token_id == NLPmaskTokenID	#used to identify predicted tokens in normalised snapshot during predictive network training only (eg transformer)	#not used

inputDataNames = ["char_input_ids", "bert_input_ids", "bert_offsets", "spacy_input_ids", "spacy_pos", "spacy_tag", "spacy_offsets"]	

#encoding vars;
C = ATNLPcontinuousVarEncodingNumBits	#vocabulary size

#sequence length vars;
if(ATNLPcompareUntransformedTokenPredictionStrict and not useNLPcharacterInput):
	L1 = contextSizeMaxBertTokens
else:
	L1 = sequenceLength

#keypoint extraction vars;
keypointModes = Literal["allKeypointCombinations", "firstKeypointConsecutivePairs", "firstKeypointPairs"]
if(ATNLPusePredictionHead):
	R, Q, L2 = ([None]*ATNLPmultiLevels, [None]*ATNLPmultiLevels, [None]*ATNLPmultiLevels)
	for l in range(ATNLPmultiLevels):
		R[l] = 5*(ATNLPmultiLevels-l)	#default: 10	#the last R (user defined) set of 2 consecutive keypoints in batch sequence	#max number of reference sets in batch sequence (if less reference sets detected in sequence some normalised snapshots will be filled with zeros)
		Q[l] = 1	#the last R (user defined) set of 2 keypoints (of distance Q) in batch sequence
		L2[l] = 8	#default: 8	#normalisation length for each reference set
	if(ATNLPuseMultiLevelTokenPrediction and ATNLPmultiLevelOnlyPredictLastLevel):
		#assume contextSizeMax = 128*4	#default: 512 
		L2[0] = 8*4	#assume approx 4 characters per BERT token	#~8*4 characters per reference set
		L2[1] = 8	#~8 reference sets per sentence
		L2[2] = 8	#~8 sentences per paragraph
else:
	R = 3	#the last R (user defined) set of 2 consecutive keypoints in batch sequence
	Q = 1   #the last R (user defined) set of 2 keypoints (of distance Q) in batch sequence
	L2 = 8	#default: 8	#normalisation length for each reference set

referenceSetPosDelimitersStr = []
if(ATNLPuseMultiLevelTokenPrediction):
	for name in ATNLPmultiLevelTokensDelimiterNames:
		if(name == 'pos'):	#l=1
			referenceSetPosDelimitersStr.append(referenceSetPosDelimiterTypes)
			#TODO: collapse auxiliary verbs (eg tagged as VBZ/VBD) with adjacent VBN into single ref set delimiter; eg has [VBZ] run [VBN], had [VBD] gone [VBN] -> has_run [VBZ], had_gone [VBD]
		elif(name == 'eos'):	#l=2
			referenceSetPosDelimitersStr.append(sentenceCharDelimiterTypes)
		elif(name == 'eop'):	#l=3
			referenceSetPosDelimitersStr.append(paragraphCharDelimiterTypes)	#new line/paragaph added for ATNLPuseMultiLevelTokenPrediction compatibility (paragraph sequences)
else:
	referenceSetPosDelimitersTagStr = referenceSetPosDelimiterTypes
	referenceSetPosDelimitersTextStr = sentenceCharDelimiterTypes + paragraphCharDelimiterTypes
	

#keypointMode="firstKeypointConsecutivePairs"	 #out shape = (B1*R, C, L2)
keypointMode="firstKeypointPairs"	 	#out shape = (B1*R*Q, C, L2)	#default (requires testing)

if(not ATNLPusePredictionHead):
	if(keypointMode == "firstKeypointConsecutivePairs"):
		S = R
	elif(keypointMode == "firstKeypointPairs"):
		S = R*Q
	#def getS(R, Q):
	#return S
		
#sublayer paramters:	
simulatedDendriticBranches = False	#optional	#performTopK selection of neurons based on local inhibition - equivalent to multiple independent fully connected weights per neuron (SDBANN)
useLinearSublayers = False

workingDrive = '/large/source/ANNpython/ATNLPpt/'
dataDrive = workingDrive	#'/datasets/'

modelName = 'modelATNLP'
databaseFolderName = '../database/'

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


