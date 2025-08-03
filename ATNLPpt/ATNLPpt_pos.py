"""ATNLPpt_pos.py

# Author:
Richard Bruce Baxter - Copyright (c) 2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ANNpt_main.py

# Usage:
see ANNpt_main.py

# Description:
ATNLPpt pos

"""

from ANNpt_globalDefs import *
import torch
import spacy
import os, csv

#load verb_dict/prep_dict (for keypoint detection):
nlp = spacy.load("en_core_web_sm", disable=("ner", "parser", "lemmatizer"))

if(ATNLPuseMultiLevelTokenPrediction):
	# Custom tokenizer to preserve \n as a token (paragraph delimiter)
	from spacy.tokenizer import Tokenizer
	from spacy.symbols import ORTH
	from spacy.tokens import Doc
	def custom_tokenizer(nlp):
		# Create a base tokenizer with default settings
		prefix_re = nlp.tokenizer.prefix_search
		suffix_re = nlp.tokenizer.suffix_search
		infix_re = nlp.tokenizer.infix_finditer
		token_match = nlp.tokenizer.token_match
		def tokenizer_func(text):
			words = []
			spaces = []
			lines = text.split('\n')
			for i, line in enumerate(lines):
				tokens = [t.text for t in nlp.tokenizer(line)]
				words.extend(tokens)
				spaces.extend([True] * len(tokens))
				if i < len(lines) - 1:
					words.append('\n')
					spaces.append(False)
			return Doc(nlp.vocab, words=words, spaces=spaces)
		return tokenizer_func
	# Replace tokenizer
	nlp.tokenizer = custom_tokenizer(nlp)

referenceSetPosDelimitersTagId = [None]*ATNLPmultiLevels
referenceSetPosDelimitersText = [None]*ATNLPmultiLevels
if(ATNLPuseMultiLevelTokenPrediction):
	for l in range(ATNLPmultiLevels):
		if(ATNLPmultiLevelTokensDelimiterTypes[l] == "pos"):
			referenceSetPosDelimitersTagId[l] = [posStringToPosInt(nlp, string) for string in referenceSetPosDelimitersStr[l]]
		else:
			referenceSetPosDelimitersTagId[l] = []
		if(ATNLPmultiLevelTokensDelimiterTypes[l] == "char"):
			referenceSetPosDelimitersText[l] = referenceSetPosDelimitersStr[l]
		else:
			referenceSetPosDelimitersText[l] = []
else:
	referenceSetPosDelimitersTagId[0] = [posStringToPosInt(nlp, string) for string in referenceSetPosDelimitersTagStr]
	referenceSetPosDelimitersText[0] = referenceSetPosDelimitersTextStr
#print("referenceSetPosDelimitersTagId = ", referenceSetPosDelimitersTagId)
#print("referenceSetPosDelimitersText = ", referenceSetPosDelimitersText)

verbPosId = posStringToPosInt(nlp, "VERB")
prepositionPosId = posStringToPosInt(nlp, "ADP")
punctPosId = posStringToPosInt(nlp, "PUNCT")
otherPosId = posStringToPosInt(nlp, "X")
VERB_DICT_PATH = "verb_dict.csv"
PREP_DICT_PATH = "prep_dict.csv"

def loadReferenceSetDelimDicts():
	# -------------------------------------------------
	# 1. Attempt to load cached dicts
	# -------------------------------------------------
	if os.path.isfile(VERB_DICT_PATH) and os.path.isfile(PREP_DICT_PATH):
		print("loading verb_dict  prep_dict from disk \u2026")
		verb_dict, prep_dict = {}, {}

		with open(VERB_DICT_PATH, newline='', encoding='utf-8') as f:
			reader = csv.reader(f)
			next(reader, None)	# skip optional header
			for word, idx in reader:
				verb_dict[word] = int(idx)

		with open(PREP_DICT_PATH, newline='', encoding='utf-8') as f:
			reader = csv.reader(f)
			next(reader, None)
			for word, idx in reader:
				prep_dict[word] = int(idx)
		result = True
	else:
		verb_dict = None
		prep_dict = None
		result = False
		
	return result, verb_dict, prep_dict
	
def generateReferenceSetDelimDicts():
	# -------------------------------------------------
	# 2. Rebuild dicts from spaCy vocab and cache them
	# -------------------------------------------------
	print("building verb_dict  prep_dict from nlp.vocab.strings; this will take approximately 1 minute")
	verb_dict = {}
	prep_dict = {}

	for word in list(nlp.vocab.strings):
		if not word.isalpha():
			continue
		doc = nlp(word)
		if not doc:	# skip empty parses
			continue
		tok = doc[0]
		if tok.pos_ == "VERB":
			verb_dict[word] = len(verb_dict)
		elif tok.pos_ == "ADP":
			prep_dict[word] = len(prep_dict)
			
	return verb_dict, prep_dict
		
def saveReferenceSetDelimDicts(verb_dict, prep_dict):
	# -------------------------------------------------
	# 3. Save dicts to cache
	# -------------------------------------------------
	# Cache to CSV for future runs
	with open(VERB_DICT_PATH, "w", newline='', encoding='utf-8') as f:
		writer = csv.writer(f)
		writer.writerow(["word", "idx"])
		for w, i in verb_dict.items():
			writer.writerow([w, i])

	with open(PREP_DICT_PATH, "w", newline='', encoding='utf-8') as f:
		writer = csv.writer(f)
		writer.writerow(["word", "idx"])
		for w, i in prep_dict.items():
			writer.writerow([w, i])

result, verb_dict, prep_dict = loadReferenceSetDelimDicts()
if(not result):
	verb_dict, prep_dict = generateReferenceSetDelimDicts()
	saveReferenceSetDelimDicts(verb_dict, prep_dict)
#print("verb_dict = ", verb_dict)
