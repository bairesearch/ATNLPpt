"""ATNLPpt_keypoints.py

# Author:
Richard Bruce Baxter - Copyright (c) 2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ANNpt_main.py

# Usage:
see ANNpt_main.py

# Description:
ATNLPpt keypoints

# Description
# -----------
# PyTorch re-write of ATNLP normalisation with upgrades 1-6:
#   (1) last token is always a key-point -> see _append_keypoints_last_token()
#   (2) key-points are obtained with spaCy; we store POS, start & end char
#   (3) generate key-point pairs:
#		 a) dev  : every ordered pair
#		 b) train: last R adjacent pairs
#		 c) eval : last R starts, span size 2->Q		  -> see _make_pairs()
#	   Missing pairs are zero-filled as required
# 

"""

from __future__ import annotations
from ANNpt_globalDefs import *
import torch
import torch.nn.functional as F
import spacy
import os, csv
from typing import List, Dict, Tuple, Literal

nlp = spacy.load("en_core_web_sm", disable=("ner", "parser", "lemmatizer"))
referenceSetPosDelimitersTagId = [posStringToPosInt(nlp, string) for string in referenceSetPosDelimitersTagStr]
#referenceSetPosDelimitersPosId = [posStringToPosInt(nlp, string) for string in referenceSetPosDelimitersPosStr]
verbPosId = posStringToPosInt(nlp, "VERB")
prepositionPosId = posStringToPosInt(nlp, "ADP")
punctPosId = posStringToPosInt(nlp, "PUNCT")
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


def build_keypoints(
	spacy_input_id : torch.Tensor,		  # (B1, L1)
	spacy_pos : torch.Tensor,		  # (B1, L1)
	spacy_tag : torch.Tensor,		  # (B1, L1)
	spacy_offsets : torch.Tensor,		  # (B1, L1, 2)
) -> Tuple[List[List[int]], List[List[Dict]]]:

	batchSize = spacy_tag.shape[0]
	kp_indices_batch = []
	kp_meta_batch = []
	
	for b in range(batchSize):
		kp_indices, kp_meta = _detect_keypoints(spacy_input_id[b], spacy_pos[b], spacy_tag[b], spacy_offsets[b])
		kp_indices.reverse()
		kp_meta.reverse()
		kp_indices_batch.append(kp_indices)
		kp_meta_batch.append(kp_meta)

	return kp_indices_batch, kp_meta_batch

# ------------------------------------------------------------------------- #
# (1) + (2)  spaCy-based key-point detector that always includes last token #
# ------------------------------------------------------------------------- #
def _detect_keypoints(	
	spacy_input_id : torch.Tensor,		  # (L1)
	spacy_pos : torch.Tensor,		  # (L1)
	spacy_tag : torch.Tensor,		  # (L1)
	spacy_offsets : torch.Tensor		  # (L1, 2)
) -> Tuple[List[int], List[Dict]]:
	"""
	Parameters
	----------
	spacy_input_id : torch.Tensor,		  # (L1)
	spacy_pos : torch.Tensor,		  # (L1)
	spacy_tag : torch.Tensor,		  # (L1)
	spacy_offsets : torch.Tensor,		  # (L1, 2)
	
	Return
	------
	kp_indices : token indices that are key-points
	kp_meta	   : [{token_idx, spacy_pos, spacy_tag, spacy_input_id, char_start, char_end}, ...]
	"""
	kp_indices, kp_meta = [], []

	L1 = spacy_tag.shape[0]
	for i in range(L1):
		spacyInt = spacy_tag[i].item()
		#print("i = ", i, ", spacyInt = ", spacyInt)
		is_kp = spacyInt in referenceSetPosDelimitersTagId or i == 0	#always treat first token in sequence as a keypoint
		if is_kp:
			#print("is_kp")
			kp_indices.append(i)
		kp_meta.append({
			"token_idx": i,
			"spacy_pos": spacy_pos[i],
			"spacy_tag": spacy_tag[i],
			"spacy_input_id": spacy_input_id[i],
			"char_start": spacy_offsets[i][0],	#tok.idx,
			"char_end": spacy_offsets[i][1],	#tok.idx + len(tok)
		})

	return kp_indices, kp_meta

def insert_keypoints_last_token(
	last_token_idx_sample: int,
	kp_use: List[int],
):
	"""
	Parameters
	----------
	last_token_idx_sample: int			#last spacy token index at which to perform keypoint detection]
	"""
	kp_use.insert(0, last_token_idx_sample)

# ------------------------------------------------------------------------- #
# (3)  build key-point pairs for dev | train | eval modes				   #
# ------------------------------------------------------------------------- #
def make_pairs(kp: List[int], mode: keypointModes, R: int, Q: int) -> Tuple[torch.Tensor, torch.Tensor]:
	"""
	Parameters
	----------
	kp   : sorted list of key-point token indices
	mode : "allKeypointCombinations" | "firstKeypointConsecutivePairs" | "firstKeypointPairs"
	R/Q  : user parameters (see requirement 3)

	Return
	------
	pairs	: (N, 2)  long tensor of [i, j] with i<j; invalid -> 0,0
	valid_ms : (N,)	bool tensor, True where pair is valid
	"""
	
	if len(kp) < 2:
		# zero-fill for the expected number of rows so caller can reshape
		if mode == "firstKeypointConsecutivePairs":
			N = R
		elif mode == "firstKeypointPairs":
			N = R * (Q - 1)
		elif mode == "allKeypointCombinations":		 # we let it be empty
			N = 0
		pairs = torch.zeros(N, 2, dtype=torch.long)
		valid = torch.zeros(N, dtype=torch.bool)
	else:
		# ------------  a) every ordered permutation -------------------- #
		if mode == "allKeypointCombinations":
			out = [(i, j) for idx_i, i in enumerate(kp) for j in kp[idx_i + 1:]]
			pairs = torch.as_tensor(out, dtype=torch.long)
			pairs = torch.sort(pairs, dim=1).values	# ensure i < j so downstream code always receives [start, end]
		# ------------  b) last R adjacent pairs ---------------------- #
		elif mode == "firstKeypointConsecutivePairs":
			adj = list(zip(kp[:-1], kp[1:]))			   # consecutive pairs
			adj = adj[:R]								 # keep the *first* R
			valid_n = len(adj)
			# zero-padding to fixed length R
			pairs = torch.zeros(R, 2, dtype=torch.long)
			if valid_n:
				pairs[:valid_n] = torch.as_tensor(adj, dtype=torch.long)
				pairs[:valid_n] = torch.sort(pairs[:valid_n], dim=1).values	# ensure i < j so downstream code always receives [start, end]
			valid = torch.zeros(R, dtype=torch.bool)
		# ------------  c) last R starts, spans 2 -> Q ------------------ #
		elif mode == "firstKeypointPairs":
			out = []
			starts = kp[:R]								# first R starts
			for s_idx, s in enumerate(starts):
				# pick up to Q-1 subsequent key-points
				next_kps = kp[s_idx + 1 : s_idx + 1 + (Q - 1)]
				if not next_kps:							# none available
					out.extend([(0, 0)] * (Q - 1))
				else:
					# pad to Q-1 so output length is constant
					for t in range(Q - 1):
						if t < len(next_kps):
							out.append((s, next_kps[t]))
						else:
							out.append((0, 0))
			pairs = torch.as_tensor(out, dtype=torch.long)		  # (R*(Q-1), 2)
			pairs = torch.sort(pairs, dim=1).values	# ensure i < j so downstream code always receives [start, end]
		else:
			raise ValueError(f"Unknown mode {mode}")
		valid = (pairs[:, 0] != pairs[:, 1])

	if reorderPairsToBeNotReversed and pairs.numel():
		_max = pairs.max().item() + 1
		order = torch.argsort(pairs[:, 0] * _max + pairs[:, 1])
		pairs = pairs[order]
		valid = valid[order]
	
	return pairs, valid

