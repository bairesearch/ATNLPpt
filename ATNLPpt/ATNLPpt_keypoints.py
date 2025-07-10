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
#		 b) train: last r adjacent pairs
#		 c) eval : last r starts, span size 2->q		  -> see _make_pairs()
#	   Missing pairs are zero-filled as required
# 

"""

from __future__ import annotations
from ANNpt_globalDefs import *
import torch
import torch.nn.functional as F
import spacy
from typing import List, Dict, Tuple, Literal

nlp = spacy.load("en_core_web_sm", disable=("ner", "parser", "lemmatizer"))
referenceSetPosDelimitersTagId = [posStringToPosInt(nlp, string) for string in referenceSetPosDelimitersTagStr]
#referenceSetPosDelimitersPosId = [posStringToPosInt(nlp, string) for string in referenceSetPosDelimitersPosStr]
verbPosId = posStringToPosInt(nlp, "VERB")
prepositionPosId = posStringToPosInt(nlp, "ADP")
punctPosId = posStringToPosInt(nlp, "PUNCT")

verb_dict = {}
prep_dict = {}
for lex in nlp.vocab:
	if lex.is_alpha and lex.has_vector:  # optional: filter real words with embeddings
		if lex.pos == nlp.vocab.strings["VERB"]:
			verb_dict[lex.text] = len(verb_dict)
		elif lex.pos == nlp.vocab.strings["ADP"]:
			prep_dict[lex.text] = len(prep_dict)

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
def make_pairs(kp: List[int], mode: keypointModes, r: int, q: int) -> Tuple[torch.Tensor, torch.Tensor]:
	"""
	Parameters
	----------
	kp   : sorted list of key-point token indices
	mode : "allKeypointCombinations" | "firstKeypointConsecutivePairs" | "firstKeypointPairs"
	r/q  : user parameters (see requirement 3)

	Return
	------
	pairs	: (N, 2)  long tensor of [i, j] with i<j; invalid -> 0,0
	valid_ms : (N,)	bool tensor, True where pair is valid
	"""
	if len(kp) < 2:
		# zero-fill for the expected number of rows so caller can reshape
		if mode == "firstKeypointConsecutivePairs":
			N = r
		elif mode == "firstKeypointPairs":
			N = r * (q - 1)
		elif mode == "allKeypointCombinations":		 # we let it be empty
			N = 0
		return torch.zeros(N, 2, dtype=torch.long), torch.zeros(N, dtype=torch.bool)

	# ------------  a) every ordered permutation -------------------- #
	if mode == "allKeypointCombinations":
		out = [(i, j) for idx_i, i in enumerate(kp) for j in kp[idx_i + 1:]]
		pairs = torch.as_tensor(out, dtype=torch.long)
		pairs = torch.sort(pairs, dim=1).values	# ensure i < j so downstream code always receives [start, end]
		valid = torch.ones(len(out), dtype=torch.bool)
		return pairs, valid

	# ------------  b) last r adjacent pairs ---------------------- #
	if mode == "firstKeypointConsecutivePairs":
		adj = list(zip(kp[:-1], kp[1:]))			   # consecutive pairs
		adj = adj[:r]								 # keep the *first* r
		valid_n = len(adj)
		# zero-padding to fixed length r
		pairs = torch.zeros(r, 2, dtype=torch.long)
		if valid_n:
			pairs[:valid_n] = torch.as_tensor(adj, dtype=torch.long)
			pairs[:valid_n] = torch.sort(pairs[:valid_n], dim=1).values	# ensure i < j so downstream code always receives [start, end]
		valid = torch.zeros(r, dtype=torch.bool)
		valid[:valid_n] = True
		return pairs, valid

	# ------------  c) last r starts, spans 2 -> q ------------------ #
	if mode == "firstKeypointPairs":
		out = []
		starts = kp[:r]								# first r starts
		for s_idx, s in enumerate(starts):
			# pick up to q-1 subsequent key-points
			next_kps = kp[s_idx + 1 : s_idx + 1 + (q - 1)]
			if not next_kps:							# none available
				out.extend([(0, 0)] * (q - 1))
			else:
				# pad to q-1 so output length is constant
				for t in range(q - 1):
					if t < len(next_kps):
						out.append((s, next_kps[t]))
					else:
						out.append((0, 0))
		pairs = torch.as_tensor(out, dtype=torch.long)		  # (r*(q-1), 2)
		pairs = torch.sort(pairs, dim=1).values	# ensure i < j so downstream code always receives [start, end]
		valid = ~(pairs[:, 0] == pairs[:, 1])
		return pairs, valid

	raise ValueError(f"Unknown mode {mode}")



