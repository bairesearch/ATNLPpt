"""ATNLPpt_transformation.py

# Author:
Richard Bruce Baxter - Copyright (c) 2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ANNpt_main.py

# Usage:
see ANNpt_main.py

# Description:
ATNLPpt transformation

# Description
# -----------
# PyTorch re-write of ATNLP normalisation with upgrades 1-6:
#   (4) I/O tensor shapes
#		 in : (B1, C, L1)
#		 out: (B2, C, L2)  where B2 = B1*R	  (train)
#							  or  B1*R*(Q-1)	(eval)
#   (5) No Python loops in the crop/resize - a single grid_sample call
#		 keypoint_idx is supplied as int tensor (B2, 2)
#   (6) Uses only PyTorch core - grid_sample provides the 'batch crop + resize'
# 

"""

from __future__ import annotations
from ANNpt_globalDefs import *
import torch
import torch.nn.functional as F
import spacy
import ATNLPpt_keypoints
from typing import List, Dict, Tuple, Literal
NL_MODEL = spacy.load("en_core_web_sm", disable=("ner", "parser", "lemmatizer"))

# ------------------------------------------------------------------------- #
# (4)(5)(6)  batch crop + resize with ONE grid_sample call (no loops)	   #
# ------------------------------------------------------------------------- #
def transform_batch(
	seq_tensor: torch.Tensor,		  # (B1, C, L1)
	R : int,
	Q : int,
	L2 : int,
	foundKeypointPairs : bool,
	keypointPairsIndices : torch.Tensor, 	 # (B2, 2)
	keypointPairsCharIdx : torch.Tensor, 	 # (B2, 2)
	keypointPairsValid : torch.Tensor, 	 # (B2,)
	src_ids : torch.Tensor	# (B2)
) -> Tuple[torch.Tensor, List[List[Dict]]]:
	"""
	Main API - returns (B2, C, L2) where B2 depends on mode, plus key-point kp_meta.
	"""				
				
	device = seq_tensor.device
	B1, C, L1 = seq_tensor.shape
	
	if foundKeypointPairs:								  # dev mode w/o keypointPairsCharIdx
		B2 = keypointPairsCharIdx.size(0)

		if(debugATNLPnormalisation):
			print("transform_batch():")
			print("seq_tensor.shape = ", seq_tensor.shape)
			print("R = ", R)
			print("Q = ", Q)
			print("B1 = ", B1)
			print("C = ", C)
			print("L1 = ", L1)
			print("L2 = ", L2)

		# ------------------  area-style crop + resize (no loops) --------------- #
		# 1. gather the relevant rows from the source batch
		src = seq_tensor[src_ids]							# (B2,C,L1)

		# 2. build a boolean mask for every crop span [start, end]
		pos	= torch.arange(L1, device=device).view(1,1,L1)		# (1,1,L1)
		start = keypointPairsCharIdx[:, 0].view(B2,1,1)						# (B2,1,1)
		end	= keypointPairsCharIdx[:, 1].view(B2,1,1)						# (B2,1,1)
		mask = (pos >= start) & (pos <= end)					# (B2,1,L1)
		mask = mask.expand(-1, C, -1)						# (B2,C,L1)

		# 3. zero-out everything outside the span
		seg = src.masked_fill(~mask, 0.0)					# (B2,C,L1)

		# 4. area-style down-sampling: max over L2 equal-width bins
		normalisedSnapshots = F.adaptive_avg_pool1d(seg, L2)				# (B2,C,L2)

		# 5. zero-out the snapshots that were marked invalid
		normalisedSnapshots[~keypointPairsValid] = 0

		S = B2 // B1
		normalisedSnapshots = normalisedSnapshots.reshape(B1, S, C, L2)	# (B1,S,C,L2)

		if(debugATNLPnormalisation):
			print("B2 = ", B2)
			print("normalisedSnapshots shape :", tuple(normalisedSnapshots.shape))

		if(ATNLPusePredictionHead):
			keypointPairsIndices, keypointPairsCharIdx, keypointPairsValid = (None, None, None)
		else:
			keypointPairsIndices = keypointPairsIndices.reshape(B1, S, 2)	#(B1, S, 2)
			keypointPairsCharIdx = keypointPairsCharIdx.reshape(B1, S, 2)	#(B1, S, 2)
			keypointPairsValid = keypointPairsValid.reshape(B1, S)	#(B1,S)
	else:
		normalisedSnapshots = torch.empty(0, C, L2)

	return normalisedSnapshots, keypointPairsIndices, keypointPairsCharIdx, keypointPairsValid


def generateSnapshotLengths():
	#use character token indices, not bert token indices or spacy indices]
	lens = keypointPairsCharIdx[:, :, 1] - keypointPairsCharIdx[:, :, 0]	#keypointPairsCharIdx is of shape (B1, S, 2)
	lens = lens.reshape(B1, R, Q)	#(B1, R, Q)
