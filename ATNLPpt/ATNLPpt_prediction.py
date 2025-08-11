"""ATNLPpt_prediction.py

# Author:
Richard Bruce Baxter - Copyright (c) 2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ANNpt_main.py

# Usage:
see ANNpt_main.py

# Description:
ATNLPpt prediction

"""

from typing import Optional, Tuple, Dict, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
from ANNpt_globalDefs import *

"""
Dense Snapshot Next-Token Model (B1-first layout, no-padding version)
====================================================================
This revision simplifies the previous implementation by **assuming every
normalised snapshot already occupies the full length L2**.  Consequently we
no longer need per-snapshot length tensors or padding masks.

Input tensor shape (unchanged)
------------------------------
	x : (B1, Q, R, L2, C)

where
* **B1** - normalised batch size,
* **Q**  - number of resolution snapshots per section,
* **R**  - number of context sections,
* **L2** - fixed normalised snapshot length.
* **C**  - vocabulary size (e.g. 30522),

The model outputs one next-token prediction per element of the **B1** axis.

Revision history
----------------
* **v0.8**(2025-08-01) -- add option ATNLPuseSequenceLevelPrediction: use input shape (B1*Q, R, L2*d)
* **v0.7**(2025-07-23) -- *Layout change:* input is now `(B1,Q,R,L2,C), change TransformerBackbone to use input of shape (B1*Q, R*L2, d), add WaveNetBackbone
* **v0.6**(2025-07-22) -- *Layout change:* input is now `(B1,R,Q,L2,C)
* **v0.5**(2025-07-18) -- assume no padding; removed `lens`-based masking and simplified the forward pass accordingly.
* **v0.4**(2025-07-18) -- bug-fix for CNN length preservation.
* **v0.3**(2025-07-18) -- *Layout change:* input is now `(B1,R,Q,C,L2)` and lens is `(B1,R,Q)`.  No other behaviour changed.
* **v0.2**(2025-07-18) -- add **top-1 accuracy** computation.
* **v0.1**(2025-07-18) -- initial release.
"""

# ---------------------------------------------------------------------------
# 1.  Encoder/Decoder - d_model projection (dense)
# ---------------------------------------------------------------------------

class DenseSnapshotEncoder(nn.Module):
	"""Linear projection pTE from R^d_input - R^d_model."""

	def __init__(self, d_input: int, d_model: int):
		super().__init__()
		self.E = nn.Linear(d_input, d_model, bias=False)

	def forward(self, S: torch.Tensor) -> torch.Tensor:
		"""(B1R,QL2,C) ->  (B1Q,RL2,d)."""
		B1Q, RL2, C = S.shape
		S = S.reshape(-1, C)	  # (B1QRL2, C)
		out = self.E(S)
		d_model = out.shape[-1]
		return out.view(B1Q, RL2, d_model)

class DenseSnapshotDecoder(nn.Module):
	"""Linear projection pTD from R^d_model - R^C."""

	def __init__(self, C: int, d_model: int):
		super().__init__()
		self.D = nn.Linear(d_model, C, bias=False)
		
	def forward(self, S: torch.Tensor) -> torch.Tensor:
		""" (B1*Q, R*L2, d) ->  (B1*Q, R*L2, C)."""
		B1Q, RL2, d = S.shape
		S = S.reshape(-1, d)	  # (B1QRL2, d)
		out = self.D(S)
		C = out.shape[-1]
		return out.view(B1Q, RL2, C)

# ---------------------------------------------------------------------------
# 2.  Causal Backbones
# ---------------------------------------------------------------------------

class TransformerBackbone(nn.Module):
	"""
	Stack of `nn.TransformerEncoder` layers (batch-first).
	Includes learnable positional embeddings along the R axis.
	"""

	def __init__(self, d_in: int, nhead: int = 8, nlayers: int = 6, max_R: int = contextSizeMax, causal: bool = True):
		super().__init__()
		self.pos_emb = nn.Embedding(max_R, d_in)
		enc_layer = nn.TransformerEncoderLayer(d_model=d_in, nhead=nhead, dim_feedforward=4 * d_in, batch_first=True)
		self.enc = nn.TransformerEncoder(enc_layer, nlayers)
		self.causal = causal

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# x: (B1*Q, R*L2, d)
		RL2 = x.size(1)
		pos_ids = torch.arange(RL2, device=x.device).unsqueeze(0)	# (1, RL2)
		x = x + self.pos_emb(pos_ids)								# broadcast
		if self.causal:
			mask = torch.triu(torch.ones(RL2, RL2, device=x.device, dtype=torch.bool), 1)
			out = self.enc(x, mask=mask)							# (B1*Q, R*L2, d)
		else:
			out = self.enc(x)
		return out

class WaveNetBackbone(nn.Module):
	"""
	Causal dilated 1-D conv stack  la WaveNet, along the R axis.
	Input/Output: (B, R, D) where D == d_in (typically L2*d from your upstream reshape).
	API mirrors TransformerBackbone.
	"""

	def __init__(
		self,
		d_in: int,
		kernel_size: int = 3,
		n_blocks: int = 2,
		layers_per_block: int = 6,
		skip_mult: int = 2
	):
		super().__init__()
		assert kernel_size % 2 == 1, "Use an odd kernel_size for clean causal padding."
		self.d_in = d_in

		blocks = []
		for _ in range(n_blocks):
			for l in range(layers_per_block):
				dil = 2 ** l
				blocks.append(_WaveNetResBlock(d_in, kernel_size, dil, skip_mult))
		self.blocks = nn.ModuleList(blocks)

		# post-processing of accumulated skip connections
		self.post = nn.Sequential(
			nn.ReLU(),
			nn.Conv1d(skip_mult * d_in, d_in, kernel_size=1),
			nn.ReLU(),
			nn.Conv1d(d_in, d_in, kernel_size=1),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# x: (B, R, D)
		B, R, D = x.shape
		assert D == self.d_in

		h = x.transpose(1, 2)			# (B, D, R)
		skip_accum = []
		for blk in self.blocks:
			h, skip = blk(h)			# both (B, D, R) or (B, skip_mult*D, R)
			skip_accum.append(skip)

		skips = torch.stack(skip_accum, dim=0).sum(dim=0)	# (B, skip_mult*D, R)
		out = self.post(skips)					# (B, D, R)
		return out.transpose(1, 2)				# (B, R, D)


class _WaveNetResBlock(nn.Module):
	"""
	Single gated residual block.
	"""
	def __init__(self, channels: int, kernel_size: int, dilation: int, skip_mult: int):
		super().__init__()
		self.dil = dilation
		self.ks = kernel_size

		# filter & gate convolutions (causal via manual left pad)
		self.conv_f = nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=0)
		self.conv_g = nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=0)

		# 1x1 projections for residual and skip paths
		self.res_proj = nn.Conv1d(channels, channels, kernel_size=1)
		self.skip_proj = nn.Conv1d(channels, skip_mult * channels, kernel_size=1)

	def forward(self, x: torch.Tensor):
		# x: (B, C, R)
		pad = (self.ks - 1) * self.dil
		x_padded = F.pad(x, (pad, 0))		# left-pad only -> causal

		f = torch.tanh(self.conv_f(x_padded))
		g = torch.sigmoid(self.conv_g(x_padded))
		z = f * g							# (B, C, R)

		skip = self.skip_proj(z)			# (B, skip_mult*C, R)
		res = self.res_proj(z) + x			# residual connection
		return res, skip

# ---------------------------------------------------------------------------
# 3.  Top-level model
# ---------------------------------------------------------------------------

class DenseSnapshotModel(nn.Module):
	def __init__(
		self,
		d_input: int,
		d_model: int = 512,
		backbone: str = "transformer",
		backbone_kwargs: Optional[Dict] = None,
		pretrained_embed: Optional[torch.Tensor] = None,
	):
		super().__init__()
		self.d_model = d_model
		self.encoder = DenseSnapshotEncoder(d_input, d_model)
		self.backbone_type = backbone
		if backbone_kwargs is None:
			backbone_kwargs = {}
		if backbone == "transformer":
			self.backbone = TransformerBackbone(d_model, **backbone_kwargs)
			self.decoder = DenseSnapshotDecoder(d_input, d_model)
		elif backbone == "wavenet":
			self.backbone = WaveNetBackbone(d_model, **backbone_kwargs)
			self.decoder = DenseSnapshotDecoder(d_input, d_model)
		else:
			printe("Unknown backbone {backbone}")
		
	def forward(self, x: torch.Tensor, trainOrTest: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
		"""Return `(logits, fused_rep)`.

		* **x** - (B1*Q,R*L2,C)
		*	**logits** - (B1*Q, R*L2-1, C) one prediction for each token
		"""
		x = self.encoder(x)				# (B1*Q,R*L2,d)
		enc = self.backbone(x)	#transformer/wavenet: (B1*Q, R*L2, d)
		logits = self.decoder(x)	#transformer/wavenet: (B1*Q, R*L2, C)

		return logits

# ---------------------------------------------------------------------------
# 5. evaluation utilities
# ---------------------------------------------------------------------------

def loss_function(logits: torch.Tensor, targets: torch.Tensor):
	#print("logits.shape = ", logits.shape)
	#print("targets.shape = ", targets.shape)
	loss = F.cross_entropy(logits, targets)
	return loss
	
def calculate_matches(logits: torch.Tensor, targets: torch.Tensor) -> float:
	"""Compute bool top-1 accuracy (1/0) for each sample in mini-batch."""
	if(useSlidingWindow):
		preds = logits.argmax(dim=-1)	#compare a single target
	else:
		#preds = logits	#compare a distribution of targets across C (normalised snapshot tokens contain a distribution of bert tokens, not a single bert token)
		preds = logits.argmax(dim=-1)	#compare a single target
		targets = targets.argmax(dim=-1)	#compare a single target
	matches = (preds == targets)
	return matches, targets
