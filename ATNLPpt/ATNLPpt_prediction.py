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
* **v0.11**(2025-08-11) -- add option ATNLPpredictTransformedTokens=False (use nn.TransformerDecoder) 
* **v0.10**(2025-08-11) -- update DenseSnapshotEncoder/DenseSnapshotDecoder/TransformerEncoderBackbone
* **v0.9**(2025-08-11) -- update loss_function/calculate_matches
* **v0.8**(2025-08-01) -- add option ATNLPuseSequenceLevelPredictionInput: use input shape (B1*Q, R, L2*d)
* **v0.7**(2025-07-23) -- *Layout change:* input is now `(B1,Q,R,L2,C), change TransformerEncoderBackbone to use input of shape (B1*Q, R*L2, d), add WaveNetBackbone
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
	"""Token DenseSnapshotEncoder -> expected embeddings."""
	def __init__(self, d_input: int, d_model: int, padding_idx: int = 0):
		super().__init__()
		self.emb = nn.Embedding(d_input, d_model, padding_idx=padding_idx)

	def forward(self, tokenProbs: torch.Tensor) -> torch.Tensor:
		"""(B1R,QL2,C) ->  (B1Q,RL2,d)."""		
		# tokenProbs: (B, L, V); rows ideally sum to 1 for non-pad positions
		# mask pad positions upstream (set probs to one-hot of padding_idx or zeros) as needed
		return tokenProbs @ self.emb.weight  # (B, L, d_model)

class DenseSnapshotDecoder(nn.Module):
	"""
	Continuous vectors -> vocab distribution
	Uses tied weights with the embedding layer for efficiency and consistency.
	"""
	def __init__(self, embedding: nn.Embedding, padding_idx: int = 0):
		super().__init__()
		# tie weights: decoder shares the same matrix as encoder embeddings
		self.emb = embedding
		self.padding_idx = padding_idx
		# Optional bias for the vocab projection
		self.bias = nn.Parameter(torch.zeros(embedding.num_embeddings))

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# x: (B, L, d_model)
		# logits: (B, L, V)
		logits = x @ self.emb.weight.T + self.bias

		'''
		# Mask PAD token's logit to -inf so it never gets predicted
		if self.padding_idx is not None:
			logits[..., self.padding_idx] = float("-inf")
		'''
		
		return logits
		
# ---------------------------------------------------------------------------
# 2.  Causal Backbones
# ---------------------------------------------------------------------------

if(ATNLPpredictTransformedTokens):
	class TransformerEncoderBackbone(nn.Module):
		"""
		Stack of `nn.TransformerEncoder` layers (batch-first).
		Includes learnable positional embeddings along the R axis.
		"""

		def __init__(self, d_in: int, max_L: int, nhead: int = 8, nlayers: int = 6, causal: bool = True):
			super().__init__()
			self.pos_emb = nn.Embedding(max_L, d_in)
			d_ff = 4 * d_in
			enc_layer = nn.TransformerEncoderLayer(d_model=d_in, nhead=nhead, dim_feedforward=d_ff, batch_first=True, norm_first=True)
			self.enc = nn.TransformerEncoder(enc_layer, nlayers)
			self.causal = causal
			self._mask_cache = {}	# cache for causal masks keyed by (S, device, dtype)

		def forward(self, x: torch.Tensor, padMask: torch.Tensor) -> torch.Tensor:
			# x: (B1*Q, R*L2, d)
			RL2 = x.size(1)
			assert RL2 <= self.pos_emb.num_embeddings, f"seq len {RL2} > max_L {self.pos_emb.num_embeddings}"
			pos_ids = torch.arange(RL2, device=x.device).unsqueeze(0)	# (1, RL2)
			x = x + self.pos_emb(pos_ids)								# broadcast
			if self.causal:
				attn_mask = self._get_causal_mask(RL2, x.device, x.dtype)
				out = self.enc(x, mask=attn_mask, src_key_padding_mask=padMask, is_causal=True)					# (B1*Q, R*L2, d)
			else:
				out = self.enc(x, src_key_padding_mask=padMask)
			return out

		def _get_causal_mask(self, S: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
			key = (S, device, dtype)
			m = self._mask_cache.get(key)
			if m is None:
				# additive mask: 0 on/below diagonal, -inf above (future positions)
				#m = torch.triu(torch.full((S, S), float('-inf'), device=device, dtype=dtype), diagonal=1)
				m = torch.triu(torch.ones(S, S, dtype=torch.bool, device=device), 1)
				self._mask_cache[key] = m
			return m
else:
	class TransformerDecoderBackbone(nn.Module):
		"""
		Stack of `nn.TransformerDecoder` layers (batch-first).
		"""

		def __init__(self, d_in: int, max_L: int, pad_id: int, nhead: int = 8, nlayers: int = 6, causal: bool = True):
			super().__init__()
			self.pos_emb = nn.Embedding(max_L, d_in)
			d_ff = 4 * d_in
			dec_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_ff, batch_first=True, norm_first=True)
			self.decoder = nn.TransformerDecoder(dec_layer, nlayers)
			self.causal = causal	#implied true via _get_causal_mask
			self._mask_cache = {}	# cache for causal masks keyed by (S, device, dtype)
			self.pad_id = pad_id

		def forward(self, x: torch.Tensor, y : torch.Tensor, xPadMask: torch.Tensor, yPadMask: torch.Tensor) -> torch.Tensor:
			# x: (B1, L1, d_model)
			# y: (B1, L1, d_model)
			B, L, _  = y.shape
			device = x.device
			pos_ids = torch.arange(L, device=device).unsqueeze(0)	# (1, L1)
			posemb = self.pos_emb(pos_ids)
			tgt = y + posemb		# broadcast
			tgt_key_padding_mask = yPadMask    # (B1, L1)
			tgt_mask = self._get_causal_mask(L, device)            # (L, L)
			mem_pad_mask = xPadMask
			out = self.decoder(tgt=tgt, memory=x, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=mem_pad_mask)
			return out

		def _get_causal_mask(self, S: int, device: torch.device) -> torch.Tensor:
			key = (S, device)
			m = self._mask_cache.get(key)
			if m is None:
				m = torch.triu(torch.ones(S, S, dtype=torch.bool, device=device), 1)
				self._mask_cache[key] = m
			return m
		
class WaveNetBackbone(nn.Module):
	"""
	Causal dilated 1-D conv stack  la WaveNet, along the R axis.
	Input/Output: (B, R, D) where D == d_in (typically L2*d from your upstream reshape).
	API mirrors TransformerEncoderBackbone.
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
		d_target: int,
		d_model: int,
		max_L: int,
		backbone: str,
	):
		super().__init__()
		self.d_model = d_model
		self.encoder = DenseSnapshotEncoder(d_input, d_model, NLPpadTokenID)
		if(not ATNLPpredictTransformedTokens):
			self.target_encoder = nn.Embedding(d_target, d_model)

		self.backbone_type = backbone
		if backbone == "transformer":
			if(ATNLPpredictTransformedTokens):
				self.backbone = TransformerEncoderBackbone(d_model, max_L)
			else:
				self.backbone = TransformerDecoderBackbone(d_model, max_L, NLPpadTokenID)
			self.decoder = DenseSnapshotDecoder(self.encoder.emb, NLPpadTokenID)
		elif backbone == "wavenet":
			self.backbone = WaveNetBackbone(d_model)
			self.decoder = DenseSnapshotDecoder(self.encoder.emb, NLPpadTokenID)
		else:
			printe("Unknown backbone {backbone}")
		
	if(ATNLPpredictTransformedTokens):
		def forward(self, x: torch.Tensor, padMask: torch.Tensor, trainOrTest: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
			"""Return `(logits, fused_rep)`.

			* **x** - (B1*Q,R*L2,C)
			*	**logits** - (B1*Q, R*L2-1, C) one prediction for each token
			"""
			x = self.encoder(x)				# (B1*Q,R*L2,d)
			enc = self.backbone(x, padMask)	#transformer/wavenet: (B1*Q, R*L2, d)
			logits = self.decoder(enc)	#transformer/wavenet: (B1*Q, R*L2, C)
			#print("logits = ", logits)

			return logits
	else:
		def forward(self, x: torch.Tensor, y: torch.Tensor, xPadMask: torch.Tensor, yPadMask: torch.Tensor, trainOrTest: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
			"""
			* **x** - (B1,R,L2*C)	#assume Q=1
			* **y** - (B1,L1)
			"""
			x = self.encoder(x)				# (B1,R,d)
			y = self.target_encoder(y)				# (B1,L1,d)
			enc = self.backbone(x, y, xPadMask, yPadMask)		#transformer: (B1, R*L2, d)
			logits = self.decoder(enc)	#transformer/wavenet: (B1, L1, C)
			#print("logits = ", logits)

			return logits

# ---------------------------------------------------------------------------
# 5. evaluation utilities
# ---------------------------------------------------------------------------

def loss_function(logits: torch.Tensor, targets: torch.Tensor, padMask: torch.Tensor, onehotTargets: bool):
	#print("logits.shape = ", logits.shape)
	#print("targets.shape = ", targets.shape)
	if(onehotTargets):
		loss = softCrossEntropyUnnormalized(logits, targets, padMask)	#CHECKTHIS: assume token distribution at l in [B, L, C] are unnormalised (across C)
	else:
		"""Cross-entropy loss over flattened B1*L dimension with PAD masked out."""
		# targets: (B, L, C) one-hot; convert to indices
		targets_idx = targets			  # (B, L)
		logits = logits.view(-1, logits.size(-1))			 # (B*L, C)
		targets_idx = targets_idx.view(-1)					# (B*L)
		loss = F.cross_entropy(logits, targets_idx, ignore_index=NLPpadTokenID)
	return loss
	
def calculate_matches(logits: torch.Tensor, targets: torch.Tensor, onehotTargets: bool) -> float:
	"""Compute bool top-1 accuracy (1/0) for each sample in mini-batch."""
	if(onehotTargets):
		targets = targets.argmax(dim=-1)	#compare a single target	#calculate top-1 accuracy
	#preds = logits	#compare a distribution of targets across C (normalised snapshot tokens contain a distribution of bert tokens, not a single bert token)
	preds = logits.argmax(dim=-1)	#compare a single target
	valid_mask = (targets != NLPpadTokenID)	#redundant (performed by calculateAccuracy)
	matches = (preds == targets) & valid_mask
	return matches

def softCrossEntropyNormalized(
	logits: torch.Tensor,			# (B, L, C)
	targets: torch.Tensor,			# (B, L, C), rows sum to ~1
	padMask: torch.Tensor,
	eps: float = 1e-12
) -> torch.Tensor:
	"""
	Soft-label CE for mutually exclusive classes.
	Masking rule: position is masked iff argmax(targets[b,l]) == NLPpadTokenID.
	"""
	logp = F.log_softmax(logits, dim=-1)						# (B, L, C)
	tokenLoss = -(targets * logp).sum(dim=-1)					# (B, L)

	weights = (~padMask).to(dtype=logits.dtype)

	return (tokenLoss * weights).sum() / weights.sum().clamp_min(eps)

def softCrossEntropyUnnormalized(
	logits: torch.Tensor,			# (B, L, C)
	targets: torch.Tensor,			# (B, L, C), non-neg; rows may not sum to 1
	padMask: torch.Tensor,
	preserveMass: bool = False,
	eps: float = 1e-12
) -> torch.Tensor:
	"""
	Soft-label CE for mutually exclusive classes with unnormalized targets.
	We renormalize per (B,L) across C. Masking uses only argmax==NLPpadTokenID.
	"""
	p = targets.clamp_min(0.0)
	mass = p.sum(dim=-1, keepdim=True)							# (B, L, 1)
	pNorm = p / mass.clamp_min(eps)								# (B, L, C)

	logp = F.log_softmax(logits, dim=-1)
	tokenLoss = -(pNorm * logp).sum(dim=-1)						# (B, L)

	baseWeights = mass.squeeze(-1).to(dtype=logits.dtype) if preserveMass \
		else torch.ones_like(tokenLoss, dtype=logits.dtype)

	weights = baseWeights * (~padMask).to(dtype=logits.dtype)

	return (tokenLoss * weights).sum() / weights.sum().clamp_min(eps)

# Case A: you have hard token IDs (B, L) with a PAD id
def derivePadMaskFromIds(tokenIds: torch.Tensor, padId: int) -> torch.Tensor:
	# returns (B, L) bool, True where PAD
	return (tokenIds == padId)

# Case B: you have soft distributions (B, L, V)
# We mark PAD if either (a) the PAD column has prob ~1, or (b) the row is all ~0
def derivePadMaskFromProbs(tokenProbs: torch.Tensor, padId: int, eps: float = 1e-6) -> torch.Tensor:
	padCol = tokenProbs[..., padId]								# (B, L)
	rowSum = tokenProbs.sum(dim=-1)								# (B, L)
	isPadByCol = (padCol >= 1.0 - eps)
	isPadByZero = (rowSum <= eps)
	return (isPadByCol | isPadByZero)
	
