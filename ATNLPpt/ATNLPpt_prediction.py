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

from typing import Optional, Tuple, Dict
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
	S : (B1, R, Q, C, L2)

where
* **B1** - normalised batch size,
* **R**  - number of context sections,
* **Q**  - number of resolution snapshots per section,
* **C**  - vocabulary size (e.g.30 522),
* **L2** - fixed normalised snapshot length.

The model outputs one next-token prediction per element of the **B1** axis.

Revision history
----------------
* **v0.5**(2025-07-18)-- assume no padding; removed `lens`-based masking and
  simplified the forward pass accordingly.
* **v0.4**(2025-07-18) -- bug-fix for CNN length preservation.
* **v0.3**(2025-07-18) -- *Layout change:* input is now `(B1,R,Q,C,L2)` and lens is `(B1,R,Q)`.  No other behaviour changed.
* **v0.2**(2025-07-18) -- add **top-1 accuracy** computation.
* **v0.1**(2025-07-18) -- initial release.
"""

# ---------------------------------------------------------------------------
# 1.  Snapshot - d_model projection (dense)
# ---------------------------------------------------------------------------

class DenseSnapshotEncoder(nn.Module):
	"""Linear projection  pTE  from R^C - R^d_model."""

	def __init__(self, C: int, d_model: int, pretrained_embed: Optional[torch.Tensor] = None):
		super().__init__()
		self.E = nn.Linear(C, d_model, bias=False)
		if pretrained_embed is not None:
			assert pretrained_embed.shape == (C, d_model)
			with torch.no_grad():
				self.E.weight.copy_(pretrained_embed)

	def forward(self, S: torch.Tensor) -> torch.Tensor:
		"""(B1,R,Q,C,L2)  (B1,R,Q,L2,d)."""
		B1, R, Q, C, L2 = S.shape
		S = S.permute(0, 1, 2, 4, 3).reshape(-1, C)	  # (B2RQL2, C)
		out = self.E(S)
		d_model = out.shape[-1]
		return out.view(B1, R, Q, L2, d_model)

# ---------------------------------------------------------------------------
# 2.  Backbones (mask removed)
# ---------------------------------------------------------------------------

class CausalCNNBackbone(nn.Module):
	"""Depth-wise causal 1-D CNN stack that preserves sequence length."""

	def __init__(self, d_model: int, kernel_size: int = 5, n_layers: int = 6):
		super().__init__()
		self.kernel_size = kernel_size
		layers = []
		for _ in range(n_layers):
			pad = kernel_size - 1  # causal (pad left)
			layers.append(nn.Conv1d(d_model, d_model, kernel_size, padding=pad, groups=d_model))
			layers.append(nn.ReLU())
		self.net = nn.Sequential(*layers)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		B1, R, Q, L, d = x.shape
		x = x.view(B1 * R * Q, L, d).transpose(1, 2)  # (B', d, L)
		x = self.net(x)
		x = x[..., -L:]							   # retain last L steps
		return x.transpose(1, 2).view(B1, R, Q, L, d)

class TransformerBackbone(nn.Module):
	"""Stack of `nn.TransformerEncoder` layers (batch-first)."""

	def __init__(self, d_model: int, nhead: int = 8, nlayers: int = 4):
		super().__init__()
		self.enc = nn.TransformerEncoder(
			nn.TransformerEncoderLayer(d_model, nhead, 4 * d_model, batch_first=True),
			nlayers,
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		B1, R, Q, L, d = x.shape
		B_ = B1 * R * Q
		out = self.enc(x.view(B_, L, d))
		return out.view(B1, R, Q, L, d)

# ---------------------------------------------------------------------------
# 3.  Top-level model (no `lens` or mask arguments)
# ---------------------------------------------------------------------------

class DenseSnapshotModel(nn.Module):
	def __init__(
		self,
		C: int,
		d_model: int = 512,
		backbone: str = "cnn",
		backbone_kwargs: Optional[Dict] = None,
		pretrained_embed: Optional[torch.Tensor] = None,
	):
		super().__init__()
		self.encoder = DenseSnapshotEncoder(C, d_model, pretrained_embed)
		if backbone_kwargs is None:
			backbone_kwargs = {}
		if backbone == "cnn":
			self.backbone = CausalCNNBackbone(d_model, **backbone_kwargs)
		elif backbone == "transformer":
			self.backbone = TransformerBackbone(d_model, **backbone_kwargs)
		else:
			raise ValueError(f"Unknown backbone {backbone}")
		self.proj = nn.Linear(d_model, C, bias=False)

	def forward(self, S: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		"""Return `(logits, fused_rep)`.

		* **S** - (B1,R,Q,C,L2)
		* **logits** - (B1,C)  one prediction per B1 sample.
		"""
		B1, R, Q, C, L2 = S.shape
		x = self.encoder(S)				# (B1,R,Q,L2,d)
		enc = self.backbone(x)			 # (B1,R,Q,L2,d)

		# Last real timestep is always indexL2-1
		last = enc[..., -1, :]			 # (B1,R,Q,d)
		fused = last.mean(dim=2).mean(dim=1)  # (B1,d)
		logits = self.proj(fused)		  # (B1,C)
		return logits, fused

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
	preds = logits.argmax(dim=-1)
	matches = (preds == targets)
	return matches
