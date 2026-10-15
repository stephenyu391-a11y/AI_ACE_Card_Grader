"""The grader: one pre-trained backbone reads all 18 views, a small head gives the five grades.

Backbone  turns each view into one feature vector (shared weights for all views).
Head      lets each of the five grades look at the views it cares about
          (attention over the 18 vectors), then scores grades 1..10.
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

from scoring import FIELDS
from views import N_VIEWS

MODELS = Path(__file__).resolve().parents[1] / "models"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
GROUPS = (slice(0, 2), slice(2, 10), slice(10, 18))  # full cards, corners, edges: three picture sizes


def views_to_tensors(views: list[Image.Image]) -> tuple[torch.Tensor, ...]:
    """One tensor per picture size: (2,3,H,W) full, (8,3,h,w) corners, (8,3,h,w) edges."""
    def one(image):
        return (torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1) - MEAN) / STD
    return tuple(torch.stack([one(v) for v in views[g]]) for g in GROUPS)


class Backbone(nn.Module):
    def __init__(self, name: str):
        super().__init__()
        self.name = name
        if name.startswith("dinov2"):
            from transformers import AutoModel

            self.net = AutoModel.from_pretrained(MODELS / name)
            self.dim = self.net.config.hidden_size
        elif name == "convnext-base-in22k":
            import timm
            from safetensors.torch import load_file

            self.net = timm.create_model("convnext_base", pretrained=False, num_classes=0)
            weights = load_file(str(MODELS / name / "model.safetensors"))
            # The checkpoint still carries its 21k-class output layer; only the body is wanted.
            self.net.load_state_dict({k: v for k, v in weights.items() if not k.startswith("head.fc.")})
            self.dim = self.net.num_features
        else:
            raise ValueError(f"unknown backbone {name!r}")

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        if self.name.startswith("dinov2"):
            return self.net(pixel_values=x).pooler_output
        return self.net(x)

    def forward(self, full: torch.Tensor, corners: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        """Batches (B,2,..), (B,8,..), (B,8,..) -> features (B, 18, dim)."""
        batch = full.shape[0]
        parts = [self._features(x.flatten(0, 1)).view(batch, x.shape[1], -1) for x in (full, corners, edges)]
        return torch.cat(parts, dim=1)


# Which views each grade may look at when own_views is on. Corners and edges are judged from
# their close-ups only, so the whole-card views (which show plainly how old a card is) cannot
# stand in for looking at the damage. Surface and the final grade see everything.
OWN_VIEWS = {"surface": slice(0, N_VIEWS), "edges": GROUPS[2], "corners": GROUPS[1], "centering": GROUPS[0],
             "grade": slice(0, N_VIEWS)}


class Head(nn.Module):
    def __init__(self, dim: int, width: int = 256, dropout: float = 0.2, own_views: bool = False):
        super().__init__()
        hidden = torch.ones(len(FIELDS), N_VIEWS, dtype=torch.bool)       # True = this grade may not look at this view
        for i, field in enumerate(FIELDS):
            hidden[i, OWN_VIEWS[field] if own_views else slice(0, N_VIEWS)] = False
        self.register_buffer("hidden", hidden, persistent=False)
        self.project = nn.Sequential(nn.LayerNorm(dim), nn.Dropout(dropout), nn.Linear(dim, width))
        self.view_kind = nn.Parameter(torch.zeros(N_VIEWS, width))   # tells the head which view is which
        self.queries = nn.Parameter(torch.randn(len(FIELDS), width) * 0.02)
        self.attend = nn.MultiheadAttention(width, num_heads=4, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(width)
        self.out = nn.Linear(width, 10)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Features (B, 18, dim) -> scores (B, 5 fields, 10 grades)."""
        tokens = self.project(features) + self.view_kind
        queries = self.queries.expand(tokens.shape[0], -1, -1)
        mixed, _ = self.attend(queries, tokens, tokens, attn_mask=self.hidden)
        return self.out(self.norm(mixed + queries))


class Grader(nn.Module):
    def __init__(self, backbone: str, own_views: bool = False):
        super().__init__()
        self.backbone = Backbone(backbone)
        self.head = Head(self.backbone.dim, own_views=own_views)

    def forward(self, full, corners, edges):
        return self.head(self.backbone(full, corners, edges))
