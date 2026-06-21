"""LeJEPA ViT-S/16 wrapper using Hugging Face transformers.

Checkpoint: OK-AI/lejepa-vits16-pretrain-in1k

LeJEPA is a self-supervised model. It does not have a classification head.
We treat it as a feature extractor and attach a fresh ClassifierHead.

Output handling:
    The official checkpoint returns a dictionary whose "latent" entry is
    the global image embedding. The wrapper uses it for classification and
    retains fallbacks for standard Hugging Face vision-model outputs.
"""

import importlib
import sys

import torch
import torch.nn as nn
from huggingface_hub import snapshot_download

from ..globals import NUM_CLASSES, LEJEPA_VIT_MODEL, LEJEPA_VIT_REVISION
from .classifier import ClassifierHead


class LeJEPAViT(nn.Module):
    """LeJEPA ViT-S/16 from Hugging Face, with a linear classification head.

    Architecture:
        backbone  – HF AutoModel (pretrained, self-supervised)
        head      – ClassifierHead mapping embed_dim → num_classes
    """

    def __init__(
        self,
        model_name: str = LEJEPA_VIT_MODEL,
        model_revision: str = LEJEPA_VIT_REVISION,
        num_classes: int = NUM_CLASSES,
        freeze_backbone: bool = False,
        trainable_last_blocks: int | None = None,
        head_dropout: float = 0.0,
    ):
        super().__init__()
        self.freeze_backbone = freeze_backbone
        self.trainable_last_blocks = trainable_last_blocks

        self.config, self.backbone = self._load_pretrained_backbone(
            model_name,
            model_revision,
        )

        # Determine embedding dimension
        embed_dim = self._get_embed_dim()

        # Fresh classification head
        self.head = ClassifierHead(embed_dim, num_classes, dropout=head_dropout)

        if freeze_backbone:
            self._freeze_backbone()
        elif trainable_last_blocks is not None:
            self._unfreeze_last_blocks(trainable_last_blocks)

    @staticmethod
    def _load_pretrained_backbone(model_name: str, revision: str):
        """Download and load the checkpoint's bundled ViT-v2 implementation."""
        snapshot_path = snapshot_download(
            repo_id=model_name,
            revision=revision,
            allow_patterns=[
                "config.json",
                "configuration_vitv2.py",
                "modelling_vitv2.py",
                "model.safetensors",
                "hf_src/**",
            ],
        )

        # The upstream files use absolute imports such as
        # ``from configuration_vitv2 import ...`` and ``from hf_src ...``.
        if snapshot_path not in sys.path:
            sys.path.insert(0, snapshot_path)
        importlib.invalidate_caches()

        config_module = importlib.import_module("configuration_vitv2")
        model_module = importlib.import_module("modelling_vitv2")
        config_class = config_module.ViTv2Config
        model_class = model_module.ViTv2PretrainedModel

        config = config_class.from_pretrained(snapshot_path)
        backbone = model_class.from_pretrained(snapshot_path, config=config)
        return config, backbone

    def _get_embed_dim(self) -> int:
        """Infer the embedding dimension from the model config."""
        # Try common config attribute names
        for attr in ("hidden_size", "embed_dim", "d_model"):
            if hasattr(self.config, attr):
                return getattr(self.config, attr)
        # Fallback: ViT-S default
        return 384

    def _freeze_backbone(self) -> None:
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

    def _unfreeze_last_blocks(self, num_blocks: int) -> None:
        """Train only the final transformer blocks and output normalization."""
        blocks = self.backbone.backbone.blocks
        if not 1 <= num_blocks <= len(blocks):
            raise ValueError(
                f"trainable_last_blocks must be between 1 and {len(blocks)}, "
                f"received {num_blocks}."
            )

        for param in self.backbone.parameters():
            param.requires_grad = False
        for block in blocks[-num_blocks:]:
            for param in block.parameters():
                param.requires_grad = True
        for param in self.backbone.backbone.norm.parameters():
            param.requires_grad = True

    def train(self, mode: bool = True):
        """Set training mode while keeping frozen backbone sections in eval."""
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        elif self.trainable_last_blocks is not None:
            self.backbone.eval()
            for block in self.backbone.backbone.blocks[-self.trainable_last_blocks:]:
                block.train(mode)
            self.backbone.backbone.norm.train(mode)
        return self

    @staticmethod
    def _extract_features(outputs) -> torch.Tensor:
        """Extract the global image embedding from a backbone output."""
        if isinstance(outputs, dict) and outputs.get("latent") is not None:
            features = outputs["latent"]
        elif hasattr(outputs, "latent") and outputs.latent is not None:
            features = outputs.latent
        elif (
            hasattr(outputs, "pooler_output")
            and outputs.pooler_output is not None
        ):
            features = outputs.pooler_output
        elif (
            hasattr(outputs, "last_hidden_state")
            and outputs.last_hidden_state is not None
        ):
            features = outputs.last_hidden_state[:, 0]
        elif isinstance(outputs, torch.Tensor):
            features = outputs[:, 0] if outputs.dim() == 3 else outputs
        else:
            if hasattr(outputs, "keys"):
                available = list(outputs.keys())
            else:
                available = [
                    name for name in ("latent", "pooler_output", "last_hidden_state")
                    if hasattr(outputs, name)
                ]
            raise ValueError(
                f"Cannot extract LeJEPA image features from {type(outputs)}. "
                f"Available output fields: {available}"
            )

        if features.dim() == 3 and features.size(1) == 1:
            features = features[:, 0]
        if features.dim() != 2:
            raise ValueError(
                "LeJEPA global features must have shape [batch, embedding_dim], "
                f"but received {tuple(features.shape)}."
            )
        return features

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return LeJEPA's global image embedding before classification."""
        outputs = self.backbone(x)
        return self._extract_features(outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert LeJEPA image embeddings into class logits."""
        return self.head(self.forward_features(x))
