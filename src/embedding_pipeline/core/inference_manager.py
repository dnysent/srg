from abc import ABC, abstractmethod
import io
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image
import warnings
from transformers import AutoImageProcessor, AutoModel, logging as transformers_logging

# Suppress specific known noise warnings
warnings.filterwarnings("ignore", message=".*QuickGELU mismatch.*")
# Suppress transformers informative warnings about Fast processors
transformers_logging.set_verbosity_error()

from ..config import OpenClipConfig, DinoV2Config


class BaseInferenceManager(ABC):
    """Abstract base class for inference managers."""

    @abstractmethod
    def get_embedding(self, file_path: str | Path) -> np.ndarray:
        """Compute embedding for an image file."""
        pass

    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension for the current model."""
        pass


class OpenClipInferenceManager(BaseInferenceManager):
    """Manages OpenCLIP model loading and image embedding computation."""

    def __init__(self, config: OpenClipConfig, models_dir: Path):
        self.config = config
        self.models_dir = models_dir
        self.device = self._resolve_device()
        
        # Ensure models directory exists
        self.models_dir.mkdir(exist_ok=True)

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.config.name,
            pretrained=self.config.pretrained,
            device=self.device,
            force_quick_gelu=self.config.quick_gelu,
            cache_dir=str(self.models_dir),
        )
        self.model.eval()

        if self.config.precision == "fp16":
            self.model = self.model.half()

    def _resolve_device(self) -> str:
        if self.config.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.config.device

    def preprocess_image(self, file_bytes: bytes) -> torch.Tensor:
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        return self.preprocess(image).unsqueeze(0).to(self.device)

    def compute_embedding(self, image_tensor: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            if self.config.precision == "fp16":
                image_tensor = image_tensor.half()
            embedding = self.model.encode_image(image_tensor)
            embedding /= embedding.norm(dim=-1, keepdim=True)
        return embedding.cpu().numpy().flatten()

    def get_embedding(self, file_path: str | Path) -> np.ndarray:
        with open(file_path, "rb") as f:
            image_bytes = f.read()
        return self.compute_embedding(self.preprocess_image(image_bytes))

    def get_embedding_dimension(self) -> int:
        dummy = torch.zeros(1, 3, 224, 224).to(self.device)
        if self.config.precision == "fp16":
            dummy = dummy.half()
        with torch.no_grad():
            embedding = self.model.encode_image(dummy)
        return embedding.shape[-1]


class DinoV2InferenceManager(BaseInferenceManager):
    """Manages DINO v2 model loading and image embedding computation."""

    def __init__(self, config: DinoV2Config, models_dir: Path):
        self.config = config
        self.models_dir = models_dir
        
        # Map simple model_type to full HF model name
        model_name_map = {
            "small": "facebook/dinov2-with-registers-small",
            "giant": "facebook/dinov2-with-registers-giant"
        }
        self.model_name = model_name_map.get(self.config.model_type, self.config.model_type)
        
        self.device = self._resolve_device()
        
        # Load processor and model
        print(f"Loading DINO v2 model {self.model_name}...")
        self.processor = AutoImageProcessor.from_pretrained(
            self.model_name, 
            cache_dir=str(self.models_dir),
            use_fast=self.config.use_fast
        )
        self.model = AutoModel.from_pretrained(
            self.model_name, 
            cache_dir=str(self.models_dir)
        )
        self.model.to(self.device)
        self.model.eval()

    def _resolve_device(self) -> str:
        if self.config.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.config.device

    def get_embedding(self, file_path: str | Path) -> np.ndarray:
        image = Image.open(file_path).convert("RGB")
        
        # Control granulation via image_size
        inputs = self.processor(
            images=image, 
            return_tensors="pt", 
            size={"height": self.config.image_size, "width": self.config.image_size}
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # DINOv2 uses the CLS token for the global representation
            # We take the pooler_output if available, otherwise CLS token
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                embedding = outputs.pooler_output
            else:
                embedding = outputs.last_hidden_state[:, 0, :]
            
            # Normalize embedding
            embedding /= embedding.norm(dim=-1, keepdim=True)
            
        return embedding.cpu().numpy().flatten()

    def get_embedding_dimension(self) -> int:
        return self.model.config.hidden_size


class InferenceManager:
    """Factory class for creating inference managers."""

    def __new__(cls, config, provider: str = "open_clip"):
        project_root = Path(__file__).parent.parent.parent.parent
        models_dir = project_root / "models"
        
        if provider == "open_clip":
            return OpenClipInferenceManager(config, models_dir)
        elif provider == "dino_v2":
            return DinoV2InferenceManager(config, models_dir)
        else:
            raise ValueError(f"Unknown provider: {provider}")
