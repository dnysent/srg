"""Configuration management with YAML support and sensible defaults."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class OpenClipConfig:
    """OpenCLIP model configuration."""

    name: str = "ViT-B-32"
    pretrained: str = "laion400m_e32"
    image_size: int = 224
    precision: str = "fp32"  # fp16, fp32, amp
    device: str = "auto"  # auto, cpu, cuda
    quick_gelu: bool = False  # Force QuickGELU (use False for some models like ViT-bigG-14)


@dataclass
class DinoV2Config:
    """DINO v2 with registers configuration."""

    model_type: str = "small"  # small, giant
    image_size: int = 224  # Granulation/resolution control
    device: str = "auto"  # auto, cpu, cuda
    use_fast: bool = False  # Use fast processor (True) or slow processor (False)


@dataclass
class SVMConfig:
    """Support Vector Machine similarity configuration."""

    kernel: str = "rbf"
    C: float = 1.0
    gamma: str = "scale"
    epsilon: float = 0.1
    degree: int = 3
    test_size: float = 0.2
    random_state: int = 42
    cost_function: str = "mse"
    cost_threshold: float = 0.5


@dataclass
class PSOConfig:
    """Particle Swarm Optimization configuration."""

    n_particles: int = 30
    n_iterations: int = 100
    n_workers: int = 4
    w: float = 0.7  # Inertia weight
    c1: float = 1.5  # Cognitive coefficient
    c2: float = 1.5  # Social coefficient
    lower_bound: float = 0.0
    upper_bound: float = 1.0
    cost_function: str = "mse"  # mse, thresholded, binary_cross_entropy
    cost_threshold: float = 0.5  # Threshold for 'thresholded' loss
    per_weight_bounds: Optional[list[tuple[float, float]]] = None


@dataclass
class SimilarityConfig:
    """Similarity computation configuration."""

    method: str = "weighted_sum"  # concatenation, weighted_sum, svm
    weights: Optional[list[float]] = None
    weight_bounds: tuple[float, float] = (0.0, 1.0)
    transform_threshold: Optional[float] = None
    svm_model_path: Optional[str] = None


@dataclass
class TSNEConfig:
    """t-SNE visualization configuration."""

    n_components: int = 3  # 2D or 3D
    perplexity: float = 30.0
    learning_rate: float = 200.0
    max_iter: int = 1000
    random_state: int = 42


@dataclass
class AppConfig:
    """Complete application configuration."""

    open_clip: OpenClipConfig = field(default_factory=OpenClipConfig)
    dino_v2: DinoV2Config = field(default_factory=DinoV2Config)
    provider: str = "open_clip"  # "open_clip" or "dino_v2"
    pso: PSOConfig = field(default_factory=PSOConfig)
    similarity: SimilarityConfig = field(default_factory=SimilarityConfig)
    tsne: TSNEConfig = field(default_factory=TSNEConfig)
    svm: SVMConfig = field(default_factory=SVMConfig)


class ConfigManager:
    """Manages configuration loading and defaults."""

    @staticmethod
    def defaults() -> AppConfig:
        """Return default configuration."""
        return AppConfig()

    @staticmethod
    def load(config_path: Optional[Path] = None) -> AppConfig:
        """Load configuration from YAML file, merging with defaults.
        
        If config_path is None, attempts to load from the default config path.
        """
        if config_path is None:
            config_path = ConfigManager.get_default_config_path()

        config = ConfigManager.defaults()

        if not config_path.exists():
            return config

        try:
            with open(config_path) as f:
                data = yaml.safe_load(f) or {}
        except Exception:
            # Fallback to defaults if file exists but is malformed
            return config

        # Map YAML sections to AppConfig fields
        section_map = {
            "open_clip": "open_clip",
            "model": "open_clip",  # legacy support
            "dino_v2": "dino_v2",
            "dinov2": "dino_v2",   # legacy support
            "pso": "pso",
            "similarity": "similarity",
            "tsne": "tsne",
            "svm": "svm",
        }

        for yaml_key, config_attr in section_map.items():
            if yaml_key in data and isinstance(data[yaml_key], dict):
                config_section = getattr(config, config_attr)
                for key, value in data[yaml_key].items():
                    if hasattr(config_section, key):
                        # Special handling for tuples/lists
                        if key == "per_weight_bounds" and value is not None:
                            value = [tuple(b) for b in value]
                        if key == "weight_bounds" and value is not None:
                            value = tuple(value)
                        setattr(config_section, key, value)

        # Top-level fields
        if "provider" in data:
            config.provider = data["provider"]

        return config

    @staticmethod
    def get_default_config_path() -> Path:
        """Return path to default config file."""
        return Path(__file__).parent / "default_config.yaml"
