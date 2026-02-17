"""Similarity computation strategies and computer."""

from .base import SimilarityStrategy
from .concatenation import ConcatenationStrategy
from .similarity_computer import SimilarityComputer
from .weighted_sum import WeightedSumStrategy
from .svm_strategy import SVMStrategy

__all__ = [
    "SimilarityStrategy",
    "ConcatenationStrategy",
    "WeightedSumStrategy",
    "SimilarityComputer",
    "SVMStrategy",
]
