"""Particle Swarm Optimization for weight discovery."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..config import PSOConfig
from ..similarity.weighted_sum import WeightedSumStrategy
from .losses import get_loss_function


@dataclass
class PSOResult:
    """Result of PSO optimization."""

    best_weights: np.ndarray
    best_cost: float
    cost_history: list[float]
    n_iterations: int


def _evaluate_particle(
    particle: np.ndarray,
    embeddings: dict[str, np.ndarray],
    ground_truth: np.ndarray,
    subfolder_names: list[str],
    cost_function_name: str = "mse",
    threshold: float = 0.93,
) -> float:
    """Evaluate cost function for a single particle (for multiprocessing).

    Args:
        particle: Weight vector.
        embeddings: Dict of embeddings.
        ground_truth: Ground truth similarity matrix.
        subfolder_names: Ordered list of subfolder names.

    Returns:
        Mean squared error cost.
    """
    from ..similarity.similarity_computer import SimilarityComputer
    strategy = WeightedSumStrategy(weights=particle.tolist())
    computer = SimilarityComputer(strategy)

    # Reorder embeddings to match ground truth order
    ordered_embeddings = {name: embeddings[name] for name in subfolder_names}
    computed_matrix, _ = computer.compute_similarity_matrix(ordered_embeddings)

    loss_fn = get_loss_function(cost_function_name)
    if cost_function_name == "thresholded":
        return loss_fn(computed_matrix, ground_truth, threshold=threshold)
    return loss_fn(computed_matrix, ground_truth)


class PSOOptimizer:
    """Particle Swarm Optimization for finding optimal weights."""

    def __init__(
        self,
        config: PSOConfig | None = None,
        n_weights: int = 12,
    ):
        """Initialize PSO optimizer.

        Args:
            config: PSO configuration. Uses defaults if None.
            n_weights: Number of weights to optimize.
        """
        self.config = config or PSOConfig()
        self.n_weights = n_weights
        self.bounds = self._resolve_bounds()

    def _resolve_bounds(self) -> list[tuple[float, float]]:
        """Resolve per-weight bounds."""
        if self.config.per_weight_bounds:
            return self.config.per_weight_bounds
        return [(self.config.lower_bound, self.config.upper_bound)] * self.n_weights

    def _initialize_particles(self) -> np.ndarray:
        """Initialize particles within bounds."""
        particles = np.zeros((self.config.n_particles, self.n_weights))
        for i in range(self.n_weights):
            low, high = self.bounds[i]
            particles[:, i] = np.random.uniform(low, high, self.config.n_particles)
        return particles

    def _clip_to_bounds(self, particles: np.ndarray) -> np.ndarray:
        """Clip particle positions to bounds."""
        for i in range(self.n_weights):
            low, high = self.bounds[i]
            particles[:, i] = np.clip(particles[:, i], low, high)
        return particles

    def optimize(
        self,
        embeddings: dict[str, np.ndarray],
        ground_truth: np.ndarray,
        gt_labels: list[str] | None = None,
        progress_callback: Callable[[int, int, float], None] | None = None,
    ) -> PSOResult:
        """Run PSO optimization.

        Args:
            embeddings: Dict of embeddings per subfolder.
            ground_truth: Ground truth NxN similarity matrix.
            gt_labels: Optional list of labels for x/y axis of ground_truth.
            progress_callback: Optional callback(iteration, total, best_cost).

        Returns:
            PSOResult with optimal weights and cost history.
        """
        # Determine subfolder order to match ground truth
        if gt_labels:
            subfolder_names = gt_labels
            # Verify all labels exist in embeddings
            for name in subfolder_names:
                if name not in embeddings:
                    raise ValueError(f"Label '{name}' in gt_labels not found in embeddings")
        else:
            subfolder_names = sorted(embeddings.keys())

        # Initialize particles and velocities
        particles = self._initialize_particles()
        velocities = np.zeros_like(particles)

        # Initialize personal and global bests
        personal_best_positions = particles.copy()
        personal_best_costs = np.full(self.config.n_particles, np.inf)

        global_best_position = particles[0].copy()
        global_best_cost = np.inf

        cost_history = []

        for iteration in range(self.config.n_iterations):
            # Evaluate particles (parallel)
            costs = self._evaluate_particles_parallel(
                particles,
                embeddings,
                ground_truth,
                subfolder_names,
                cost_function_name=self.config.cost_function,
                threshold=self.config.cost_threshold,
            )

            # Update personal bests
            for i, cost in enumerate(costs):
                if cost < personal_best_costs[i]:
                    personal_best_costs[i] = cost
                    personal_best_positions[i] = particles[i].copy()

                # Update global best
                if cost < global_best_cost:
                    global_best_cost = cost
                    global_best_position = particles[i].copy()

            cost_history.append(global_best_cost)

            # Update velocities and positions
            for i in range(self.config.n_particles):
                r1, r2 = np.random.random(2)

                cognitive = (
                    self.config.c1 * r1 * (personal_best_positions[i] - particles[i])
                )
                social = self.config.c2 * r2 * (global_best_position - particles[i])

                velocities[i] = self.config.w * velocities[i] + cognitive + social
                particles[i] = particles[i] + velocities[i]

            # Clip to bounds
            particles = self._clip_to_bounds(particles)

            if progress_callback:
                progress_callback(
                    iteration + 1, self.config.n_iterations, global_best_cost
                )

        return PSOResult(
            best_weights=global_best_position,
            best_cost=global_best_cost,
            cost_history=cost_history,
            n_iterations=self.config.n_iterations,
        )

    def _evaluate_particles_parallel(
        self,
        particles: np.ndarray,
        embeddings: dict[str, np.ndarray],
        ground_truth: np.ndarray,
        subfolder_names: list[str],
        cost_function_name: str = "mse",
        threshold: float = 0.5,
    ) -> list[float]:
        """Evaluate all particles in parallel."""
        costs = [0.0] * len(particles)

        if self.config.n_workers <= 1:
            # Sequential evaluation
            for i, particle in enumerate(particles):
                costs[i] = _evaluate_particle(
                    particle,
                    embeddings,
                    ground_truth,
                    subfolder_names,
                    cost_function_name=cost_function_name,
                    threshold=threshold,
                )
        else:
            # Parallel evaluation
            with ProcessPoolExecutor(max_workers=self.config.n_workers) as executor:
                futures = {
                    executor.submit(
                        _evaluate_particle,
                        particle,
                        embeddings,
                        ground_truth,
                        subfolder_names,
                        cost_function_name,
                        threshold,
                    ): i
                    for i, particle in enumerate(particles)
                }
                for future in as_completed(futures):
                    idx = futures[future]
                    costs[idx] = future.result()

        return costs

    def save_weights(self, weights: np.ndarray, output_path: str) -> None:
        """Save optimal weights to CSV file."""
        np.savetxt(output_path, weights.reshape(1, -1), delimiter=",")

    def load_ground_truth(self, path: str) -> np.ndarray:
        """Load ground truth matrix from CSV (plain NxN, no headers)."""
        return np.loadtxt(path, delimiter=",")
