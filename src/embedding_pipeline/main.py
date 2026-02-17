"""CLI entry point for embedding pipeline."""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from .config import ConfigManager
from .core import EmbeddingCombiner, FolderProcessor, InferenceManager
from .optimization import PSOOptimizer, SVMTrainer
from .similarity import SimilarityComputer
import sys


def cmd_compute_embeddings(args, config):
    """Compute embeddings for all images in input folder."""
    provider = "open_clip"
    if args.dino_v2:
        provider = "dino_v2"
    elif args.open_clip:
        provider = "open_clip"
    else:
        # Default to config or open_clip if not specified
        provider = config.provider

    model_config = config.dino_v2 if provider == "dino_v2" else config.open_clip
    
    if provider == "open_clip":
        print(f"Loading OpenCLIP model {model_config.name}...")
    else:
        print(f"Loading DINO v2 model {model_config.model_type}...")

    inference_manager = InferenceManager(model_config, provider=provider)

    print(f"Processing images from {args.input}...")
    processor = FolderProcessor(inference_manager)
    processor.process_folder(
        source_root=Path(args.input),
        target_root=Path(args.output),
        show_progress=True,
    )
    print(f"Embeddings saved to {args.output}")


def cmd_combine_embeddings(args, config):
    """Combine individual CSV embeddings per subfolder."""
    combiner = EmbeddingCombiner()
    results = combiner.combine_embeddings(
        input_folder=Path(args.input),
        output_folder=Path(args.output),
    )
    print(f"Combined {len(results)} subfolders. Output in {args.output}")
    for name, path in results.items():
        print(f"  - {name}: {path}")


def cmd_compute_similarity(args, config):
    """Compute NxN similarity matrix."""
    # Load combined embeddings
    combiner = EmbeddingCombiner()
    embeddings = combiner.load_combined_embeddings(Path(args.input))

    if not embeddings:
        print("No embeddings found in input folder.")
        sys.exit(1)

    # Create computer
    method = args.method or config.similarity.method
    weights = args.weights if args.weights else config.similarity.weights
    
    computer = SimilarityComputer.from_method(method, weights=weights)

    print(f"Computing {len(embeddings)}x{len(embeddings)} similarity matrix using {method}...")
    
    # Get transform threshold if specified
    transform_threshold = getattr(args, 'transform_threshold', None) or config.similarity.transform_threshold
    
    matrix, names = computer.compute_similarity_matrix(
        embeddings, show_progress=True, transform_threshold=transform_threshold
    )
    
    if transform_threshold is not None:
        print(f"Applied threshold transformation with threshold={transform_threshold}")

    # Save matrix
    output_path = Path(args.output)
    np.savetxt(output_path, matrix, delimiter=",")
    print(f"Similarity matrix saved to {output_path}")

    # Save labels
    labels_path = output_path.with_suffix(".labels.txt")
    with open(labels_path, "w") as f:
        for name in names:
            f.write(f"{name}\n")
    print(f"Labels saved to {labels_path}")


def cmd_optimize_weights(args, config):
    """Optimize weights using PSO."""
    # Load embeddings
    combiner = EmbeddingCombiner()
    embeddings = combiner.load_combined_embeddings(Path(args.embeddings))

    if not embeddings:
        print("No embeddings found.")
        sys.exit(1)

    # Load ground truth
    ground_truth = np.loadtxt(args.ground_truth, delimiter=",")

    # Get number of images from first embedding
    first_emb = next(iter(embeddings.values()))
    n_weights = first_emb.shape[0]

    # Override config with CLI args if provided
    pso_config = config.pso
    if args.n_particles:
        pso_config.n_particles = args.n_particles
    if args.n_iterations:
        pso_config.n_iterations = args.n_iterations
    if args.n_workers:
        pso_config.n_workers = args.n_workers

    print(f"Running PSO optimization with {pso_config.n_particles} particles, "
          f"{pso_config.n_iterations} iterations, {pso_config.n_workers} workers...")

    optimizer = PSOOptimizer(pso_config, n_weights=n_weights)

    def progress_callback(iteration, total, best_cost):
        print(f"  Iteration {iteration}/{total}, best cost: {best_cost:.6f}")

    # Load labels if provided
    gt_labels = None
    if args.gt_labels:
        with open(args.gt_labels, "r") as f:
            content = f.read().strip()
            # Handle comma-separated strings
            gt_labels = [s.strip() for s in content.split(",") if s.strip()]

    result = optimizer.optimize(embeddings, ground_truth, gt_labels=gt_labels, progress_callback=progress_callback)

    # Normalize weights
    best_weights = result.best_weights
    sum_weights = np.sum(best_weights)
    if sum_weights > 0:
        best_weights = best_weights / sum_weights

    # Save weights
    np.savetxt(args.output, best_weights.reshape(1, -1), delimiter=",")
    print(f"Optimal weights saved to {args.output}")
    print(f"Final cost: {result.best_cost:.6f}")
    print(f"Normalized Weights: {best_weights}")


def cmd_train_svm(args, config):
    """Train SVM model for similarity prediction."""
    # Load embeddings
    combiner = EmbeddingCombiner()
    embeddings = combiner.load_combined_embeddings(Path(args.embeddings))

    if not embeddings:
        print("No embeddings found.")
        sys.exit(1)

    # Load ground truth
    ground_truth = np.loadtxt(args.ground_truth, delimiter=",")

    # Load labels if provided
    gt_labels = None
    if args.gt_labels:
        with open(args.gt_labels, "r") as f:
            content = f.read().strip()
            # Handle comma-separated strings
            gt_labels = [s.strip() for s in content.split(",") if s.strip()]

    # Override config with CLI args if provided
    svm_config = config.svm
    if args.kernel:
        svm_config.kernel = args.kernel
    if args.c:
        svm_config.C = args.c

    print(f"Training SVM with kernel={svm_config.kernel}, C={svm_config.C}...")

    trainer = SVMTrainer(svm_config)

    def progress_callback(msg):
        print(f"  {msg}")

    X, y = trainer.create_training_data(embeddings, ground_truth, gt_labels=gt_labels)
    result = trainer.train(X, y, progress_callback=progress_callback)

    # Save model
    output_path = Path(args.output)
    trainer.save_model(result.model, output_path)
    print(f"SVM model saved to {output_path}")
    print(f"Train MSE: {result.train_mse:.6f}, Val MSE: {result.val_mse:.6f}")


def cmd_sum_embeddings(args, config):
    """Sum embeddings from two folders and save to a third folder."""
    input_a = Path(args.input_a)
    input_b = Path(args.input_b)
    output = Path(args.output)

    if not input_a.exists() or not input_b.exists():
        print(f"Error: One of the input folders does not exist: {input_a} or {input_b}")
        sys.exit(1)

    output.mkdir(parents=True, exist_ok=True)

    # Get all CSV files from input_a
    csv_files = list(input_a.glob("**/*.csv"))
    print(f"Found {len(csv_files)} embedding files. Summing...")

    count = 0
    for csv_a in csv_files:
        relative = csv_a.relative_to(input_a)
        csv_b = input_b / relative

        if not csv_b.exists():
            print(f"Warning: Corresponding file not found in input_b: {relative}")
            continue

        # Load and sum
        data_a = np.loadtxt(csv_a, delimiter=",")
        data_b = np.loadtxt(csv_b, delimiter=",")

        if data_a.shape != data_b.shape:
            print(f"Warning: Shape mismatch for {relative}: {data_a.shape} vs {data_b.shape}")
            continue

        summed = data_a + data_b
        
        # Save to output
        out_path = output / relative
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(out_path, summed, delimiter=",")
        count += 1

    print(f"Successfully summed and saved {count} embedding files to {output}")


def cmd_gui(args, config):
    """Launch browser GUI."""
    from .gui import create_app

    import uvicorn

    app = create_app(config, dark_mode=args.dark_mode, no_tsne=args.no_tsne)
    mode_str = " (dark mode)" if args.dark_mode else ""
    print(f"Starting GUI server{mode_str} on http://localhost:{args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)




def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Image Embedding Pipeline - Multi-Model Support"
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to YAML config file (optional)",
    )

    subparsers = parser.add_subparsers(dest="mode", required=True)

    # compute-embeddings
    p1 = subparsers.add_parser(
        "compute-embeddings",
        help="Compute embeddings for images in folder",
    )
    p1.add_argument("-i", "--input", type=Path, required=True, help="Input folder with images")
    p1.add_argument("-o", "--output", type=Path, required=True, help="Output folder for CSVs")
    
    group = p1.add_mutually_exclusive_group()
    group.add_argument("--open-clip", action="store_true", help="Use OpenCLIP model")
    group.add_argument("--dino-v2", action="store_true", help="Use DINO v2 with registers model")

    # combine-embeddings
    p2 = subparsers.add_parser(
        "combine-embeddings",
        help="Combine individual CSV embeddings per subfolder",
    )
    p2.add_argument("-i", "--input", type=Path, required=True, help="Input folder with CSVs")
    p2.add_argument("-o", "--output", type=Path, required=True, help="Output folder for combined CSVs")

    # compute-similarity
    p3 = subparsers.add_parser(
        "compute-similarity",
        help="Compute NxN similarity matrix",
    )
    p3.add_argument("-i", "--input", type=Path, required=True, help="Folder with combined CSVs")
    p3.add_argument("-o", "--output", type=Path, required=True, help="Output CSV file")
    p3.add_argument(
        "--method",
        choices=["concatenation", "weighted_sum"],
        help="Similarity method",
    )
    p3.add_argument(
        "--weights",
        type=float,
        nargs="+",
        help="Weights for weighted_sum method",
    )
    p3.add_argument(
        "--transform-threshold",
        type=float,
        help="Apply threshold transformation (subtract threshold, clamp to 0, rescale to [0,1])",
    )

    # optimize-weights
    p4 = subparsers.add_parser(
        "optimize-weights",
        help="Optimize weights using PSO",
    )
    p4.add_argument("-e", "--embeddings", type=Path, required=True, help="Folder with combined CSVs")
    p4.add_argument("-g", "--ground-truth", type=Path, required=True, help="Ground truth CSV")
    p4.add_argument("-o", "--output", type=Path, required=True, help="Output weights CSV")
    p4.add_argument("-l", "--gt-labels", type=Path, help="CSV/Txt file with GT labels in order")
    p4.add_argument("--n-particles", type=int, help="Number of particles")
    p4.add_argument("--n-iterations", type=int, help="Number of iterations")
    p4.add_argument("--n-workers", type=int, help="Number of parallel workers")

    # gui
    p5 = subparsers.add_parser("gui", help="Launch browser GUI")
    p5.add_argument("--port", type=int, default=8000, help="Server port")
    p5.add_argument("--dark-mode", action="store_true", help="Start GUI in dark mode")
    p5.add_argument("--no-tsne", action="store_true", help="Disable 3D t-SNE plot")

    # train-svm
    p6 = subparsers.add_parser("train-svm", help="Train SVM model for similarity prediction")
    p6.add_argument("-e", "--embeddings", type=Path, required=True, help="Folder with combined CSVs")
    p6.add_argument("-g", "--ground-truth", type=Path, required=True, help="Ground truth CSV")
    p6.add_argument("-o", "--output", type=Path, required=True, help="Output .pkl model path")
    p6.add_argument("-l", "--gt-labels", type=Path, help="CSV/Txt file with GT labels in order")
    p6.add_argument("--kernel", type=str, help="SVM kernel (linear, rbf, poly, etc.)")
    p6.add_argument("--c", type=float, help="SVM C parameter")

    # sum-embeddings
    p7 = subparsers.add_parser("sum-embeddings", help="Sum embeddings from two folders")
    p7.add_argument("-A", "--input-a", type=Path, required=True, help="First input folder")
    p7.add_argument("-B", "--input-b", type=Path, required=True, help="Second input folder")
    p7.add_argument("-o", "--output", type=Path, required=True, help="Output folder")


    args = parser.parse_args()

    # Load config (defaults to default_config.yaml if args.config is None)
    config = ConfigManager.load(args.config)

    # Dispatch
    commands = {
        "compute-embeddings": cmd_compute_embeddings,
        "combine-embeddings": cmd_combine_embeddings,
        "compute-similarity": cmd_compute_similarity,
        "optimize-weights": cmd_optimize_weights,
        "gui": cmd_gui,
        "train-svm": cmd_train_svm,
        "sum-embeddings": cmd_sum_embeddings,
    }

    commands[args.mode](args, config)


if __name__ == "__main__":
    main()
