"""FastAPI application for GUI."""

import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..config import AppConfig
from ..core import EmbeddingCombiner
from ..similarity import SimilarityComputer

logger = logging.getLogger(__name__)


class SimilarityRequest(BaseModel):
    """Request for similarity computation."""

    embeddings_folder: str
    method: str = "weighted_sum"
    weights: list[float] | None = None
    weight_bounds: list[tuple[float, float]] | None = None
    transform_threshold: float | None = None
    model_path: str | None = None


class CompareSubfoldersRequest(BaseModel):
    """Request for comparing two subfolders."""

    embeddings_folder: str
    left_subfolder: str
    right_subfolder: str
    method: str = "weighted_sum"
    weights: list[float] | None = None
    display_mode: str = "corresponding"
    transform_threshold: float | None = None
    model_path: str | None = None


class TopKRequest(BaseModel):
    """Request for Top-K similar components."""

    embeddings_folder: str
    component_name: str
    k: int = 5
    method: str = "weighted_sum"
    weights: list[float] | None = None
    similarity_threshold: float = 0.0
    transform_threshold: float | None = None
    model_path: str | None = None


def create_app(
    config: AppConfig, dark_mode: bool = False, no_tsne: bool = False
) -> FastAPI:
    """Create FastAPI application.

    Args:
        config: Application configuration.
        dark_mode: Whether to start in dark mode.
    """
    app = FastAPI(title="Image Embedding Pipeline GUI")

    # Store for active WebSocket connections
    active_connections: list[WebSocket] = []

    # Serve static files
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        """Serve main HTML page with dark mode injection."""
        html_path = Path(__file__).parent / "static" / "index.html"
        if html_path.exists():
            content = html_path.read_text()
            # Inject dark mode class if enabled
            if dark_mode:
                content = content.replace('<body class="', '<body class="dark ')
                content = content.replace("<body>", '<body class="dark">')
            # Inject dark mode state for JavaScript
            inject_script = f"<script>window.DARK_MODE = {str(dark_mode).lower()}; window.HIDE_TSNE = {str(no_tsne).lower()};</script>"
            content = content.replace("</head>", f"{inject_script}</head>")
            return HTMLResponse(content)
        return HTMLResponse("<h1>Static files not found</h1>")

    @app.get("/api/subfolders")
    async def list_subfolders(folder: str) -> dict[str, Any]:
        """List available subfolders with combined embeddings."""
        folder_path = Path(folder)
        if not folder_path.exists():
            return {"error": "Folder not found", "subfolders": []}

        combiner = EmbeddingCombiner()
        info = combiner.get_embedding_info(folder_path)

        subfolders = [f.stem for f in sorted(folder_path.glob("*.csv"))]

        # Get embedding labels from first subfolder's source (if available)
        embedding_labels = _get_embedding_labels_from_folder(folder_path, subfolders)

        return {
            "subfolders": subfolders,
            "info": info,
            "embedding_labels": embedding_labels,
        }

    @app.get("/api/validate-images-folder")
    async def validate_images_folder(folder: str) -> dict[str, Any]:
        """Validate images folder and return info."""
        folder_path = Path(folder)
        if not folder_path.exists():
            return {"error": "Folder not found"}

        # Count subfolders (components) and images
        subfolders = [d for d in folder_path.iterdir() if d.is_dir()]
        if not subfolders:
            return {"error": "No component folders found"}

        # Get image count from first subfolder
        first_subfolder = subfolders[0]
        images = list(first_subfolder.glob("*.png")) + list(
            first_subfolder.glob("*.jpg")
        )

        # Get embedding labels from images
        embedding_labels = _get_embedding_labels_from_images(first_subfolder)

        return {
            "n_components": len(subfolders),
            "n_images_per_component": len(images),
            "embedding_labels": embedding_labels,
        }

    @app.get("/api/images")
    async def list_images(folder: str, subfolder: str) -> dict[str, Any]:
        """List images in a subfolder."""
        try:
            folder_path = Path(folder)
            subfolder_path = folder_path / subfolder

            if not subfolder_path.exists():
                return {"error": "Component not found", "images": []}

            images = sorted(
                [f.name for f in subfolder_path.glob("*.png")],
                key=lambda x: int(x.split("_")[0]) if x.split("_")[0].isdigit() else 0,
            )

            return {"images": images}
        except Exception as e:
            logger.error(f"Error in list_images: {str(e)}", exc_info=True)
            return {"error": f"Failed to list images: {str(e)}", "images": []}

    @app.get("/api/image")
    async def get_image(folder: str, subfolder: str, filename: str):
        """Serve an image file."""
        image_path = Path(folder) / subfolder / filename
        if image_path.exists():
            return FileResponse(image_path)
        return {"error": "Image not found"}

    @app.post("/api/compute-similarity")
    async def compute_similarity(request: SimilarityRequest) -> dict[str, Any]:
        """Compute NxN similarity matrix."""
        try:
            folder_path = Path(request.embeddings_folder)
            if not folder_path.exists():
                return {"error": "Folder not found"}

            combiner = EmbeddingCombiner()
            embeddings = combiner.load_combined_embeddings(folder_path)

            if not embeddings:
                return {"error": "No embeddings found"}

            computer = SimilarityComputer.from_method(
                request.method, weights=request.weights, model_path=request.model_path
            )

            matrix, names = computer.compute_similarity_matrix(
                embeddings,
                transform_threshold=request.transform_threshold,
            )

            return {
                "matrix": matrix.tolist(),
                "labels": names,
                "method": request.method,
                "transform_applied": request.transform_threshold is not None,
            }
        except Exception as e:
            logger.error(f"Error in compute_similarity: {str(e)}", exc_info=True)
            return {"error": f"Similarity computation failed: {str(e)}"}

    @app.post("/api/compare-subfolders")
    async def compare_subfolders(request: CompareSubfoldersRequest) -> dict[str, Any]:
        """Compare two specific subfolders."""
        try:
            folder_path = Path(request.embeddings_folder)
            if not folder_path.exists():
                return {"error": "Folder not found"}

            combiner = EmbeddingCombiner()
            embeddings = combiner.load_combined_embeddings(folder_path)

            if request.left_subfolder not in embeddings:
                return {"error": f"Component {request.left_subfolder} not found"}
            if request.right_subfolder not in embeddings:
                return {"error": f"Component {request.right_subfolder} not found"}

            emb_left = embeddings[request.left_subfolder]
            emb_right = embeddings[request.right_subfolder]

            computer = SimilarityComputer.from_method(
                request.method, weights=request.weights, model_path=request.model_path
            )

            comparison = computer.compare_two_subfolders(
                emb_left,
                emb_right,
                transform_threshold=request.transform_threshold,
            )

            return {
                "left_subfolder": request.left_subfolder,
                "right_subfolder": request.right_subfolder,
                "overall_similarity": comparison["overall_similarity"],
                "individual_similarities": comparison["individual_similarities"],
                "pairwise_matrix": comparison["pairwise_matrix"],
                "display_mode": request.display_mode,
            }
        except Exception as e:
            logger.error(f"Error in compare_subfolders: {str(e)}", exc_info=True)
            return {"error": f"Component comparison failed: {str(e)}"}

    @app.get("/api/embedding/{folder:path}/{subfolder}")
    async def get_embedding(folder: str, subfolder: str) -> dict[str, Any]:
        """Get embedding matrix for a subfolder."""
        try:
            folder_path = Path(folder)
            csv_path = folder_path / f"{subfolder}.csv"

            if not csv_path.exists():
                return {"error": "Embedding not found"}

            import numpy as np

            embedding = np.loadtxt(csv_path, delimiter=",")
            if embedding.ndim == 1:
                embedding = embedding.reshape(1, -1)

            return {
                "subfolder": subfolder,
                "embedding": embedding.tolist(),
                "shape": list(embedding.shape),
            }
        except Exception as e:
            logger.error(f"Error in get_embedding: {str(e)}", exc_info=True)
            return {"error": f"Failed to load embedding: {str(e)}"}

    @app.get("/api/load-csv")
    async def load_csv(path: str) -> dict[str, Any]:
        """Load a CSV file (for ground truth matrix or weights)."""
        try:
            import numpy as np

            csv_path = Path(path)
            if not csv_path.exists():
                return {"error": f"File not found: {path}"}

            data = np.loadtxt(csv_path, delimiter=",")

            # Handle 1D arrays (single row of weights)
            if data.ndim == 1:
                return {
                    "matrix": data.tolist(),
                    "shape": [len(data)],
                }

            return {
                "matrix": data.tolist(),
                "shape": list(data.shape),
            }
        except Exception as e:
            logger.error(f"Error in load_csv: {str(e)}", exc_info=True)
            return {"error": f"Failed to load CSV: {str(e)}"}

    @app.post("/api/compute-tsne")
    async def compute_tsne(
        embeddings_folder: str,
        perplexity: float = 30.0,
        max_iter: int = 1000,
        selected_indices: str | None = None,  # Comma-separated indices
        scale_factor: float = 1.0,
        transform_threshold: float | None = None,  # Threshold transformation
        cluster_method: str | None = None,  # "kmeans" or "dbscan"
        cluster_k: int = 3,  # K for K-Means
        cluster_eps: float = 0.5,  # Epsilon for DBSCAN
        cluster_min_samples: int = 2,  # Min samples for DBSCAN
    ) -> dict[str, Any]:
        """Compute t-SNE on all embeddings with optional clustering."""
        from ..visualization import TSNEComputer
        from ..config import TSNEConfig

        folder_path = Path(embeddings_folder)
        if not folder_path.exists():
            return {"error": "Folder not found"}

        combiner = EmbeddingCombiner()
        embeddings = combiner.load_combined_embeddings(folder_path)

        if not embeddings:
            return {"error": "No embeddings found"}

        # Parse selected indices
        parsed_indices = None
        if selected_indices:
            try:
                parsed_indices = [int(x.strip()) for x in selected_indices.split(",")]
            except ValueError:
                return {"error": "Invalid selected_indices format"}

        # Use config with provided parameters
        tsne_config = TSNEConfig(
            perplexity=perplexity,
            max_iter=max_iter,
        )
        computer = TSNEComputer(tsne_config)

        try:
            result = computer.compute_tsne(
                embeddings,
                selected_indices=parsed_indices,
                scale_factor=scale_factor,
                transform_threshold=transform_threshold,
                cluster_method=cluster_method,
                cluster_k=cluster_k,
                cluster_eps=cluster_eps,
                cluster_min_samples=cluster_min_samples,
            )

            return {
                "coordinates": result.coordinates.tolist(),
                "labels": result.labels,
                "subfolder_indices": result.subfolder_indices,
                "image_indices": result.image_indices,
                "subfolders": result.subfolders,
                "cluster_labels": result.cluster_labels,
            }
        except Exception as e:
            return {"error": f"t-SNE computation failed: {str(e)}"}

    @app.post("/api/top-k")
    async def top_k_search(request: TopKRequest) -> dict[str, Any]:
        """Find Top-K similar components using a weighted KD-Tree for fast cosine search."""
        try:
            folder_path = Path(request.embeddings_folder)
            if not folder_path.exists():
                return {"error": "Folder not found"}

            combiner = EmbeddingCombiner()
            embeddings = combiner.load_combined_embeddings(folder_path)

            if request.component_name not in embeddings:
                return {"error": f"Component {request.component_name} not found"}

            import numpy as np
            from scipy.spatial import KDTree

            names = sorted(embeddings.keys())
            n_components = len(names)
            if n_components == 0:
                return {"component": request.component_name, "matches": []}

            # Prepare weighted embeddings for KD-Tree
            # Each component is a matrix (N_images x D)
            # We flatten it to a single vector of size (N_images * D)
            # And scale each block by sqrt(weight) to support weighted cosine similarity via Euclidean distance

            first_emb = embeddings[names[0]]
            n_images, d = first_emb.shape

            # If method is SVM, we must do a linear scan because we can't use KDTree for arbitrary kernels
            if request.method == "svm":
                if not request.model_path:
                    return {"error": "Method 'svm' requires a model_path"}
                
                computer = SimilarityComputer.from_method(
                    request.method, model_path=request.model_path
                )
                
                matches = []
                target_emb = embeddings[request.component_name]
                
                for name in names:
                    if name == request.component_name:
                        continue
                    
                    raw_similarity = computer.strategy.compute_similarity(
                        target_emb, embeddings[name]
                    )
                    
                    if raw_similarity < request.similarity_threshold:
                        continue
                        
                    display_score = raw_similarity
                    if request.transform_threshold is not None:
                        display_score = max(0.0, raw_similarity - request.transform_threshold)
                        if display_score > 0:
                            display_score = display_score / (1.0 - request.transform_threshold)
                            
                    matches.append({
                        "name": name,
                        "score": float(display_score),
                        "raw_similarity": float(raw_similarity),
                    })
                
                # Sort by score and take top K
                matches.sort(key=lambda x: x["score"], reverse=True)
                return {"component": request.component_name, "matches": matches[:request.k]}

            # If method is concatenation, force equal weights
            weights = request.weights
            if request.method == "concatenation":
                weights = None

            # Use weights or default to equal weights
            if weights is None:
                weights = [1.0 / n_images] * n_images
            elif len(weights) < n_images:
                weights = weights + [0.0] * (n_images - len(weights))

            # Normalize weights to sum to 1 to preserve cosine similarity range [0, 1]
            weights_arr = np.array(weights[:n_images])
            sum_weights = np.sum(weights_arr)
            if sum_weights > 0:
                weights_arr = weights_arr / sum_weights
            sqrt_weights = np.sqrt(weights_arr)

            def get_weighted_flat(emb):
                if request.method == "concatenation":
                    # For concatenation: Flatten first, then normalize entire ND vector
                    flat = emb.flatten()
                    norm = np.linalg.norm(flat)
                    if norm > 0:
                        flat = flat / norm
                    return flat
                else:
                    # For weighted_sum: Normalize each image individually, then scale by sqrt(weight)
                    norms = np.linalg.norm(emb, axis=1, keepdims=True)
                    norms = np.where(norms == 0, 1, norms)
                    normalized = emb / norms
                    # Scale each normalized image embedding by its sqrt weight
                    weighted = normalized * sqrt_weights.reshape(-1, 1)
                    return weighted.flatten()

            all_weighted_embs = np.array(
                [get_weighted_flat(embeddings[name]) for name in names]
            )
            target_weighted = get_weighted_flat(
                embeddings[request.component_name]
            ).reshape(1, -1)

            # Build KDTree and query
            tree = KDTree(all_weighted_embs)
            # Find K+1 because the component itself will be the closest match (distance 0)
            distances, indices = tree.query(
                target_weighted, k=min(request.k + 1, n_components)
            )

            # Handle distances and indices if they are scalars (when k=1)
            if np.isscalar(distances):
                distances = np.array([distances])
                indices = np.array([indices])
            else:
                distances = distances[0]
                indices = indices[0]

            matches = []
            for d_val, idx in zip(distances, indices):
                name = names[idx]
                if name == request.component_name:
                    continue

                # Convert Euclidean distance of normalized vectors back to cosine similarity
                # s = 1 - d^2/2
                # This is the "overall similarity value"
                raw_similarity = 1.0 - (float(d_val) ** 2) / 2.0

                # Filter by search threshold (refers to overall similarity value)
                if raw_similarity < request.similarity_threshold:
                    continue

                # Final score for display: apply transformation if enabled for consistent matrix matching
                display_score = raw_similarity
                if request.transform_threshold is not None:
                    display_score = max(
                        0.0, raw_similarity - request.transform_threshold
                    )
                    if display_score > 0:
                        display_score = display_score / (
                            1.0 - request.transform_threshold
                        )

                matches.append(
                    {
                        "name": name,
                        "score": float(display_score),
                        "raw_similarity": float(raw_similarity),
                    }
                )

                if len(matches) >= request.k:
                    break

            return {"component": request.component_name, "matches": matches}
        except Exception as e:
            logger.error(f"Error in top_k_search: {str(e)}", exc_info=True)
            return {"error": f"Top-K search failed: {str(e)}"}

    @app.websocket("/ws/progress")
    async def progress_websocket(websocket: WebSocket):
        """WebSocket for real-time progress updates."""
        await websocket.accept()
        active_connections.append(websocket)
        try:
            while True:
                # Keep connection alive
                data = await websocket.receive_text()
                # Echo back any messages
                await websocket.send_text(f"Received: {data}")
        except WebSocketDisconnect:
            active_connections.remove(websocket)

    async def broadcast_progress(message: dict):
        """Broadcast progress to all connected clients."""
        for connection in active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except Exception:
                pass

    return app


def _get_embedding_labels_from_folder(
    folder_path: Path, subfolders: list[str]
) -> list[str]:
    """Extract embedding labels from folder structure.

    Labels are extracted from the text portion of filenames like {index}_{text}.csv
    """
    labels = []

    # Try to get labels from source embeddings folder (before combining)
    # Look for individual CSV files in subdirectories
    for subfolder in subfolders:
        subfolder_path = folder_path.parent / "embeddings" / subfolder
        if subfolder_path.exists():
            csv_files = sorted(
                subfolder_path.glob("*.csv"),
                key=lambda x: (
                    int(x.stem.split("_")[0]) if x.stem.split("_")[0].isdigit() else 0
                ),
            )
            for csv_file in csv_files:
                parts = csv_file.stem.split("_", 1)
                if len(parts) > 1:
                    labels.append(parts[1])
                else:
                    labels.append(csv_file.stem)
            if labels:
                return labels

    return labels


def _get_embedding_labels_from_images(folder_path: Path) -> list[str]:
    """Extract embedding labels from image filenames.

    Labels are extracted from the text portion of filenames like {index}_{text}.png
    """
    labels = []

    image_files = sorted(
        list(folder_path.glob("*.png")) + list(folder_path.glob("*.jpg")),
        key=lambda x: (
            int(x.stem.split("_")[0]) if x.stem.split("_")[0].isdigit() else 0
        ),
    )

    for img_file in image_files:
        parts = img_file.stem.split("_", 1)
        if len(parts) > 1:
            labels.append(parts[1])
        else:
            labels.append(img_file.stem)

    return labels
