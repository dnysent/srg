"""Folder processing with strict ordering by filename number prefix."""

import csv
import re
from pathlib import Path

from tqdm import tqdm

from .inference_manager import InferenceManager


def extract_index(filename: str) -> int:
    """Extract leading number from filename.

    Args:
        filename: Filename like '5_image.png' or '0_something.csv'

    Returns:
        The leading number, or -1 if not found.
    """
    match = re.match(r"^(\d+)_", filename)
    return int(match.group(1)) if match else -1


class FolderProcessor:
    """Processes folders of images, computing and saving embeddings."""

    def __init__(self, inference_manager: InferenceManager):
        """Initialize with inference manager.

        Args:
            inference_manager: The model wrapper for computing embeddings.
        """
        self.inference_manager = inference_manager

    def process_folder(
        self,
        source_root: Path,
        target_root: Path,
        show_progress: bool = True,
    ) -> None:
        """Process all subfolders, computing embeddings for each image.

        Images are processed in order by their filename number prefix (0-11).
        Each embedding is saved as a separate CSV file.

        Args:
            source_root: Root folder containing subfolders with images.
            target_root: Root folder for output CSV files.
            show_progress: Whether to show progress bar.
        """
        # Collect all image files
        image_files = []
        for subfolder in sorted(source_root.iterdir()):
            if subfolder.is_dir():
                images = sorted(
                    subfolder.glob("*.png"),
                    key=lambda p: extract_index(p.name),
                )
                image_files.extend(images)

        # Process with progress bar
        iterator = tqdm(image_files, desc="Computing embeddings") if show_progress else image_files

        for img_path in iterator:
            relative_path = img_path.relative_to(source_root)
            csv_path = target_root / relative_path.with_suffix(".csv")
            csv_path.parent.mkdir(parents=True, exist_ok=True)

            embedding = self.inference_manager.get_embedding(img_path)

            with open(csv_path, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(embedding)

    def process_single_subfolder(
        self,
        subfolder: Path,
        target_folder: Path,
    ) -> None:
        """Process a single subfolder.

        Args:
            subfolder: Folder containing images.
            target_folder: Folder for output CSV files.
        """
        target_folder.mkdir(parents=True, exist_ok=True)
        images = sorted(subfolder.glob("*.png"), key=lambda p: extract_index(p.name))

        for img_path in images:
            csv_path = target_folder / f"{img_path.stem}.csv"
            embedding = self.inference_manager.get_embedding(img_path)

            with open(csv_path, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(embedding)
