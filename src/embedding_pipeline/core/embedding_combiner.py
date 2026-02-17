"""Combine individual CSV embeddings into single file per subfolder."""

from pathlib import Path

import numpy as np

from .folder_processor import extract_index


class EmbeddingCombiner:
    """Combines individual embedding CSVs into single file per subfolder."""

    def combine_embeddings(
        self,
        input_folder: Path,
        output_folder: Path,
        generate_labels: bool = True,
        generate_identity_gt: bool = False,
    ) -> dict[str, Path]:
        """Combine individual CSV embeddings from each subfolder.

        For each subfolder containing individual embedding CSV files,
        creates a combined CSV with one row per embedding, ordered by
        the filename number prefix.

        Args:
            input_folder: Folder containing subfolders with individual CSVs.
            output_folder: Folder for combined CSV output.
            generate_labels: If True, write labels.txt mapping row/column indices.
            generate_identity_gt: If True, generate identity ground-truth matrix.

        Returns:
            Dict mapping subfolder names to output file paths.
        """
        output_folder.mkdir(parents=True, exist_ok=True)
        results = {}
        ordered_names = []

        for subfolder in sorted(input_folder.iterdir()):
            if not subfolder.is_dir():
                continue

            csv_files = sorted(
                subfolder.glob("*.csv"),
                key=lambda p: extract_index(p.name),
            )

            if not csv_files:
                continue

            embeddings = []
            for csv_file in csv_files:
                embedding = np.loadtxt(csv_file, delimiter=",")
                if embedding.ndim == 1:
                    embedding = embedding.reshape(1, -1)
                embeddings.append(embedding)

            combined = np.vstack(embeddings)
            output_path = output_folder / f"{subfolder.name}.csv"
            np.savetxt(output_path, combined, delimiter=",")
            results[subfolder.name] = output_path
            ordered_names.append(subfolder.name)

        # Write labels file for ground-truth matrix reference
        if generate_labels and ordered_names:
            labels_path = output_folder / "labels.txt"
            with open(labels_path, "w") as f:
                f.write("# Ground-Truth Matrix Labels\n")
                f.write("# Row/Column Index -> Subfolder Name\n")
                f.write("#\n")
                for idx, name in enumerate(ordered_names):
                    f.write(f"{idx},{name}\n")

        # Generate identity ground-truth matrix (1.0 on diagonal, 0.0 elsewhere)
        if generate_identity_gt and ordered_names:
            n = len(ordered_names)
            identity_gt = np.eye(n)
            gt_path = output_folder / "ground_truth_identity.csv"
            np.savetxt(gt_path, identity_gt, delimiter=",")

        return results

    def load_combined_embeddings(
        self,
        folder: Path,
    ) -> dict[str, np.ndarray]:
        """Load all combined embedding files from a folder.

        Args:
            folder: Folder containing combined CSV files.

        Returns:
            Dict mapping subfolder names to embedding arrays (N x D).
        """
        embeddings = {}
        for csv_file in sorted(folder.glob("*.csv")):
            name = csv_file.stem
            data = np.loadtxt(csv_file, delimiter=",")
            if data.ndim == 1:
                data = data.reshape(1, -1)
            embeddings[name] = data
        return embeddings

    def get_subfolder_count(self, folder: Path) -> int:
        """Count number of combined embedding files."""
        return len(list(folder.glob("*.csv")))

    def get_embedding_info(self, folder: Path) -> dict:
        """Get information about embeddings in folder.

        Returns:
            Dict with 'n_subfolders', 'n_images_per_subfolder', 'embedding_dim'.
        """
        csv_files = list(folder.glob("*.csv"))
        if not csv_files:
            return {"n_subfolders": 0, "n_images_per_subfolder": 0, "embedding_dim": 0}

        first_file = csv_files[0]
        data = np.loadtxt(first_file, delimiter=",")
        if data.ndim == 1:
            data = data.reshape(1, -1)

        return {
            "n_subfolders": len(csv_files),
            "n_images_per_subfolder": data.shape[0],
            "embedding_dim": data.shape[1],
        }
