# Embedding Explorer

Compare image embeddings of multiple components.

---

## 🛠️ Installation

### Prerequisites
- Python 3.10 or higher
- Conda or Mamba (recommended)

### Full Installation
This installs all components including the GUI, optimization engine, and visualizers.

```bash
# Create and activate environment
conda create -n surrogate python=3.10 -y
conda activate surrogate

# Install the project and dependencies
pip install -e .
```

---

## 📂 Project Structure

```text
surrogate/
├── src/embedding_pipeline/
├── src/embedding_pipeline/
│   ├── main.py              # CLI Entry point
│   ├── core/                # Inference, Folder Processing, Combining
│   ├── config/              # YAML-based configuration management
│   ├── similarity/          # Pluggable similarity strategies (Sum, Concat, SVM)
│   ├── optimization/        # PSO weight optimization and SVM training
│   ├── visualization/       # t-SNE reduction logic
│   └── gui/                 # FastAPI + Tailwind/Plotly frontend
├── models/                  # Isolated model cache (Auto-created)
├── README.md                # This file
└── pyproject.toml           # Project metadata and dependencies
```

---

## 💻 Usage

The application uses a sub-command structure: `python -m src.embedding_pipeline [command] [args]`.

### 1. Compute Embeddings
Generates individual `.csv` embeddings for every image in a folder structure.

```bash
python -m src.embedding_pipeline compute-embeddings -i ./input_images -o ./embeddings --open-clip
```

### 2. Combine Embeddings
Aggregates individual image embeddings into one `.csv` file per object subfolder.

```bash
python -m src.embedding_pipeline combine-embeddings -i ./embeddings -o ./combined
```

### 3. Compute Similarity
Calculates an NxN similarity matrix. Supports weighted sum, concatenation, and SVM.

```bash
# Using SVM
python -m src.embedding_pipeline compute-similarity -i ./combined -o matrix.csv --method svm --model-path ./model.pkl
```

### 4. Train SVM
Trains an SVR model to predict similarity from multi-view matching data.

```bash
python -m src.embedding_pipeline train-svm -e ./combined -g ground_truth.csv -o model.pkl -l labels.txt
```

### 5. Sum Embeddings
Sums corresponding embedding vectors from two parallel folder structures.

```bash
python -m src.embedding_pipeline sum-embeddings --input-a ./f1 --input-b ./f2 -o ./summed_output
```

### 6. Optimize Weights (PSO)
Automatically finds the best perspective weights to match a ground truth similarity matrix.

```bash
# With labels file for alignment
python -m src.embedding_pipeline optimize-weights -e ./combined -g ground_truth.csv -o weights.csv -l labels.txt
```

### 7. Launch GUI
Starts the interactive web dashboard.

```bash
python -m src.embedding_pipeline gui --dark-mode
```

---

## 📋 CLI Argument Reference

### `compute-embeddings`
| Argument | Flag | Description |
| :--- | :--- | :--- |
| `--input` | `-i` | Root folder containing object subfolders. |
| `--output` | `-o` | Target folder for .csv embeddings. |
| `--open-clip` | | Use OpenCLIP (default). |
| `--dino-v2` | | Use DINO v2. |

### `compute-similarity`
| Argument | Flag | Description |
| :--- | :--- | :--- |
| `--input` | `-i` | Folder containing combined CSVs. |
| `--output` | `-o` | Output file path for the matrix. |
| `--method` | | `weighted_sum`, `concatenation`, or `svm`. |
| `--weights` | | space-separated list of float weights. |
| `--model-path` | | Path to pre-trained SVM model (required for `svm` method). |

### `train-svm`
| Argument | Flag | Description |
| :--- | :--- | :--- |
| `--embeddings` | `-e` | Folder containing combined CSVs. |
| `--ground-truth`| `-g` | Path to ground truth similarity matrix. |
| `--output` | `-o` | Output path for `.pkl` model. |
| `--gt-labels` | `-l` | TxT file with comma-separated subfolder labels for GT matrix order. |
| `--kernel` | | SVM kernel (default: `rbf`). |

### `sum-embeddings`
| Argument | Flag | Description |
| :--- | :--- | :--- |
| `--input-a` | | First input folder of embeddings. |
| `--input-b` | | Second input folder of embeddings. |
| `--output` | `-o` | Output folder for summed embeddings. |

### `optimize-weights`
| Argument | Flag | Description |
| :--- | :--- | :--- |
| `--embeddings` | `-e` | Folder containing combined CSVs. |
| `--ground-truth`| `-g` | Path to NxN ground truth similarity matrix. |
| `--output` | `-o` | Output path for **normalized** weights CSV. |
| `--gt-labels` | `-l` | TxT file with comma-separated subfolder labels for GT matrix order. |

---

## ⚙️ Configuration

`src/embedding_pipeline/config/default_config.yaml`.

### Custom Configuration
You can provide a custom configuration file using the `--config` flag:

```bash
python -m src.embedding_pipeline compute-embeddings -i ./input -o ./output --config my_config.yaml
```