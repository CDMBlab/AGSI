# AGSI: Single-cell Multi-omics Integration

A deep learning framework for integrating unpaired single-cell RNA-seq and ATAC-seq data.

## Installation

```bash
git clone https://github.com/CDMBlab/AGSI.git
cd AGSI

pip install torch scanpy scikit-learn POT scipy numpy --break-system-packages
```

**Requirements:** Python 3.8+, PyTorch 1.12+

## Quick Start

```bash
python main.py \
    --data_path ./data/ \
    --source_data rna.h5ad \
    --target_data atac.h5ad \
    --source_preprocess Standard \
    --target_preprocess TFIDF
```

## Usage

### Basic Command

```bash
python main.py \
    --data_path <path_to_data> \
    --source_data <rna_file.h5ad> \
    --target_data <atac_file.h5ad> \
    --batch_size 512 \
    --train_epoch 20
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data_path` | - | Directory containing data files |
| `--source_data` | - | RNA-seq filename (.h5ad) |
| `--target_data` | - | ATAC-seq filename (.h5ad) |
| `--source_preprocess` | Standard | RNA preprocessing: Standard/TFIDF |
| `--target_preprocess` | TFIDF | ATAC preprocessing: Standard/TFIDF |
| `--batch_size` | 512 | Training batch size |
| `--train_epoch` | 20 | Epochs per iteration |
| `--max_iteration` | 20 | Maximum iterations |
| `--reliability_threshold` | 0.95 | Base reliability threshold |
| `--adaptive_threshold` | False | Enable adaptive thresholding |
| `--n_components` | 15 | Number of gene modules (LDA topics) |
| `--wasserstein_alpha` | 0.5 | Weight for cosine similarity |
| `--umap_plot` | False | Generate UMAP visualizations |

## Input Format

Data files should be in `.h5ad` (AnnData) format:

**RNA-seq (source):**
- Expression matrix in `adata.X`
- Cell type labels in `adata.obs['CellType']` (required)

**ATAC-seq (target):**
- Gene activity matrix in `adata.X`
- Optional: Cell type labels in `adata.obs['CellType']` (for evaluation)

## Output

The method saves integrated results to:
- `{data_path}/{source_data}-integrated.h5ad`
- `{data_path}/{target_data}-integrated.h5ad`

Each output file contains:
- `adata.obsm['Embedding']`: Integrated embeddings
- `adata.obs['Prediction']`: Predicted cell types (target only)
- `adata.obs['Reliability']`: Reliability scores (target only)
- `adata.obsm['X_umap']`: UMAP coordinates (if `--umap_plot` enabled)

UMAP plots are saved to `figures/` directory.

## Example

```python
import scanpy as sc

# Load results
target_adata = sc.read_h5ad('data/atac-integrated.h5ad')

# Access predictions
predictions = target_adata.obs['Prediction']
reliability = target_adata.obs['Reliability']
embeddings = target_adata.obsm['Embedding']

# Visualization
sc.pl.umap(target_adata, color=['Prediction', 'Reliability'])
```

## Project Structure

```
AGSI/
├── main.py           # Main entry point
├── model_utils.py    # Neural network models
├── data_utils.py     # Data loading and preprocessing
└── eval_utils.py     # Evaluation and visualization
```
