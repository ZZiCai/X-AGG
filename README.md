# X-AGG: Explainability-Aligned Attributed Graph Generation

Official implementation of **"Explainability-Aligned Attributed Graph Generation"** (submitted to IEEE Transactions on Knowledge and Data Engineering).

## Overview

X-AGG is a novel framework for generating attributed graphs that preserves not only structural and attribute distributions but also **explainability alignment** with the original graph. Unlike existing graph generation methods that focus solely on statistical properties, X-AGG ensures that the generated graphs maintain consistent feature importance patterns and edge explanations, making them suitable for downstream tasks requiring interpretability.

### Key Features

- **Explainability-Aligned Generation**: Preserves feature importance rankings and edge explanation patterns
- **Conditional Variational Graph Autoencoder (CVGAE)**: Leverages both structural and attribute conditioning
- **Comprehensive Evaluation**: Includes structural metrics (degree, clustering), attribute metrics (MMD, EMD, JS), and explainability metrics (Spearman correlation, Jaccard@K)
- **Multiple Dataset Support**: Tested on Cora, Pubmed, Flickr, Photo, Computers and CS

## Project Structure

```
X-AGG/
├── main.py                    # Main training script
├── utils.py                   # Utility functions (argument parsing, logging, metrics)
├── config/
│   └── config.yaml           # Configuration file
├── modules/
│   ├── generator.py          # CVGAE model implementation
│   ├── graphEncoder.py       # Graph encoder architectures
│   └── preprocessor.py       # Data preprocessing and conditioning matrix generation
├── graph_metrics/
│   ├── __init__.py
│   ├── comp_tools.py         # Comparison tools for graph statistics
│   ├── eval_attribute.py     # Attribute distribution evaluation
│   ├── metrics.py            # Core metric implementations
│   └── stat_tools.py         # Statistical analysis tools
├── data/                      # Dataset directory
├── logs/                      # Training logs
├── models/                    # Saved models
└── graphs/                    # Generated graphs
```



## Installation

```bash
# Clone the repository
git clone https://github.com/ZZiCai/X-AGG.git
cd X-AGG

# Create conda environment
conda create -n xagg python=3.11 -y
conda activate xagg

# Install dependencies
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121

pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.1+cu121.html
pip install matplotlib pandas powerlaw scikit-learn networkx numpy

```

## Quick Start

### Basic Usage

Run the model with default configuration (Cora dataset):

```bash
python main.py
```

### Configuration

The model is configured through `config/config.yaml`. Key parameters include:

```yaml
data: Cora                    # Dataset name
num_epoch: 200                # Training epochs
batch_size: 64                # Batch size
lr: 1e-3                      # Learning rate
weight_decay: 0.0001          # Weight decay
hidden_size: 512              # Hidden dimension size
device: 'cuda:0'              # Device (cuda:0 or cpu)
seed: 2025                    # Random seed
eval_per_epochs: 5            # Evaluation frequency
save_model_intervals: 20      # Model saving frequency
num_neighbors: [-1]           # Neighbor sampling (-1 for all)
experiment_name: ''           # Experiment identifier
ablation: ''                  # Ablation study mode:no-M_A, no-M_X, no-M
```

### Command-Line Parameter Updates

You can override configuration parameters via command line:

```bash
# Update single parameter
python main.py --update data=Pubmed

# Update multiple parameters
python main.py --update data=CS num_epoch=300 lr=0.001 batch_size=128

# Run ablation studies
python main.py --update ablation=no-M_A experiment_name=ablation_no_struct
python main.py --update ablation=no-M_X experiment_name=ablation_no_attr
python main.py --update ablation=no-M experiment_name=ablation_no_cond
```

### Supported Datasets

The framework supports the following datasets:

| Dataset | Type | Nodes | Edges | Features | Classes |
|---------|------|-------|-------|----------|---------|
| Cora | Citation | 2,708 | 10,556 | 1,433 | 7 |
| Pubmed | Citation | 19,717 | 88,648 | 500 | 3 |
| Flickr    | Social        | 7,575  | 479,476 | 12,047   | 9       |
| Photo     | Co-purchase   | 7,650  | 238,162 | 745      | 8       |
| Computers | Co-purchase | 13,752 | 491,722 | 767 | 10 |
| CS        | Co-authorship | 18,333 | 163,788 | 6,805    | 15      |

## Output Files

The framework generates the following outputs:

```
logs/
  └── {dataset}-{experiment_name}-{timestamp}.log    # Training logs

models/
  └── {dataset}-{experiment_name}-{timestamp}.pth    # Trained model

graphs/
  ├── {dataset}-{experiment_name}-{timestamp}_gen_adj.pkl   # Generated adjacency matrix
  └── {dataset}-{experiment_name}-{timestamp}_gen_attr.pkl  # Generated node attributes

data/{dataset}/
  ├── cond_feat.npy    # Preprocessed feature conditioning matrix
  └── cond_struct.npy  # Preprocessed structural conditioning matrix
```

