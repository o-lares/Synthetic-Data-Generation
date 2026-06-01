# Synthetic Data Generation

A hands-on experiment exploring synthetic data generation methods and evaluating their impact on machine learning models trained for real-world applications.

## Purpose

This project is a learning-focused exploration into how synthetic data can be generated from real-world datasets and whether models trained on (or augmented with) synthetic data can perform comparably to those trained on original data alone. The goal is to build intuition around the strengths, limitations, and practical trade-offs of various synthetic data generation techniques.

## Dataset

This project uses the [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing), which contains data from direct marketing campaigns of a Portuguese banking institution. The classification task is to predict whether a client will subscribe to a term deposit.

The dataset is heavily imbalanced: ~88% "no" vs ~12% "yes".

## Experiments

A Random Forest classifier is trained under four different strategies for handling the dataset's class imbalance:

1. **Baseline** — vanilla Random Forest with no imbalance correction
2. **Class Weight Balancing** — using scikit-learn's `class_weight='balanced'` to penalize misclassifying the minority class
3. **SMOTE (Synthetic Minority Oversampling)** — generating synthetic minority class samples in feature space using linear interpolation (`imbalanced-learn`)
4. **CTGAN (Conditional Tabular GAN)** — training a GAN-based generative model on the minority class to produce realistic synthetic samples

Each approach is evaluated on the same held-out test set using precision, recall, and F1-score.

### Results (Minority Class — "yes" subscribers)

| Strategy | Precision | Recall | F1-score |
|---|---|---|---|
| Baseline | 0.68 | 0.29 | 0.40 |
| Class Weight Balanced | 0.43 | 0.84 | 0.57 |
| SMOTE | 0.54 | 0.59 | 0.57 |
| **CTGAN** | **0.54** | **0.53** | **0.54** |

### CTGAN Configuration

```python
ctgan = CTGAN(
    epochs=400,
    batch_size=250,
    generator_dim=(128, 128),
    discriminator_dim=(128, 128),
    verbose=True
)
```

CTGAN was trained exclusively on the minority class samples from the training set. Synthetic samples were generated to bring the class distribution to 50/50 before training the Random Forest.

## Getting Started

### Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/) — a fast Python package and project manager

### Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/o-lares/Synthetic-Data-Generation.git
   cd Synthetic-Data-Generation
   ```

2. **Install dependencies**

   `uv` will automatically create a virtual environment and install everything defined in `pyproject.toml`:

   ```bash
   uv sync
   ```

3. **Run the notebook**

   ```bash
   uv run jupyter notebook synth_notebook.ipynb
   ```

   Or if you're using VS Code, just open `synth_notebook.ipynb` and select the `.venv` Python environment as your kernel.

## Tech Stack

- **Python 3.12**
- **pandas** / **NumPy** — data manipulation
- **scikit-learn** — modeling and evaluation
- **imbalanced-learn** — SMOTE synthetic oversampling
- **CTGAN** — GAN-based tabular data synthesis
- **PyTorch** — CTGAN backend
- **matplotlib** / **seaborn** — visualization
- **ucimlrepo** — UCI dataset fetching
- **uv** — dependency management
