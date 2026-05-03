# Synthetic Data Generation

A hands-on experiment exploring synthetic data generation methods and evaluating their impact on machine learning models trained for real-world applications.

## Purpose

This project is a learning-focused exploration into how synthetic data can be generated from real-world datasets and whether models trained on (or augmented with) synthetic data can perform comparably to those trained on original data alone. The goal is to build intuition around the strengths, limitations, and practical trade-offs of various synthetic data generation techniques.

## Dataset

This project uses the [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing), which contains data from direct marketing campaigns of a Portuguese banking institution. The classification task is to predict whether a client will subscribe to a term deposit.

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
- **matplotlib** / **seaborn** — visualization
- **ucimlrepo** — UCI dataset fetching
- **uv** — dependency management
