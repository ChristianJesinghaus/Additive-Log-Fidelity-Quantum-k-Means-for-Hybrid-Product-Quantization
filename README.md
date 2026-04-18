# Artifact for “Additive Log-Fidelity Quantum k-Means for Hybrid Product Quantization”

This repository accompanies the short NIER submission on additive log-fidelity quantum k-means for hybrid product quantization. It reproduces the paper’s main experiments on Digits, downsampled Fashion-MNIST 8x8, and Signed-Mirror-64, and it contains the scripts used to generate the paper tables, plots, and summary reports.

## Reproducing the paper

### 1) Environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

After confirming a clean run in a fresh environment, save an exact lock file for the submission artifact:

```bash
pip freeze > requirements-lock.txt
```

### 2) Generate the datasets used in the paper

```bash
python create_digits_full_npz.py
python create_fashion8x8_npz.py
python create_signed_mirror64_npz.py
```

These scripts create the following files under `datasets/`:

- `digits64_full.npz`
- `fashion_mnist_8x8_full.npz`
- `signed_mirror64_full.npz`

### 3) Run the paper experiments

Run all paper plans:

```bash
python run_paper_experiments.py --plan all --datasets-dir datasets --output-dir experiments/paper_runs
```

Or run only the main comparison:

```bash
python run_paper_experiments.py --plan all_main --datasets-dir datasets --output-dir experiments/paper_runs
```

Save per-partition training histories as well:

```bash
python run_paper_experiments.py --plan all --datasets-dir datasets --output-dir experiments/paper_runs --save-histories
```

### 4) Build the paper report

```bash
python make_paper_report.py --input-dir experiments/paper_runs --output-dir experiments/paper_report
```

The generated report folder contains aggregated tables, plots, and helper text files for the manuscript.

## Variant names used in the code and in the paper

The experiment runner uses compact internal names:

- `exact_knn` -> exact kNN
- `classical` -> classical PQ
- `quantum_exact` -> exact-overlap reference
- `quantum_shot` -> shot-based swap-test estimator

In the paper, the last two names are used to avoid the misleading impression that the `quantum_exact` path is a hardware-based quantum run. It is the same fidelity-based assignment rule with exact overlaps computed classically from normalized block vectors.

## Demo path (not the paper reproduction path)

The repository still includes `hybrid_quantum_example.py` and `config.txt` for a small demo-style run on a split `example_data.npz` file. That path is useful for quick smoke tests, but it is not the recommended route for reproducing the paper tables.

Example:

```bash
python create_digits_npz.py
python hybrid_quantum_example.py
```

## Optional inspection helpers

Test a saved model:

```bash
python test_saved_model.py
```

Create a confusion matrix for a saved classical run:

```bash
python classical_confusion.py --model_dir experiments/models/<your_model_dir>
```

## Reproducibility notes

- The paper’s shared real-data configuration uses `shots = 2000`, `tau = 1e-2`, `epsilon = 1e-3`, `m = 8`, and `K = 10`.
- Sign-aware encoding is enabled only for the Signed-Mirror-64 study.
- The quantum kNN path now uses the same tie-break rule as the classical PQ baseline: majority vote first, then smallest summed approximate distance among tied labels.

## Citation

If you use this repository or its results, please cite the accompanying paper or the machine-readable `CITATION.cff` file included in the repository.
