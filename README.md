# PLCFedFS

Implementation of **Uncertainty-Coupled Latent-Environment Feature Selection for Non-IID Federated Data under Differential Privacy**.

## Installation

Use Python 3.11+. Create and activate a virtual environment before installing:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

In PowerShell, replace the activation line with `.venv\Scripts\Activate.ps1`.

## Data and preprocessing

Use [Gas Sensor Array Drift](https://archive.ics.uci.edu/dataset/224/gas+sensor+array+drift+dataset) or the Arcene, Madelon and Gisette feature-selection benchmark distributions. Preserve only records whose labels are publicly available. Gas chronological batches can define clients and shorter windows proxy contexts. For challenge datasets, register a client partition and any controlled nuisance transformations before evaluating feature recovery.

Create a CSV with `label,client,split` and numeric feature columns. Labels are contiguous zero-based IDs; `split` is `train`, `val`, or `test`. All records of a client must remain in one split. Optional `context` labels and stable feature names supplied through `--anchors` are evaluator-only annotations.

List every feature column for real datasets. The resulting numeric archive contains `x[N,P]`, `y[N]`, `client[N]`, `split[N]`, optional `context[N]`, and optional zero-based stable-feature indices `anchors`. Missing values must be imputed using training-only settings before conversion. `--bound 5` publicly clips each covariate to `[-5,5]` and rescales it; it does not estimate private bounds from test records. Retain a public fixed feature schema across clients.

Use `val` clients to select feature-score and context thresholds by running a separate development archive that designates those clients as its evaluation split. Freeze settings, then use unseen `test` clients in the final archive. Test clients never participate in discovery or selection.

```bash
python prepare.py --csv data/records.csv --features f0 f1 f2 f3 --anchors f0 f1 --output data/federation.npz
```

## Run and evaluate

```bash
plcfedfs --data data/federation.npz --output outputs/full --features 2 --candidates 4 --contexts 3 --folds 3 --epsilon 4 --delta 0.00001 --repeats 5 --seed 0
```

The example lists four raw feature columns. Replace the names and feature/candidate counts together for your actual panel.

## Ablations

Supported controls: `client_only`, `no_uncertainty`, `no_noise_aware`, `no_redundancy`. Reuse the same inputs, seed and budget. For retrained controls, evaluate the corresponding saved model.

```bash
plcfedfs --data data/federation.npz --output outputs/client_only --features 2 --candidates 4 --contexts 3 --folds 3 --epsilon 4 --delta 0.00001 --repeats 5 --seed 0 --variant client_only
```

## Outputs

`results.json` retains selected features, repeats, released-summary privacy accounting and computed transfer metrics; `feature_scores.png` plots the candidate scores.

Source inspection: Python 3.13.5, numpy 2.1.3, scipy 1.15.3, scikit-learn 1.6.1, matplotlib 3.10.0 (installed versions inspected). Checks covered Python syntax, configuration parsing and CLI arguments. No training, full evaluation or CI was run in this update.
