# PLCFedFS

Implementation of **Uncertainty-Coupled Latent-Environment Feature Selection for Non-IID Federated Data under Differential Privacy**.

PLCFedFS discovers cross-fitted soft contexts, releases jointly clipped private feature statistics, combines context evidence with equal total client mass, and selects a raw feature subset using lower-confidence utility and noise-aware redundancy penalties.

## Install

Use Python 3.11+ in an activated virtual environment:

```bash
python -m pip install .
```

## Prepare federated data

Use [Gas Sensor Array Drift](https://archive.ics.uci.edu/dataset/224/gas+sensor+array+drift+dataset) or the Arcene, Madelon and Gisette feature-selection benchmark distributions. Preserve only records whose labels are publicly available. Gas chronological batches can define clients and shorter windows proxy contexts. For challenge datasets, register a client partition and any controlled nuisance transformations before evaluating feature recovery.

Create a CSV with `label,client,split` and numeric feature columns. Labels are contiguous zero-based IDs; `split` is `train`, `val`, or `test`. All records of a client must remain in one split. Optional `context` labels and stable feature names supplied through `--anchors` are evaluator-only annotations.

```bash
python prepare.py --csv data/records.csv --features f0 f1 f2 f3 --anchors f0 f1 --output data/federation.npz
```

List every feature column for real datasets. The resulting numeric archive contains `x[N,P]`, `y[N]`, `client[N]`, `split[N]`, optional `context[N]`, and optional zero-based stable-feature indices `anchors`. Missing values must be imputed using training-only settings before conversion. `--bound 5` publicly clips each covariate to `[-5,5]` and rescales it; it does not estimate private bounds from test records. Retain a public fixed feature schema across clients.

Use `val` clients to select feature-score and context thresholds by running a separate development archive that designates those clients as its evaluation split. Freeze settings, then use unseen `test` clients in the final archive. Test clients never participate in discovery or selection.

## Select features and evaluate transfer

```bash
plcfedfs --data data/federation.npz --output outputs/full --features 40 --candidates 80 --contexts 3 --folds 3 --epsilon 4 --delta 0.00001 --repeats 5
```

Set `--features` no larger than the private candidate union. The short four-column conversion example therefore requires `--features 2 --candidates 4`; the default 40-feature setting targets the full benchmarks.

Each client privately screens candidate features. Their union is used for subsequent relevance and redundancy statistics. Context discovery fits a private local classifier and label-stratified diagonal mixtures on training folds, predicts only held-out folds, and repeats the process twice. Component alignment uses already-private centers. Private mass and reliability pilot statistics control context merging and degeneration to a client-only representation.

Joint per-record vectors contain weighted class/feature cross-statistics, mass and second moments. The server estimates one-vs-rest point-biserial relevance, sampling/assignment/privacy uncertainty, sign support, and within/across-client dispersion. Context weights sum to one within each client and every client gets total mass `1/E`. Greedy selection subtracts reliable pairwise redundancy from lower-confidence feature utility.

Important settings:

| Parameter | Meaning |
| --- | --- |
| `--minimum-context 20` | Minimum privately estimated effective context size |
| `--reliability 0.1` | Minimum aggregate private context reliability |
| `--predictor-steps`, `--em-steps` | Private discovery update counts |
| `--projection 4` | Data-independent covariate projection dimension |
| `--sign-weight`, `--client-penalty`, `--latent-penalty` | Hierarchical stability-score coefficients |
| `--confidence 1.64` | Lower-confidence uncertainty margin |
| `--redundancy 0.1` | Pairwise suppression strength |

## Baselines and ablations

```bash
plcfedfs --data data/federation.npz --output outputs/client-only --variant client_only
plcfedfs --data data/federation.npz --output outputs/no-uncertainty --variant no_uncertainty
plcfedfs --data data/federation.npz --output outputs/no-noise-aware --variant no_noise_aware
plcfedfs --data data/federation.npz --output outputs/no-redundancy --variant no_redundancy
```

The private client-only selector uses the same screening, fusion and downstream logistic classifier with one context per client. Other variants remove the confidence margin, noise-aware pairwise factor, or redundancy penalty. Run identical client partitions and paired seeds. Sweep privacy budget, context count, minimum size, feature budget and controlled shift strength using separate output directories.

## Metrics and privacy accounting

`results.json` reports unseen-client macro-F1, balanced accuracy, worst-client macro-F1, subset Jaccard stability across repeats, stable-anchor precision/recall/F1 when available, context ARI/NMI when available, transmitted summary bytes, client runtime, server selection runtime, and the full stage privacy ledger. `feature_scores.png` plots the first run's lower-confidence scores. A single repetition has no across-repeat Jaccard estimate and reports `null`.

The accountant uses Gaussian RDP for replace-one adjacency within a fixed client and public fold. Joint vectors are L2 clipped before Gaussian release with sensitivity twice the clipping radius. All discovery training, reliability pilots, candidate screening, relevance and redundancy releases are charged. Repeated selections compose in the same client ledger; disjoint clients use parallel composition. The configured epsilon is a bound for the complete repeated selector release, with a conservative release count. Evaluation labels, transfer-classifier fitting and recovery/attack metrics are benchmark analysis outside the private selector release interface.

For a latent-context membership attack, supply a separately prepared shadow/audit archive with `x` containing only released summary-derived attacker features, binary `member` labels and `split` values `train`/`test`:

```bash
plcfedfs --data data/federation.npz --output outputs/attack --attack data/attack.npz
```

Construct attacker rows from independent shadow releases using summary statistics, private effective-size bins, and selected-feature indicators. Label whether the target belongs to the registered small latent context. Keep target record identities disjoint between attacker fitting and audit; balance membership labels in each partition. The evaluator fits a balanced logistic attacker and reports AUC, precision, recall, and advantage `2*abs(AUC-0.5)`. No privacy attack score is synthesized when this evidence is unavailable.

## Across-dataset statistics

Build a CSV `block,method,value` from actual runs. A block is an independent dataset/shift condition, and repeated seeds within a block are averaged before comparison.

```bash
python statistical_analysis.py --csv outputs/comparison.csv --reference full --output outputs/statistics.json
```

The script reports Friedman statistics, exact signed-rank sign-enumeration comparisons, monotone Holm-adjusted p values, wins, paired median differences, and bootstrap confidence intervals. Exact enumeration supports at most 20 nonzero blocks; the paper's 12-block design fits this limit. Duplicate seeds do not create additional independent blocks.

## Implementation choices

The selector uses conservative fixed Gaussian release allocation rather than adaptive budget allocation. Discovery uses a compact random-projection/residual/confidence/gradient signature and a bounded number of noisy sufficient-statistic updates. Relevance uncertainty is a first-order variance proxy, not a separate exact confidence theorem for clipped noisy ratios. Redundancy sketches are second-moment matrices over the private candidate union and use public feature scaling. The explicit release ledger, degeneration rule and variance decomposition remain part of the selection mechanism.
