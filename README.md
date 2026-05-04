# Continual Learning Replay Experiments

This repository contains continual-learning replay experiments for Split
CIFAR-100 and sampled Split DBpedia14. The code compares fine-tuning, random
replay, spaced replay, risk-ranked replay, representation-drift replay, and
MIR-style replay under matched replay budgets.

The research results are summarized in [docs/RESULTS.md](docs/RESULTS.md).

## Repository Layout

- `src/`: data streams, metrics, replay buffers, baselines, predictors, and training code
- `scripts/`: experiment runners, diagnostics, and plotting scripts
- `configs/`: dataset, protocol, and experiment configuration files
- `tests/`: unit and integration tests
- `docs/figures/`: result figures used by the results summary

## Setup

Windows PowerShell:

```powershell
.\setup_env.ps1
```

Unix/macOS:

```bash
./setup_env.sh
```

## Verify

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Run Experiments

Run a smoke Split CIFAR-100 core comparison:

```powershell
.\.venv\Scripts\python.exe scripts\run_core_comparison.py --config configs\experiments\core_comparison_smoke.yaml
```

Run a full Split CIFAR-100 core comparison:

```powershell
.\.venv\Scripts\python.exe scripts\run_core_comparison.py --config configs\experiments\core_comparison_split_cifar100.yaml
```

Run the MIR replay comparison:

```powershell
.\.venv\Scripts\python.exe scripts\run_mir_replay_comparison.py --config configs\experiments\mir_replay_comparison_split_cifar100.yaml
```

Run one sampled Split DBpedia14 NLP seed with CUDA:

```powershell
$env:HF_HOME = ".tmp\hf-home"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:TOKENIZERS_PARALLELISM = "false"
.\.venv-nlp\Scripts\python.exe scripts\run_nlp_continual_pilot.py --config configs\experiments\nlp_dbpedia14_speed_pilot.yaml --seed 0
```

## Regenerate Figures

```powershell
.\.venv\Scripts\python.exe scripts\plot_final_synthesis.py
.\.venv\Scripts\python.exe scripts\plot_nlp_three_seed_results.py --root experiments\runs\nlp_continual_pilot --output docs\figures\nlp_three_seed_main_results.png
.\.venv\Scripts\python.exe scripts\plot_proposal_signal_comparison.py --output docs\figures\cifar100_signal_family_comparison.png
```

Generated experiment artifacts are written under `experiments/runs/`.
