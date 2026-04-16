# Spaced-Replay Continual Finetuning

## Research Goal
Develop and evaluate spacing-inspired replay policies that reduce catastrophic forgetting during continual finetuning under limited memory and compute.

## Research Question
Can sample-level forgetting signals, such as uncertainty, loss history, gradient norms, and representation drift, predict when examples should be replayed better than random or fixed replay schedules?

## Data Needed
- Split CIFAR-100 for the primary continual vision benchmark.
- Split CUB for a harder fine-grained vision benchmark.
- Optional: sentiment/domain adaptation datasets for DistilBERT experiments if time permits.

## Folder Structure
- `data/`: raw and processed datasets, dataset download scripts
- `notebooks/`: experiments and analysis notebooks
- `src/`: core code (models, training loops, utils)
- `experiments/`: experiment configurations and logs
- `models/`: saved model checkpoints
- `tests/`: unit and integration tests
- `scripts/`: helper scripts (evaluation, preprocessing)
- `docs/`: documentation and notes

## Step-by-step Plan
1. Prepare sequential Split CIFAR-100 and Split CUB data loaders.
2. Implement continual finetuning baselines: fine-tuning, experience replay, MIR, ESMER, and uncertainty-based replay.
3. Log sample-level forgetting signals: loss trajectories, uncertainty, gradient norms, and representation drift.
4. Implement the spaced-replay scheduler and buffer management policy.
5. Report final accuracy, average forgetting, training cost, buffer utilization, predictor precision-recall AUC, and scheduler ablations.

## Windows Support
This repository supports a Windows-only workflow. Use Windows PowerShell with Python 3.14 or newer available on `PATH`; CUDA is optional, and the code should fall back to CPU when no GPU backend is available. The original proposal mentions Apple Metal/MPS hardware, but this implementation is intended to run on Windows as well.

## Quick setup
Use the included env setup scripts to create a virtual environment and install dependencies.

Windows PowerShell:
```powershell
.\setup_env.ps1
```

Unix/macOS:
```bash
./setup_env.sh
```

Files of interest:
- [requirements.txt](requirements.txt)
- [setup_env.ps1](setup_env.ps1)
- [setup_env.sh](setup_env.sh)
- [src/main.py](src/main.py)
- [docs/ACTION_PLAN.md](docs/ACTION_PLAN.md)
