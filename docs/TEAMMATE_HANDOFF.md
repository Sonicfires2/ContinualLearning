# Teammate Handoff

This is the quickest way to understand the repository, rerun the current work,
and continue the next research task.

## Current Status

The codebase is ready for a teammate to continue the Split CIFAR-100 continual
learning study. The main infrastructure is implemented:

- Split CIFAR-100 task streams with stable sample IDs.
- Shared continual training, seen-task evaluation, metrics, and artifact
  logging.
- Baselines for fine-tuning, random replay, fixed-periodic replay, spaced
  replay, MIR replay, risk-gated replay, learned-risk replay, hybrid replay,
  and gradient-signal diagnostics.
- Offline forgetting labels, time-to-forgetting targets, learned predictors,
  signal ablations, and result documents.
- Unit tests for the core modules and smoke tests for the major methods.

The current scientific result is mixed but mostly negative for the proposed
intervention:

```text
Replay helps.
MIR helps more.
Forgetting can be predicted offline.
The tested learned-risk replay policies do not yet beat random replay online.
```

That is still useful. The project is no longer missing the basic experiment
scaffold; it is now at a decision point about whether to stop with a clean
negative result or pivot toward a more MIR-like interference signal.

## How Close Are We To The Result?

If the goal is a defensible class/project research result, the repository is
close. It already supports the main conclusion that the current
spacing-inspired and risk-guided replay variants do not beat random replay on
the locked Split CIFAR-100 setup.

If the goal is a new method that clearly improves over random replay and MIR,
the repository is not close yet. The current evidence says the learned predictor
is good at forecasting forgetting in saved logs, but that forecast has not
turned into better replay choices during training.

The immediate next task should therefore be small and diagnostic, not a big new
benchmark.

## Key Results So Far

Three-seed core results:

| Method | Final accuracy | Avg forgetting |
| --- | ---: | ---: |
| fine-tuning | `0.0470` | `0.4310` |
| random replay | `0.1016` | `0.3027` |
| spaced replay proxy | `0.0986` | `0.3131` |
| MIR replay | `0.11636666666666667` | `0.2167037037037037` |

Seed-0 learned replay and diagnostic results:

| Method | Final accuracy | Avg forgetting | Replay samples |
| --- | ---: | ---: | ---: |
| random replay | `0.10129999999999999` | `0.30433333333333334` | `45216` |
| learned risk-gated, threshold `0.90` | `0.0379` | `0.40311111111111114` | `14425` |
| learned fixed-budget | `0.0759` | `0.3587777777777778` | `45216` |
| learned hybrid, 50/50 class-balanced | `0.0879` | `0.3428888888888889` | `45216` |
| MIR replay | `0.1183` | `0.21400000000000002` | `45216` |

Predictor diagnostics:

| Feature group | Average precision |
| --- | ---: |
| cheap all features | `0.9083240127221096` |
| cheap plus gradient | `0.9080918805327551` |
| gradient only | `0.8386703932509996` |
| best cheap heuristic | `0.8471253916174554` |

Plain-English interpretation:

```text
The predictor can often tell which examples are likely to be forgotten later.
But replay needs to pick examples that protect the model right now.
Those are related questions, but they are not the same question.
MIR is stronger because it asks which old examples the current update would hurt.
```

## Start Here

Read these in order:

1. [README.md](../README.md) for the project overview and setup.
2. [ACTION_PLAN.md](./ACTION_PLAN.md) for the research sequence and current
   next task.
3. [RESULTS_ANALYSIS_RETROSPECTIVE.md](./RESULTS_ANALYSIS_RETROSPECTIVE.md)
   for the simple explanation of what the results mean.
4. [PROGRESS_LOG.md](./PROGRESS_LOG.md) for the chronological implementation
   record.
5. [GRADIENT_SIGNAL_DIAGNOSTIC_TASK21.md](./GRADIENT_SIGNAL_DIAGNOSTIC_TASK21.md)
   for the most recent completed task.

## Setup

Windows PowerShell:

```powershell
.\setup_env.ps1
.\.venv\Scripts\python.exe -m pytest -q
```

Expected current test result:

```text
87 passed
```

The full Split CIFAR-100 runs require CIFAR-100 data under `data/`. The checked
configs use `download: false` so full experiments do not silently download data.
Set `download: true` in a local copy only if you intentionally want torchvision
to fetch the dataset.

## Important Directories

| Path | Purpose |
| --- | --- |
| `src/data/` | Split CIFAR-100 task construction and stable sample metadata. |
| `src/training/` | Shared continual-training and evaluation loop. |
| `src/baselines/` | Runnable baseline and replay methods. |
| `src/replay/` | Replay buffer, spaced scheduler, and MIR selection helpers. |
| `src/signals/` | Sample-level and gradient signal extraction. |
| `src/predictors/` | Forgetting-risk, learned predictor, ablation, and diagnostic logic. |
| `src/metrics/` | Accuracy matrix and forgetting metrics. |
| `configs/experiments/` | Smoke and Split CIFAR-100 experiment configs. |
| `scripts/` | Analysis and comparison scripts. |
| `docs/` | Research plan, task notes, results, and retrospective. |

## Common Commands

Run all tests:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Run a smoke baseline:

```powershell
.\.venv\Scripts\python.exe -m src.baselines.random_replay --config configs\experiments\random_replay_smoke.yaml
```

Run the core Split CIFAR-100 comparison:

```powershell
.\.venv\Scripts\python.exe scripts\run_core_comparison.py --config configs\experiments\core_comparison_split_cifar100.yaml
```

Run the MIR comparison:

```powershell
.\.venv\Scripts\python.exe scripts\run_mir_replay_comparison.py --config configs\experiments\mir_replay_comparison_split_cifar100.yaml
```

Run the current learned hybrid method:

```powershell
.\.venv\Scripts\python.exe -m src.baselines.learned_hybrid_replay --config configs\experiments\learned_hybrid_replay_split_cifar100.yaml
```

Run the most recent gradient diagnostic:

```powershell
.\.venv\Scripts\python.exe -m src.baselines.gradient_signal_diagnostic --config configs\experiments\gradient_signal_diagnostic_split_cifar100.yaml
```

## Artifact Policy

Commit source code, tests, configs, and documentation. Do not commit generated
dataset files, checkpoints, logs, or full experiment run directories.

Important ignored outputs:

- `data/raw/`
- `data/processed/`
- `data/cifar-100-python/`
- `experiments/runs/`
- `experiments/*/`

The result documents in `docs/` preserve the important metrics, so the GitHub
repository does not need to carry multi-gigabyte generated artifacts.

## Remaining Tasks

The next task in [ACTION_PLAN.md](./ACTION_PLAN.md) is:

```text
22 - Run a decision checkpoint and small targeted rescue ablations before any
stretch benchmark
```

Recommended Task 22 implementation:

1. Run `learned_hybrid_replay_task22_frac025_class_balanced_split_cifar100.yaml`.
2. Run `learned_hybrid_replay_task22_frac025_random_split_cifar100.yaml`.
3. Optionally run
   `learned_hybrid_replay_task22_class_balanced_only_split_cifar100.yaml`.
4. Compare against random replay, learned fixed-budget replay, learned hybrid
   50/50, and MIR seed 0.
5. Write a short decision note:
   either stop with the clean negative result, or pivot to MIR-like
   current-interference or representation-drift diagnostics.

Task 22 command template:

```powershell
.\.venv\Scripts\python.exe -m src.baselines.learned_hybrid_replay --config configs\experiments\learned_hybrid_replay_task22_frac025_class_balanced_split_cifar100.yaml
```

Tasks after that:

| Task | Status |
| --- | --- |
| Task 22: targeted rescue ablations and decision checkpoint | next |
| Task 23: MIR-like interference or representation-drift diagnostics | optional if continuing method work |
| Task 24: Split CUB or DistilBERT stretch benchmark | blocked until Split CIFAR-100 conclusion is stable |
| Task 25: final synthesis, plots, tables, and written argument | final |

## GitHub Commit Checklist

Before pushing, check the working tree:

```powershell
git status --short
```

The initial repository tracked only a small skeleton, so many real project files
may appear as untracked. The intended commit should include:

- `.gitignore`
- `README.md`
- `requirements.txt`
- `setup_env.ps1`
- `setup_env.sh`
- `pytest.ini`
- `Project_Proposal-2.tex`
- `Project_Proposal-2.pdf`
- `configs/`
- `docs/`
- `scripts/`
- `src/`
- `tests/`
- `data/README.md`
- `notebooks/README.md`
- `experiments/README.md`

Do not include ignored data or experiment outputs.

Suggested commands after reviewing `git status`:

```powershell
git add .gitignore README.md requirements.txt setup_env.ps1 setup_env.sh pytest.ini
git add Project_Proposal-2.tex Project_Proposal-2.pdf
git add configs docs scripts src tests
git add data\README.md notebooks\README.md experiments\README.md
git status --short
git commit -m "Add continual learning replay research scaffold"
```
