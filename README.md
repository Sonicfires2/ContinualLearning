# Spaced-Replay Continual Finetuning

## Start Here For Handoff
If you are picking up this repository, start with
[docs/TEAMMATE_HANDOFF.md](docs/TEAMMATE_HANDOFF.md). It explains the current
status, how close the project is to a final result, what has been implemented,
how to run the code, and what task should happen next.

## Research Goal
Develop and evaluate spacing-inspired replay policies that reduce catastrophic forgetting during continual finetuning under limited memory and compute.

## Research Question
Can sample-level forgetting signals, such as uncertainty, loss history, gradient norms, and representation drift, predict when examples should be replayed better than random or fixed replay schedules?

The proposal's scheduler is explicitly time-based: estimate a sample-specific forgetting time `T_i` and schedule replay near `t + T_i`, so the method tests long-term retention rather than only risk ranking.

## Data Needed
- Split CIFAR-100 is the locked primary benchmark for the core study.
- Split CUB is an optional extension after the core Split CIFAR-100 result is stable.
- Optional: sentiment/domain adaptation datasets for DistilBERT experiments only after the vision core is complete.

## Folder Structure
- `data/`: raw and processed datasets, dataset download scripts
- `notebooks/`: experiments and analysis notebooks
- `src/`: core code (models, training loops, utils)
- `experiments/`: experiment configurations and logs
- `models/`: saved model checkpoints
- `tests/`: unit and integration tests
- `scripts/`: helper scripts (evaluation, preprocessing)
- `docs/`: documentation and notes

## Locked Core Study
The current locked core protocol is:

1. Split CIFAR-100 is the required benchmark.
2. Fine-tuning, random replay, fixed-periodic replay, and spaced replay are the required comparison methods.
3. Loss history and predictive uncertainty are the required first signals.
4. The spaced scheduler must estimate or approximate sample-specific replay due times from forgetting-time predictions.
5. Final accuracy and average forgetting are the required first metrics.
6. MIR is implemented as the first stronger replay baseline; expensive signals, Split CUB, and DistilBERT remain later-stage additions.
7. Event-triggered risk-gated replay is implemented as the first compute-saving
   scheduler variant; its one-seed pilot saved replay samples but did not
   preserve retention.
8. Learned forgetting predictors are implemented for offline comparison:
   logistic regression and linear SVM classifiers improve average precision over
   the best cheap heuristic on the informative seed-0 replay artifacts, but they
   have not yet been validated as online replay gates.
9. Signal ablations show the full cheap feature set is best, with
   `history_summary` as the strongest compact group; this supports testing a
   learned predictor replay gate next.
10. Learned-predictor risk-gated replay is implemented and tested; the first
    seed-0 pilots are negative, showing that better offline prediction does not
    automatically improve online replay.
11. Fixed-budget learned-risk replay is implemented and tested. It uses the
    same seed-0 replay budget as random replay, but performs worse, which means
    pure learned-risk ranking is not yet operationally useful.
12. Balanced hybrid replay is implemented and tested. A 50/50 learned-risk plus
    class-balanced seed-0 pilot improves over pure learned-risk replay but still
    does not beat random replay or MIR.
13. Gradient-norm diagnostics are implemented. The final-layer gradient signal
    is measurable, but it does not improve forgetting prediction over the cheap
    feature set and adds meaningful runtime/storage cost.
14. The current retrospective conclusion is that forgetting can be predicted
    offline, but the tested risk-guided replay policies do not yet beat random
    replay as interventions.
15. Task 22 rescue ablations are complete. Pure class-balanced replay slightly
    beat random replay on seed 0, while lower learned-risk hybrids still did
    not clearly beat random replay. This points to diversity/class coverage as
    more useful than the current learned-risk selector.
16. Task 23 MIR-like interference diagnostics are complete. The learned
    future-forgetting score does not agree with MIR's current-interference
    ranking; its average precision for MIR top-k candidates is only `0.2160`
    against a `0.25` base rate.
17. Task 25 final synthesis is complete. The stable conclusion is that
    forgetting is predictable offline, but the tested spacing-inspired and
    learned-risk replay policies do not beat random replay online; MIR remains
    the strongest implemented replay method.
18. A speed-first NLP pivot plan is available for running a sampled
    DistilBERT continual-finetuning pilot on the local RTX GPU.
19. The first one-seed NLP result is complete on sampled Split DBpedia14:
    fine-tuning forgets badly, while random replay and spaced replay both
    preserve old text tasks; spaced replay does not beat random replay.

See:
- [docs/TEAMMATE_HANDOFF.md](docs/TEAMMATE_HANDOFF.md)
- [docs/CORE_EXPERIMENT_PROTOCOL.md](docs/CORE_EXPERIMENT_PROTOCOL.md)
- [docs/PROJECT_PROPOSAL_TEXT.md](docs/PROJECT_PROPOSAL_TEXT.md)
- [docs/DATASET_CIFAR100.md](docs/DATASET_CIFAR100.md)
- [docs/TIME_TO_FORGETTING_TARGETS.md](docs/TIME_TO_FORGETTING_TARGETS.md)
- [docs/SPACED_REPLAY_SCHEDULER.md](docs/SPACED_REPLAY_SCHEDULER.md)
- [docs/CORE_COMPARISON_TASK13.md](docs/CORE_COMPARISON_TASK13.md)
- [docs/MIR_REPLAY_BASELINE.md](docs/MIR_REPLAY_BASELINE.md)
- [docs/RISK_GATED_REPLAY_SCHEDULER.md](docs/RISK_GATED_REPLAY_SCHEDULER.md)
- [docs/LEARNED_FORGETTING_PREDICTORS.md](docs/LEARNED_FORGETTING_PREDICTORS.md)
- [docs/SIGNAL_ABLATIONS_TASK17.md](docs/SIGNAL_ABLATIONS_TASK17.md)
- [docs/LEARNED_RISK_GATED_REPLAY_TASK18.md](docs/LEARNED_RISK_GATED_REPLAY_TASK18.md)
- [docs/FIXED_BUDGET_LEARNED_REPLAY_PLAN.md](docs/FIXED_BUDGET_LEARNED_REPLAY_PLAN.md)
- [docs/LEARNED_FIXED_BUDGET_REPLAY_TASK19.md](docs/LEARNED_FIXED_BUDGET_REPLAY_TASK19.md)
- [docs/LEARNED_HYBRID_REPLAY_TASK20.md](docs/LEARNED_HYBRID_REPLAY_TASK20.md)
- [docs/GRADIENT_SIGNAL_DIAGNOSTIC_TASK21.md](docs/GRADIENT_SIGNAL_DIAGNOSTIC_TASK21.md)
- [docs/TASK22_DECISION_CHECKPOINT.md](docs/TASK22_DECISION_CHECKPOINT.md)
- [docs/MIR_INTERFERENCE_DIAGNOSTIC_TASK23.md](docs/MIR_INTERFERENCE_DIAGNOSTIC_TASK23.md)
- [docs/FINAL_SYNTHESIS_TASK25.md](docs/FINAL_SYNTHESIS_TASK25.md)
- [docs/NLP_SPEED_ACTION_PLAN.md](docs/NLP_SPEED_ACTION_PLAN.md)
- [docs/NLP_PROGRESS.md](docs/NLP_PROGRESS.md)
- [docs/RESULTS_ANALYSIS_RETROSPECTIVE.md](docs/RESULTS_ANALYSIS_RETROSPECTIVE.md)
- [configs/protocols/core_experiment.yaml](configs/protocols/core_experiment.yaml)
- [docs/PROGRESS_LOG.md](docs/PROGRESS_LOG.md)

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

Verify the current codebase:
```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The expected current result is `91 passed`.

Files of interest:
- [requirements.txt](requirements.txt)
- [setup_env.ps1](setup_env.ps1)
- [setup_env.sh](setup_env.sh)
- [src/main.py](src/main.py)
- [src/research_protocol.py](src/research_protocol.py)
- [docs/ACTION_PLAN.md](docs/ACTION_PLAN.md)
- [docs/CORE_EXPERIMENT_PROTOCOL.md](docs/CORE_EXPERIMENT_PROTOCOL.md)
- [docs/PROGRESS_LOG.md](docs/PROGRESS_LOG.md)
