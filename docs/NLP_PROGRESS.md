# NLP Continual-Learning Progress

## Current Status

The speed-first NLP branch now has a first one-seed result on a sampled
DistilBERT continual-learning benchmark.

This is a strict NLP version of the research question:

```text
Can forgetting-aware replay reduce forgetting during continual DistilBERT
finetuning on sequential text-classification tasks?
```

The first benchmark is sampled Split DBpedia14:

| Setting | Value |
| --- | --- |
| Dataset | `fancyzhx/dbpedia_14` |
| Model | `distilbert-base-uncased` |
| Device | NVIDIA GeForce RTX 5070 Ti Laptop GPU |
| Tasks | 7 |
| Classes per task | 2 |
| Train examples per class | 1000 |
| Eval examples per class | 250 |
| Total train examples | 14000 |
| Total eval examples | 3500 |
| Max sequence length | 128 |
| Seed | 0 |
| Epochs per task | 1 |
| Replay memory capacity | 2000 |
| Replay batch size | 32 |

The runner is:

[scripts/run_nlp_continual_pilot.py](../scripts/run_nlp_continual_pilot.py)

The config is:

[configs/experiments/nlp_dbpedia14_speed_pilot.yaml](../configs/experiments/nlp_dbpedia14_speed_pilot.yaml)

Generated run artifacts are saved under:

```text
experiments/runs/nlp_continual_pilot/
```

Those generated artifacts are intentionally ignored by Git. The important
results are copied into this document.

## Progress Tracker

| Step | Status | Notes |
| --- | --- | --- |
| Create CUDA NLP environment | complete | `.venv-nlp` uses CUDA PyTorch and sees the RTX GPU. |
| Choose fast NLP benchmark | complete | Sampled Split DBpedia14, 7 tasks. |
| Implement NLP task stream | complete | Stable sample IDs, class-incremental text tasks, tokenized DistilBERT inputs. |
| Implement fine-tuning baseline | complete | No replay. |
| Implement random replay baseline | complete | Uniform replay from bounded text memory. |
| Implement spaced replay proxy | complete | High-risk/near-due text examples replayed first. |
| Produce final accuracy and forgetting table | complete | One-seed table below. |
| Add NLP forgetting predictor | next | Use saved `eval_signals.json` rows to predict future text forgetting. |
| Add class-balanced replay | optional next | Useful because class-balanced replay helped in the image study. |
| Add MIR-style NLP diagnostic | later | Valuable but slower; not required for the first NLP result. |
| Run 3 seeds | optional if time | Needed for stronger claims. |

## One-Seed NLP Result Table

| Method | Final accuracy | Avg forgetting | Replay samples | Time, sec |
| --- | ---: | ---: | ---: | ---: |
| fine-tuning | `0.30428571428571427` | `0.806` | `0` | `72.27` |
| random replay | `0.9880000000000001` | `0.006000000000000005` | `12096` | `98.98` |
| spaced replay | `0.9857142857142858` | `0.008000000000000007` | `12096` | `100.03` |

## Plain-English Interpretation

Fine-tuning alone forgets badly. It learns the current text topics, but older
topic tasks collapse after later tasks. Its final accuracy is only about `30%`,
and its average forgetting is very high at `0.806`.

Random replay almost eliminates forgetting on this sampled NLP benchmark. It
keeps a small memory of old text examples and mixes them into later training.
Final accuracy reaches about `98.8%`, and average forgetting drops to about
`0.006`.

The first spaced replay proxy is also very strong, but it does not beat random
replay in this first run. It reaches about `98.6%` final accuracy and `0.008`
average forgetting using the same number of replay samples as random replay.

So the first NLP result matches the image-study lesson:

```text
Replay helps a lot.
The first spacing/risk rule does not yet beat random replay.
```

The NLP result is still useful because it shows that the research question can
be studied on text, not only images.

## Accuracy Matrices

Each row is the model state after finishing a task. Each column is accuracy on a
previously seen task. Future tasks are `null` in the saved artifacts.

### Fine-Tuning

```text
[
  [0.986, null, null, null, null, null, null],
  [0.0, 0.996, null, null, null, null, null],
  [0.0, 0.0, 1.0, null, null, null, null],
  [0.0, 0.0, 0.156, 0.992, null, null, null],
  [0.0, 0.0, 0.0, 0.0, 1.0, null, null],
  [0.0, 0.0, 0.412, 0.096, 0.5, 0.998, null],
  [0.0, 0.004, 0.134, 0.0, 0.5, 0.498, 0.994],
]
```

### Random Replay

```text
[
  [0.986, null, null, null, null, null, null],
  [0.99, 0.992, null, null, null, null, null],
  [0.978, 0.988, 0.992, null, null, null, null],
  [0.966, 0.99, 0.99, 0.988, null, null, null],
  [0.98, 0.99, 0.99, 0.976, 1.0, null, null],
  [0.976, 0.99, 0.992, 0.984, 0.996, 0.998, null],
  [0.964, 0.992, 0.986, 0.988, 1.0, 0.994, 0.992],
]
```

### Spaced Replay Proxy

```text
[
  [0.986, null, null, null, null, null, null],
  [0.978, 0.992, null, null, null, null, null],
  [0.982, 0.99, 0.992, null, null, null, null],
  [0.974, 0.992, 0.992, 0.986, null, null, null],
  [0.97, 0.99, 0.988, 0.988, 1.0, null, null],
  [0.98, 0.984, 0.988, 0.984, 0.75, 0.998, null],
  [0.962, 0.99, 0.984, 0.984, 0.99, 0.998, 0.992],
]
```

## What This Says About The NLP Research Goal

The NLP branch has already answered the first two questions:

1. DistilBERT does forget old NLP tasks under sequential fine-tuning.
2. Replay strongly reduces forgetting.

It has not yet supported the spacing claim:

```text
The first spaced replay proxy does not beat random replay on seed 0.
```

That does not end the NLP project. It gives the same useful diagnosis as the
image study: the current risk-to-timing rule is probably too crude. The next NLP
task should test whether the saved text-example signals can predict future
forgetting, then decide whether to redesign the replay selector.

## Reproduction Command

Use the CUDA-enabled NLP environment:

```powershell
$env:HF_HOME = ".tmp\hf-home"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:TOKENIZERS_PARALLELISM = "false"
.\.venv-nlp\Scripts\python.exe scripts\run_nlp_continual_pilot.py --config configs\experiments\nlp_dbpedia14_speed_pilot.yaml
```

CUDA check:

```powershell
.\.venv-nlp\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Expected CUDA output:

```text
2.11.0+cu128
True
NVIDIA GeForce RTX 5070 Ti Laptop GPU
```

