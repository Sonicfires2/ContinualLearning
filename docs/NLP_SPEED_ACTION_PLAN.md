# NLP Speed-First Action Plan

## Purpose

This plan pivots the project from the image-based Split CIFAR-100 study to a
strictly NLP continual-finetuning study while prioritizing a usable result
inside a 10-hour work window.

Current execution status: the first one-seed sampled DBpedia14 result is
complete and documented in [NLP_PROGRESS.md](./NLP_PROGRESS.md).

The research idea stays the same:

```text
Track sample-level forgetting curves during sequential finetuning.
Use forgetting risk or replay timing to decide which old examples to replay.
Compare against simple replay baselines under the same memory and compute budget.
```

The benchmark and model change:

```text
Split CIFAR-100 + MLP
becomes
sampled NLP text classification + DistilBERT
```

## Current GPU Status

GPU setup is complete in a separate NLP environment:

```text
venv: .venv-nlp
python: 3.13.13
torch: 2.11.0+cu128
torchvision: 0.26.0+cu128
cuda_available: true
GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU
VRAM: 12227 MiB
```

DistilBERT has been downloaded into a repo-local Hugging Face cache under
`.tmp/hf-home`.

GPU sanity check:

```text
DistilBERT batch size: 32
max sequence length: 128
forward pass time: 0.2288 seconds
```

This means the 10-hour NLP pilot is realistic if the experiment stays sampled
and avoids full-dataset sweeps.

## Recommended Primary Benchmark

Use sampled Split DBpedia14 with DistilBERT.

Dataset:

- Hugging Face dataset: `fancyzhx/dbpedia_14`
- Task type: topic classification
- Labels: 14 text topic classes
- Split design: 7 continual tasks, 2 classes per task

Why this is the best first NLP benchmark:

- It is strictly NLP.
- It uses text classification with a transformer model.
- It is class-incremental, like Split CIFAR-100, so forgetting should be easier
  to observe.
- It avoids the ambiguity of sentiment-domain tasks where every task is still
  positive/negative classification.
- It is easier to explain in a report:

```text
The model learns new text topics over time and may forget earlier topics.
```

## Backup NLP Benchmark

Use sampled sentiment domains only if the report specifically needs domain
adaptation:

```text
Rotten Tomatoes -> IMDb -> Yelp Polarity -> Amazon Polarity
```

This is more obviously "domain continual finetuning", but it may show weaker
forgetting because all tasks share the same positive/negative labels.

Recommended backup datasets:

- `cornell-movie-review-data/rotten_tomatoes`
- `stanfordnlp/imdb`
- `fancyzhx/yelp_polarity`
- `fancyzhx/amazon_polarity`

## Better Sampled Pilot Configuration

The first serious run should use this configuration:

| Setting | Value |
| --- | --- |
| Model | `distilbert-base-uncased` |
| Dataset | `fancyzhx/dbpedia_14` |
| Continual tasks | 7 |
| Classes per task | 2 |
| Train examples per class | 1000 |
| Eval examples per class | 250 |
| Total train examples | 14000 |
| Total eval examples | 3500 |
| Max sequence length | 128 |
| Current-task batch size | 32 |
| Replay batch size | 32 |
| Effective replay-augmented batch | 64 |
| Epochs per task | 1 |
| Replay memory capacity | 2000 text examples |
| Seeds for first result | 1 seed |
| Seeds if time remains | 3 seeds |

If memory pressure appears, reduce current-task batch size to 16 and replay
batch size to 16 before reducing the dataset size.

## Required Methods For The 10-Hour Result

Run these methods first:

| Priority | Method | Why |
| ---: | --- | --- |
| 1 | Fine-tuning | Shows whether NLP continual finetuning forgets. |
| 2 | Random replay | Required simple replay baseline. |
| 3 | Fixed-periodic replay | Timing control baseline if easy to reuse. |
| 4 | Spaced/risk replay proxy | Directly targets the proposal idea. |
| 5 | Offline forgetting predictor | Shows whether text forgetting is predictable. |

Do not run MIR in the first 10-hour sprint unless the first result finishes
early. MIR is valuable but slower because it needs virtual updates and extra
candidate scoring.

## Metrics To Report

Use the same core metrics as the image study:

| Metric | Meaning |
| --- | --- |
| Final accuracy | How much the model knows after all NLP tasks. |
| Average forgetting | How much old-task performance dropped. |
| Replay samples | How much replay compute was used. |
| Predictor AP | Whether future-forgotten text examples are ranked high. |
| Predictor ROC-AUC | Whether forgotten and retained examples are separable. |
| Top-k precision | Whether the highest-risk text examples are truly fragile. |

For the NLP report, also include:

| Metric | Meaning |
| --- | --- |
| Per-task final accuracy | Which text topics were forgotten most. |
| Tokenization/training time | Shows the cost of the method. |
| Replay class coverage | Shows whether replay covers old text topics evenly. |

## 10-Hour Execution Schedule

### Hour 0: Environment

Status: complete.

- Created `.venv-nlp`.
- Installed CUDA PyTorch.
- Installed Transformers, Datasets, Accelerate, scikit-learn, and reporting
  dependencies.
- Verified DistilBERT runs on the RTX GPU.

### Hour 1: NLP Dataset And Task Stream

Implement a sampled DBpedia14 task stream:

- load `fancyzhx/dbpedia_14`;
- build stable `sample_id` values;
- split 14 classes into 7 sequential tasks;
- sample 1000 train and 250 eval examples per class;
- tokenize with `distilbert-base-uncased`;
- cache tokenized examples under `.tmp/nlp_cache` or `data/processed/nlp`.

Expected time:

```text
30-75 minutes, including dataset download and tokenization
```

### Hours 2-3: Fine-Tuning Baseline

Implement or adapt a DistilBERT continual trainer:

- train task by task;
- evaluate all seen tasks after each task;
- save accuracy matrix;
- log sample-level loss, confidence, uncertainty, and correctness.

Expected run time:

```text
5-20 minutes after tokenization
```

Expected output:

- final accuracy;
- average forgetting;
- sample-level signal log.

### Hours 3-4: Random Replay Baseline

Add text replay memory:

- store tokenized examples and metadata;
- sample old examples uniformly;
- mix replay batch with current-task batch;
- log replay counts and source task/class coverage.

Expected run time:

```text
10-30 minutes
```

This is the most important baseline. The proposed spacing method must beat this
before any positive claim is credible.

### Hours 4-5: Spaced/Risk Replay Proxy

Adapt the existing scheduler idea to text:

```text
risk_score = function(loss, uncertainty, low true-label probability, loss increase)
estimated_T_i = max_interval - risk_score * (max_interval - min_interval)
next_due_step = current_step + estimated_T_i
```

Replay due/high-risk text examples first.

Expected run time:

```text
10-35 minutes
```

Report honestly whether it beats random replay.

### Hours 5-6: Forgetting Labels And Predictor

Build NLP forgetting labels from evaluation traces:

```text
At evaluation anchor t, was this text example correct?
After later tasks, did it become incorrect?
```

Train a lightweight predictor:

- logistic regression first;
- linear SVM only if time remains;
- features from sample-level text evaluation logs.

Expected time:

```text
10-30 minutes
```

This gives a fast answer to:

```text
Can text-example forgetting be predicted from cheap signals?
```

### Hours 6-8: Better Run Or Three-Seed Expansion

If the first single-seed run succeeds:

Option A, preferred:

- run fine-tuning, random replay, and spaced replay for seeds 1 and 2;
- report mean and standard deviation.

Option B, if implementation took longer:

- keep one seed;
- add a stronger class-balanced replay baseline;
- improve plots and documentation.

### Hours 8-10: Report And Decision

Write the NLP result summary:

- benchmark description;
- method table;
- final accuracy and forgetting table;
- predictor AP/ROC-AUC table;
- per-task forgetting plot;
- clear conclusion.

The conclusion can be positive, negative, or mixed. The important thing is to
separate these claims:

```text
1. Did DistilBERT forget old NLP tasks?
2. Did random replay reduce forgetting?
3. Did spacing/risk replay beat random replay?
4. Could forgetting be predicted offline?
```

## Expected Runtime After CUDA Setup

These are realistic estimates for the sampled DBpedia14 pilot on the RTX 5070
Ti Laptop GPU:

| Work item | Expected time |
| --- | ---: |
| DBpedia14 download and sampling | 5-20 min |
| Tokenization | 5-20 min |
| Fine-tuning baseline | 5-20 min |
| Random replay | 10-30 min |
| Spaced/risk replay | 10-35 min |
| Predictor analysis | 10-30 min |
| Plot/report update | 30-90 min |
| Total after implementation | 1.5-3.5 hours |
| Total including implementation | 5-9 hours |

The model download has already been done once, so future runs should skip most
of that cost unless the cache is deleted.

## Success Criteria

Minimum result inside 10 hours:

- DistilBERT sequential NLP fine-tuning result; complete
- random replay result; complete
- spaced/risk replay result; complete
- final accuracy and average forgetting table; complete
- short explanation of whether spacing helped.

Better result inside 10 hours:

- all minimum results;
- forgetting predictor AP/ROC-AUC;
- per-task forgetting table or plot;
- class-balanced replay if time remains;
- seed-0 result plus at least one extra seed for the most important comparison.

Do not spend the first sprint on:

- full Yelp/Amazon runs;
- MIR;
- representation drift;
- large hyperparameter search;
- language generation tasks.

Those are good later additions, but they are not the shortest path to an NLP
result.

## Commands

Activate the NLP environment:

```powershell
.\.venv-nlp\Scripts\Activate.ps1
```

Check CUDA:

```powershell
.\.venv-nlp\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Use a repo-local Hugging Face cache:

```powershell
$env:HF_HOME = ".tmp\hf-home"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
```

## Source Notes

Hugging Face documentation supports this setup:

- Transformers text classification guide:
  https://huggingface.co/docs/transformers/tasks/sequence_classification
- Datasets loading and splits:
  https://huggingface.co/docs/datasets/load_hub

Candidate datasets:

- DBpedia14: https://hf.co/datasets/fancyzhx/dbpedia_14
- Rotten Tomatoes: https://hf.co/datasets/cornell-movie-review-data/rotten_tomatoes
- IMDb: https://hf.co/datasets/stanfordnlp/imdb
- Yelp Polarity: https://hf.co/datasets/fancyzhx/yelp_polarity
- Amazon Polarity: https://hf.co/datasets/fancyzhx/amazon_polarity

## Completed Seed-0 Result

| Method | Final accuracy | Avg forgetting | Replay samples | Time, sec |
| --- | ---: | ---: | ---: | ---: |
| fine-tuning | `0.30428571428571427` | `0.806` | `0` | `72.27` |
| random replay | `0.9880000000000001` | `0.006000000000000005` | `12096` | `98.98` |
| spaced replay | `0.9857142857142858` | `0.008000000000000007` | `12096` | `100.03` |

The first NLP result is clear: replay is crucial, and the first spaced replay
proxy is close to random replay but does not beat it.
