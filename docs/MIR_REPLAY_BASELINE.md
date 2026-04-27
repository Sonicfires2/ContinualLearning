# Task 14 MIR Replay Baseline

## Purpose

Task 14 adds Maximally Interfered Retrieval as a stronger replay baseline.
The proposal names MIR as a credible comparator because it does not replay
uniformly from memory. It asks which stored samples are most likely to be harmed
by the incoming update, then replays those samples.

This matters for the research goal because beating random replay is not enough
to show that the proposed spacing-inspired method is competitive. A strong
study should also ask whether replay timing and forgetting prediction can match
or improve on an interference-aware replay selector.

Primary source:

- Aljundi et al., [Online Continual Learning with Maximal Interfered Retrieval](https://papers.nips.cc/paper/9357-online-continual-learning-with-maximal-interfered-retrieval), NeurIPS 2019.

## Implemented Variant

Implementation:

- [src/replay/mir.py](../src/replay/mir.py)
- [src/baselines/mir_replay.py](../src/baselines/mir_replay.py)
- [src/experiments/mir_replay_comparison.py](../src/experiments/mir_replay_comparison.py)
- [scripts/run_mir_replay_comparison.py](../scripts/run_mir_replay_comparison.py)

Configs:

- [configs/experiments/mir_replay_smoke.yaml](../configs/experiments/mir_replay_smoke.yaml)
- [configs/experiments/mir_replay_split_cifar100.yaml](../configs/experiments/mir_replay_split_cifar100.yaml)
- [configs/experiments/mir_replay_comparison_smoke.yaml](../configs/experiments/mir_replay_comparison_smoke.yaml)
- [configs/experiments/mir_replay_comparison_split_cifar100.yaml](../configs/experiments/mir_replay_comparison_split_cifar100.yaml)

Tests:

- [tests/replay/test_mir.py](../tests/replay/test_mir.py)
- [tests/baselines/test_mir_replay_baseline.py](../tests/baselines/test_mir_replay_baseline.py)
- [tests/experiments/test_mir_replay_comparison.py](../tests/experiments/test_mir_replay_comparison.py)

The implemented policy is ER-MIR with the MI-1 score:

```text
pre_loss_i = CE(f_theta(x_i), y_i)
theta_prime = theta - alpha * grad_theta CE(f_theta(x_current), y_current)
post_loss_i = CE(f_theta_prime(x_i), y_i)
interference_i = post_loss_i - pre_loss_i
```

At each replay-eligible optimizer step:

1. sample `mir_candidate_size` candidates from the replay buffer;
2. compute each candidate's loss before the virtual current-batch update;
3. apply the virtual update only in memory;
4. compute candidate losses after the virtual update;
5. restore the real model parameters exactly;
6. replay the top `replay_batch_size` candidates by `post_loss - pre_loss`.

Only selected candidates count against the replay budget. Candidate scoring is
extra selection overhead, not extra replay training data.

## Fairness Controls

The full comparison used:

- protocol: `core_split_cifar100_v2`
- benchmark: Split CIFAR-100
- task count: `10`
- classes per task: `10`
- split seed: `0`
- model: flatten MLP, hidden dimension `256`
- epochs per task: `1`
- batch size: `32`
- learning rate: `0.01`
- replay capacity: `2000`
- replay batch size: `32`
- MIR candidate size: `128`
- MIR virtual learning rate: `0.01`
- seeds: `0`, `1`, `2`

MIR used the same replay sample budget as Task 13 random replay and spaced
replay: `45216` replay samples per seed.

Leakage guard:

- the replay buffer only contains already-seen training samples;
- candidate scoring uses only the current incoming batch and current replay
  memory;
- no future-task labels, future evaluations, or future signal logs are used
  during selection;
- tests verify that candidate sampling does not update replay utilization
  counters and that virtual updates restore model parameters.

## Aggregate Results

Full summary artifact:

```text
experiments/task14_mir_replay_comparison/task14_mir_replay_comparison_split_cifar100_summary.json
```

Three-seed MIR result:

| Method | Final accuracy mean | Final accuracy std | Avg forgetting mean | Avg forgetting std | Replay samples mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| MIR replay | `0.11636666666666667` | `0.0020033305601755646` | `0.2167037037037037` | `0.003425275213477844` | `45216` |

Task 14 summary deltas against Task 13:

| Reference method | Final accuracy delta, MIR minus reference | Avg forgetting delta, MIR minus reference |
| --- | ---: | ---: |
| fine-tuning | `0.06933333333333333` | `-0.21433333333333335` |
| random replay | `0.014800000000000021` | `-0.08603703703703705` |
| fixed-periodic replay, `k=1` | `0.014800000000000021` | `-0.08603703703703705` |
| spaced replay, due-time proxy | `0.017733333333333337` | `-0.09640740740740741` |

Per-seed MIR results:

| Seed | Final accuracy | Average forgetting | Replay samples | Mean selected interference score |
| ---: | ---: | ---: | ---: | ---: |
| `0` | `0.1183` | `0.21400000000000002` | `45216` | `0.19384874675030078` |
| `1` | `0.1165` | `0.22055555555555553` | `45216` | `0.19290109919141898` |
| `2` | `0.1143` | `0.21555555555555556` | `45216` | `0.19785929542092207` |

## Scientific Interpretation

MIR is currently the strongest implemented method in this repository. It
improves final accuracy and substantially reduces average forgetting compared
with random replay, fixed-periodic replay, and the first spaced-replay due-time
proxy under the same replay sample budget.

This result is good for the project, but it is not evidence for the proposal's
spacing mechanism. MIR is an interference-aware sample selector, not a
cognitive-spacing scheduler and not a `T_i` estimator. The result instead says:

- replay sample choice matters in this setup;
- the current spaced scheduler is weaker than a literature-grounded strong
  replay selector;
- Task 15 should test an event-triggered risk-gated scheduler that skips replay
  when no buffered sample is predicted to be near forgetting, and later
  ablations should diagnose weak risk scores, weak `T_i` timing, and budget-fill
  behavior.

## Current Claim Boundary

The project can now claim:

- a MIR-style stronger baseline is implemented, tested, and run across the same
  seed list as Task 13;
- MIR improves over random replay and the current spaced replay proxy in this
  Split CIFAR-100 setup;
- the first spaced replay proxy is not competitive with MIR.

The project cannot yet claim:

- spacing-inspired replay improves continual learning;
- the current scheduler estimates true sample-specific `T_i` accurately;
- gradient norms or representation drift explain the MIR advantage;
- MIR is universally superior beyond this protocol.

## Verification

Focused tests:

```text
.\.venv\Scripts\python.exe -m pytest tests\replay\test_mir.py tests\baselines\test_mir_replay_baseline.py tests\experiments\test_mir_replay_comparison.py -q
```

Result:

```text
8 passed
```

Smoke comparison:

```text
.\.venv\Scripts\python.exe scripts\run_mir_replay_comparison.py --config configs\experiments\mir_replay_comparison_smoke.yaml
```

Full comparison:

```text
.\.venv\Scripts\python.exe scripts\run_mir_replay_comparison.py --config configs\experiments\mir_replay_comparison_split_cifar100.yaml
```

Full repository verification after Task 14:

```text
.\.venv\Scripts\python.exe -m pytest -q
```

Result:

```text
59 passed
```
