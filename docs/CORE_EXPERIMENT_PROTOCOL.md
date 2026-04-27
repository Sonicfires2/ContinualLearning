# Core Experiment Protocol

## Purpose

This document locks the project's core experimental scope so future implementation work stays aligned with the research goal instead of gradually expanding into a harder but less answerable project.

The machine-readable version of this protocol lives at [configs/protocols/core_experiment.yaml](../configs/protocols/core_experiment.yaml).

## Locked Protocol

Protocol ID: `core_split_cifar100_v2`  
Status: `locked`  
Locked on: `2026-04-24`

Version `v2` corrects the protocol after reading the text-friendly proposal
source [Project_Proposal-2.tex](../Project_Proposal-2.tex). The proposal
explicitly requires estimating a sample-specific forgetting time `T_i` and
comparing the adaptive scheduler against fixed schedules such as replay once
every `k` iterations.

## Research Goal

Develop and evaluate spacing-inspired replay policies that reduce catastrophic forgetting during continual fine-tuning under limited memory and compute.

## Research Question

Can sample-level forgetting signals, such as uncertainty, loss history, gradient norms, and representation drift, predict when examples should be replayed better than random or fixed replay schedules?

## Core Scope

### Required benchmark

- Split CIFAR-100
- Default split assumption: `10` tasks with `10` classes per task

### Required methods

- naive sequential fine-tuning
- random replay
- fixed-periodic replay every `k` optimizer steps
- spaced replay

### Recommended next baseline

- MIR

### Stretch-only methods

- uncertainty-guided replay
- ESMER

### Required forgetting signals for the first valid study

- loss history
- predictive uncertainty

### Required telemetry to support later replay analysis

- replay count
- last replay step
- estimated forgetting time `T_i`
- next scheduled replay step
- scheduler selection reason

### Optional later signals

- gradient norm
- representation drift

### Optional later benchmarks

- Split CUB
- DistilBERT sequential domain adaptation

## Locked Evaluation Protocol

Every direct comparison in the core study must keep these controls fixed:

- task order
- model architecture
- optimizer settings
- replay buffer capacity
- replay batch size
- training epochs per task

Every core run must:

- evaluate all seen tasks after each task finishes
- preserve the same benchmark split across compared methods
- save enough information to reconstruct the run

### Required metrics

- final accuracy
- average forgetting

### Recommended metrics

- average accuracy
- training time
- replay utilization
- predictor precision-recall AUC
- time-to-forgetting prediction error

### Required run artifacts

- accuracy matrix
- run config
- seed list
- method description
- scheduler trace for replay-based methods

## Why This Protocol Comes First

This protocol is implemented before new training code because the research question only makes sense if all later comparisons are constrained by the same benchmark, methods, and evaluation rules.

Without a locked protocol, the project can accidentally compare methods under different budgets, different task orders, or different benchmark choices, which would weaken the scientific claim even if the code looks correct.

## Future-Proof Design Rules

- The protocol exists in both human-readable and machine-readable form so future code can load the same contract the team reads.
- The protocol is versioned so later extensions can be added without rewriting history.
- Required, recommended, and stretch items are separated so the codebase can grow without blurring what is necessary for the core claim.
- Optional benchmarks and expensive signals are recorded now, but explicitly kept outside the locked core path until the primary result is stable.

## Change Control

If the team changes any of the following, it must create a new protocol version and a progress-log entry:

- required benchmark
- required methods
- required metrics
- fixed evaluation controls

This keeps future work additive and traceable instead of letting the target move silently.
