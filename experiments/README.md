# Experiments

Store generated experiment logs and results here. Experiment configs live in
[configs/experiments](../configs/experiments), and the main research result
summaries live in [docs](../docs).

Generated experiment directories are intentionally ignored by Git because full
runs can become multi-gigabyte artifacts. Commit source code, configs, tests,
and documentation instead.

Every experiment directory should record:

- protocol ID and version
- dataset or benchmark split
- compared methods
- seeds
- fixed controls such as model, optimizer, and replay budget
- reported metrics and saved artifacts

Until a new protocol version is created, the default reference is [configs/protocols/core_experiment.yaml](../configs/protocols/core_experiment.yaml).
