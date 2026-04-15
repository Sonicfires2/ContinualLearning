# Continual Learning Research Project

## Research Goal
Develop and evaluate continual learning methods for neural networks that reduce catastrophic forgetting while allowing efficient adaptation to new tasks.

## Research Question
How can we design training protocols and model components that enable stable knowledge retention across sequential tasks while maintaining strong performance on new tasks?

## Data Needed
- Standard image classification datasets for continual learning benchmarks (e.g., CIFAR-10, CIFAR-100, MNIST, PermutedMNIST)
- Optionally: domain-specific datasets if extending to other modalities (text, audio)

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
1. Prepare datasets and data loaders for sequential tasks
2. Implement baseline continual learning methods (fine-tune, replay, EWC)
3. Implement proposed method(s) and training protocol
4. Run experiments on benchmarks and collect metrics (accuracy, forgetting)
5. Analyze results and iterate on model/hyperparameters

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
