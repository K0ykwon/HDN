# TWR-LM

Tokenless Write-Think-Read Latent Machine research scaffold in PyTorch.

The repo is organized around one config-driven training stack that supports:
- TWR-LM as the primary tokenless memory backbone
- comparable baselines under the same trainer
- synthetic sequence classification for fast iteration
- structured experiment logging, checkpoints, and run summaries

## Install

```bash
pip install -e .
```

## Quick Start

Run the default TWR debug experiment:

```bash
python scripts/train.py --experiment configs/experiment/twr_debug.yaml
```

Run the Transformer baseline:

```bash
python scripts/train.py --experiment configs/experiment/transformer_debug.yaml
```

Run the Mamba/SSM placeholder baseline:

```bash
python scripts/train.py --experiment configs/experiment/mamba_placeholder_debug.yaml
```

Artifacts are written under `experiments/runs/<run_name>/`.

Each run saves:
- `config_snapshot.json`
- `metrics.jsonl`
- `summary.json`
- `analysis.json`
- `checkpoint.pt`
