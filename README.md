# TWR-LM

`TWR-LM`은 `tokenless latent backbone` 실험을 위한 PyTorch 연구 저장소입니다. 현재 메인라인은 예전 `write / think / slot-gate` 계열이 아니라 `overlapping latent encoder -> hierarchical latent backbone -> summary readout` 구조입니다.

## Mainline

- compression 이후 persistent state는 token-wise hidden state가 아니라 latent sequence / latent pyramid입니다.
- backbone과 readout은 attention-free latent 연산을 기준으로 정리되어 있습니다.
- 현재 유지 대상 TWR 실험은 `twr_backbone_*` 계열입니다.
- 예전 `twr_*`, `ablation_*`, `count_compare_*` 설정과 관련 문서는 정리했습니다.

핵심 구현 파일:

- [src/twr/models/twr_lm.py](/home/dause/Desktop/TWR-LM/src/twr/models/twr_lm.py)
- [src/twr/modules/latent_encoder.py](/home/dause/Desktop/TWR-LM/src/twr/modules/latent_encoder.py)
- [src/twr/modules/latent_backbone.py](/home/dause/Desktop/TWR-LM/src/twr/modules/latent_backbone.py)
- [src/twr/training/trainer.py](/home/dause/Desktop/TWR-LM/src/twr/training/trainer.py)

## Layout

```text
configs/
  data/
  experiment/
  model/
  train/
experiments/
  runs/
  analysis/
scripts/
  train.py
  run_benchmark_twr.py
  run_benchmark_suite.py
  analyze_results.py
src/twr/
  baselines/
  data/
  eval/
  models/
  modules/
  training/
  utils/
tests/
```

## Main Commands

default TWR ListOps:

```bash
python3 scripts/train.py --experiment configs/experiment/twr_backbone_lra_listops.yaml
```

default TWR 4-benchmark sweep:

```bash
python3 scripts/run_benchmark_twr.py
```

default TWR vs baseline comparison suite:

```bash
python3 scripts/run_benchmark_suite.py
```

compression ablation:

```bash
python3 scripts/run_compression_ablation_suite.py
```

로컬 캐시 기반 데이터 준비:

```bash
python3 scripts/prepare_local_datasets.py --dataset all
```

## Notes

- `Hyperpartisan`, `LongBench TREC`는 로컬 캐시 우선 설정을 사용합니다.
- 현행 상태는 `README`, `configs/`, `src/`, `scripts/`, `tests/` 기준으로 보면 됩니다.
- 집계 결과는 [experiments/analysis/summary_report.md](/home/dause/Desktop/TWR-LM/experiments/analysis/summary_report.md)에 기록됩니다.
