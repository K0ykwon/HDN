# TWR-LM

`TWR-LM`은 Transformer를 대체할 수 있는 `tokenless latent backbone`을 실험하는 PyTorch 연구 코드베이스입니다.

현재 메인라인 구현은 초기의 `write -> memory slots -> adaptive think` 설계가 아니라, 더 단순하고 재현 가능한 `overlapping latent encoder -> hierarchical latent backbone -> query readout` 구조를 사용합니다.

## 현재 방향

- 입력 토큰은 overlapping window 단위 latent sequence로 압축됩니다.
- 압축 이후 persistent state는 token-wise hidden state가 아니라 latent sequence뿐입니다.
- backbone은 공유 refinement와 pairwise merge를 반복해 latent pyramid를 구성합니다.
- 최종 출력은 multi-query readout으로 읽습니다.
- 목표는 `Transformer 수준의 정확도`를 `더 작은 혹은 비슷한 규모의 tokenless backbone`으로 달성하는 것입니다.

## 현재 모델 구조

```text
tokens
  -> PretrainedLatentEncoder
  -> latent sequence
  -> HierarchicalLatentBackbone
       - shared local refinement
       - selective pairwise merge
       - multiscale query readout
  -> logits
```

핵심 구현 파일:

- [src/twr/models/twr_lm.py](/home/dause/Desktop/TWR-LM/src/twr/models/twr_lm.py)
- [src/twr/modules/latent_encoder.py](/home/dause/Desktop/TWR-LM/src/twr/modules/latent_encoder.py)
- [src/twr/modules/latent_backbone.py](/home/dause/Desktop/TWR-LM/src/twr/modules/latent_backbone.py)
- [src/twr/training/trainer.py](/home/dause/Desktop/TWR-LM/src/twr/training/trainer.py)

## 현재 실험 상태

최근 비교는 `TWR lean`, `Transformer`, `Mamba` 기준으로 정리되어 있습니다.

### ListOps

- `TWR lean`: `0.2285`, `82,587 params`
- `Transformer`: `0.2383`, `86,026 params`
- `Mamba`: `0.1289`, `95,530 params`

### RULER needle

- `TWR lean`: `0.0840`, `100,065 params`
- `Transformer`: `0.0801`, `117,136 params`
- `Mamba`: `0.0781`, `252,048 params`

### LongBench TREC

- `TWR lean`: `0.1800`, `494,083 params`
- `Transformer`: `0.1800`, `627,250 params`
- `Mamba`: `0.1800`, `1,272,242 params`

### Hyperpartisan

- `TWR lean`: `0.5400`, `475,283 params`
- `Transformer`: `0.5450`, `624,130 params`
- `Mamba`: `0.5150`, `1,266,050 params`

요약:

- `ListOps`, `Hyperpartisan`에서는 lean TWR가 Transformer에 거의 근접했습니다.
- `RULER`에서는 lean TWR가 근소 우세지만, 큰 TWR가 보이던 장점은 줄었습니다.
- `LongBench TREC`에서는 lean TWR가 Transformer와 동률입니다.
- 이번 설정에서는 Mamba가 네 벤치 모두 우세를 가져오지 못했습니다.

상세 비교는 [experiments/analysis/summary_report.md](/home/dause/Desktop/TWR-LM/experiments/analysis/summary_report.md) 를 보면 됩니다.

## 디렉토리 개요

```text
configs/
  data/         dataset 설정
  experiment/   실행 단위 config
  model/        TWR / Transformer / Mamba model config
  train/        학습 설정
experiments/
  runs/         실험 결과
  analysis/     집계 결과
scripts/
  train.py
  run_benchmark_full_suite.py
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

## 빠른 시작

lean TWR ListOps:

```bash
python3 scripts/train.py --experiment configs/experiment/twr_backbone_lra_listops_lean.yaml
```

Transformer ListOps:

```bash
python3 scripts/train.py --experiment configs/experiment/transformer_lra_listops_full.yaml
```

전체 비교 스위트:

```bash
python3 scripts/run_benchmark_full_suite.py
```

## 참고

- `twr_codex_pack/` 아래 문서들은 초기 write-think-read 설계용 스캐폴드입니다.
- 현재 저장소의 메인라인 구현과 완전히 일치하지 않을 수 있으므로, 현행 상태는 이 README와 `configs/`, `src/` 코드를 기준으로 봐야 합니다.
