# TWR-LM

Tokenless Write-Think-Read Latent Machine 연구 코드베이스입니다.  
이 저장소는 데모보다 실험 재현성과 비교 실험을 우선하는 PyTorch 스캐폴드로 구성되어 있습니다.

## 프로젝트 목표

TWR-LM v1은 다음 원칙을 따릅니다.

- 입력 토큰은 먼저 일시적인 이벤트 표현으로 인코딩됩니다.
- 이벤트는 작은 latent memory bank에 순차적으로 기록됩니다.
- write 이후에는 토큰별 hidden state를 persistent state로 유지하지 않습니다.
- 이후 연산은 latent memory 위에서만 반복 수행됩니다.
- adaptive think depth와 slot-wise gating을 지원합니다.
- 동일한 trainer 위에서 baseline, ablation, 재현 가능한 logging을 함께 지원합니다.

## 모델 구조

현재 구현된 TWR-LM의 흐름은 아래와 같습니다.

```text
tokens
  -> EventEncoder
  -> SequentialSoftWrite
  -> LatentMemory slots
  -> ThinkLoop
  -> ReadoutHead
  -> logits
```

각 단계의 역할은 다음과 같습니다.

1. `EventEncoder`
   토큰 id와 위치 정보를 결합해 `event` 벡터로 변환합니다. 이 표현은 write 단계에만 사용되는 transient representation입니다.

2. `LatentMemory`
   배치별로 복제되는 learnable initial memory를 생성합니다. write 이후 persistent state는 이 memory tensor뿐입니다.

3. `SequentialSoftWrite`
   시퀀스의 각 event를 slot들에 soft attention으로 기록합니다.  
   기본 구현은 event별 순차 write이며, ablation용 write variant도 config로 바꿀 수 있습니다.

4. `ThinkLoop`
   write가 끝난 latent memory만 입력으로 받아 반복 정제합니다.  
   각 step마다:
   - `RefineBlock`이 slot mixing + GLU MLP 기반 delta를 계산하고
   - `step_gate`가 해당 think step의 강도를 조절하며
   - `slot_gate`가 slot별 갱신량을 조절합니다.

5. `ReadoutHead`
   최종 latent slots를 평균 풀링해 classification logits를 만듭니다.

## 모델 특징

TWR-LM v1의 핵심 특징은 다음과 같습니다.

- `tokenless persistent state`
  토큰은 event로 변환된 뒤 write 단계에서 latent memory에 반영되고, 이후 persistent state로 유지되지 않습니다. 반복 계산의 대상은 오직 latent slots입니다.

- `small latent memory bank`
  긴 토큰 시퀀스를 그대로 유지하지 않고, 고정된 개수의 slot에 정보를 압축해 저장합니다. 계산량과 상태 표현을 memory bank 중심으로 통제하기 쉽습니다.

- `sequential write, latent-only think`
  입력 처리와 추론 단계를 분리합니다. 입력은 순차적으로 write하고, 그 다음 think 단계는 입력 토큰이 아니라 latent memory만 반복 정제합니다.

- `adaptive think depth`
  각 샘플에 대해 think step의 강도를 `step_gate`로 조절합니다. 쉬운 샘플은 얕게, 어려운 샘플은 더 깊게 계산하는 실험을 할 수 있습니다.

- `slot-wise gating`
  모든 slot을 매 step 동일하게 업데이트하지 않고, `slot_gate`로 slot별 갱신량을 조절합니다. 어떤 slot이 실제로 활성화되는지 분석하기 쉽습니다.

- `experiment-first design`
  TWR 자체보다도 비교 가능성과 재현성을 우선합니다. 동일한 trainer와 logging 체계에서 Transformer, Perceiver, Mamba placeholder baseline 및 ablation을 함께 돌릴 수 있습니다.

- `simple and debuggable v1`
  v1은 구조를 과도하게 복잡하게 만들지 않았습니다. event encoder, write, think, readout이 분리돼 있어 병목과 기여도를 모듈 단위로 추적하기 쉽습니다.

- `analysis-friendly outputs`
  forward 결과로 logits뿐 아니라 `effective_depth`, `step_gates`, `slot_gates`, slot histogram 등을 함께 반환합니다. 그래서 성능만이 아니라 "얼마나 깊게 생각했는지", "어떤 slot을 썼는지"를 같이 볼 수 있습니다.

## 핵심 모듈 설명

- [`src/twr/models/twr_lm.py`](/workspaces/HDN/src/twr/models/twr_lm.py)
  TWR-LM 전체 조립 지점입니다. `event_encoder`, `writer`, `think`, `readout`를 연결합니다.

- [`src/twr/modules/event_encoder.py`](/workspaces/HDN/src/twr/modules/event_encoder.py)
  token embedding + positional embedding 뒤에 projection을 적용해 event를 만듭니다.

- [`src/twr/modules/sequential_write.py`](/workspaces/HDN/src/twr/modules/sequential_write.py)
  각 event를 latent slots에 순차 기록합니다. slot usage 통계도 함께 반환합니다.

- [`src/twr/modules/latent_memory.py`](/workspaces/HDN/src/twr/modules/latent_memory.py)
  learnable initial latent bank를 정의합니다.

- [`src/twr/modules/think_loop.py`](/workspaces/HDN/src/twr/modules/think_loop.py)
  adaptive depth와 slot gate를 포함한 반복 refine loop입니다.

- [`src/twr/modules/refine_block.py`](/workspaces/HDN/src/twr/modules/refine_block.py)
  low-rank slot mixer와 GLU feed-forward로 memory delta를 계산합니다.

- [`src/twr/modules/readout.py`](/workspaces/HDN/src/twr/modules/readout.py)
  slot pooled representation을 최종 class logits로 변환합니다.

## 실험 스택

실행은 experiment config 하나로 시작합니다.

```text
configs/experiment/*.yaml
  -> configs/model/*.yaml
  -> configs/data/*.yaml
  -> configs/train/*.yaml
  -> merged config
  -> scripts/train.py
  -> trainer
```

- [`src/twr/utils/config.py`](/workspaces/HDN/src/twr/utils/config.py)
  experiment 파일이 model/data/train 설정 파일을 참조하도록 하고, `overrides`를 deep merge합니다.

- [`scripts/train.py`](/workspaces/HDN/scripts/train.py)
  experiment 설정을 읽어 training entrypoint를 호출합니다.

- [`src/twr/training/trainer.py`](/workspaces/HDN/src/twr/training/trainer.py)
  dataloader 생성, optimizer/scheduler, epoch loop, validation, checkpoint, summary/analysis 저장을 담당합니다.

현재 trainer는 다음을 지원합니다.

- synthetic sequence classification
- Hugging Face text classification 데이터셋
- TWR, Transformer, Perceiver, Mamba placeholder baseline
- depth/slot 관련 보조 지표 기록

## 디렉토리 구조

```text
.
├── configs/
│   ├── data/           # synthetic, long-context, hf text 데이터 설정
│   ├── experiment/     # 실행 단위 실험 설정, model/data/train 조합 및 override 정의
│   ├── model/          # TWR, Transformer, Perceiver, Mamba placeholder 모델 설정
│   └── train/          # epoch, lr, seed, penalty 등 학습 설정
├── experiments/
│   └── README.md       # 실행 결과물 폴더 관련 메모
├── scripts/
│   ├── train.py        # 기본 학습 진입점
│   ├── evaluate.py     # 평가 스크립트
│   └── run_*.py        # ablation/suite 실행용 보조 스크립트
├── src/twr/
│   ├── baselines/      # Transformer, Perceiver, Mamba placeholder
│   ├── data/           # synthetic/hf dataset 및 collator
│   ├── eval/           # metric, analysis 유틸리티
│   ├── models/         # model factory, TWR-LM 본체
│   ├── modules/        # event/write/think/read 세부 모듈
│   ├── training/       # trainer, loss
│   └── utils/          # config, logging, seed, profiling
├── tests/              # config/model/write 관련 테스트
├── AGENTS.md           # 작업 지침
└── README.md
```

## 주요 설정 파일

- `configs/model/twr_v1.yaml`
  기본 TWR-LM 설정입니다. `slots`, `slot_dim`, `think_steps`, `adaptive_depth`, `use_slot_gate` 등이 정의됩니다.

- `configs/data/synthetic_parity.yaml`
  빠른 디버깅용 synthetic parity 태스크입니다.

- `configs/experiment/twr_debug.yaml`
  `model/data/train` 설정 파일을 묶는 가장 기본적인 실행 단위입니다.

## 설치

```bash
pip install -e .
```

## 빠른 시작

기본 TWR 디버그 실험:

```bash
python scripts/train.py --experiment configs/experiment/twr_debug.yaml
```

Transformer baseline:

```bash
python scripts/train.py --experiment configs/experiment/transformer_debug.yaml
```

Mamba/SSM placeholder baseline:

```bash
python scripts/train.py --experiment configs/experiment/mamba_placeholder_debug.yaml
```

## 실행 결과물

실험 결과는 `experiments/runs/<run_name>/` 아래에 저장됩니다.

각 run은 보통 다음 파일을 생성합니다.

- `config_snapshot.json`
- `metrics.jsonl`
- `summary.json`
- `analysis.json`
- `checkpoint.pt`

## 현재 코드베이스에서 보는 TWR-LM의 의미

이 저장소의 TWR-LM v1은 "토큰을 계속 들고 다니는 sequence backbone"이 아니라:

- 토큰을 event로 잠깐 변환하고
- 작은 latent slots에 기록한 뒤
- 남은 계산은 memory slots 위에서만 수행하는

실험 중심의 단순하고 디버그 가능한 tokenless latent 모델입니다.
