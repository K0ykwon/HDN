# HDN 프로젝트 개요

## 1. 프로젝트명
**HDN (Homeostatic Developmental Network)**

부제: **학습 중 구조가 생성·분화·심화·소거되는 성장형 신경망**

---

## 2. 프로젝트 한줄 정의
HDN은 **최소한의 입력→출력 직결 구조**에서 시작해, 학습 과정에서 필요할 때만 은닉 표현을 생성하고, 과부하된 노드를 분화시키며, 병렬 확장만으로 부족할 경우 깊이를 늘리고, 불필요하거나 중복된 구조는 pruning하는 **발달형 신경망 프레임워크**이다.

---

## 3. 문제의식
일반적인 신경망은 학습 전에 층 수와 노드 수를 대부분 고정한 뒤 학습한다. 하지만 실제로는 다음과 같은 문제가 발생한다.

- 일부 노드나 경로에 정보 처리 부담이 과도하게 집중됨
- 반대로 거의 사용되지 않는 유휴 노드가 존재함
- 모델 크기를 넉넉하게 잡아야 안정적으로 성능이 나오는 경우가 많음
- 특정 소수 노드나 경로에 대한 의존성이 높아 강건성이 떨어질 수 있음

HDN은 이러한 문제를 해결하기 위해, 신경망 구조를 고정된 설계물이 아니라 **학습 과정에서 발달하는 구조**로 본다.

---

## 4. 핵심 아이디어
HDN의 핵심 철학은 다음과 같다.

1. **처음에는 최소 구조만 둔다.**  
   입력에서 출력으로 바로 가는 아주 작은 구조로 시작한다.

2. **필요할 때만 내부 표현을 만든다.**  
   현재 구조로 설명되지 않는 오차 패턴이 나타날 때 새로운 hidden representation을 생성한다.

3. **정보가 과도하게 몰리는 노드는 분화시킨다.**  
   하나의 노드가 서로 다른 기능적 패턴을 동시에 떠안으면 split을 통해 역할을 나눈다.

4. **width 확장만으로 부족하면 depth를 만든다.**  
   병렬 노드 추가로 해결되지 않는 경우 새로운 변환 단계(깊이)를 추가한다.

5. **쓰이지 않거나 중복된 구조는 제거한다.**  
   usage, redundancy, contribution 기준으로 pruning하여 구조를 정리한다.

즉, HDN은 **Birth → Split → Deepen → Prune** 의 순환을 통해 구조를 스스로 조직한다.

---

## 5. 목표
HDN의 목표는 단순히 모델을 키우는 것이 아니다. 다음 두 가지를 동시에 달성하는 것이 목적이다.

### 5.1 강건성 향상
- 특정 노드 또는 경로에 대한 과의존 감소
- 일부 노드 제거 또는 교란 상황에서의 성능 저하 완화
- 입력 변화나 corruption에 대한 더 나은 대응

### 5.2 모델 효율화
- 필요한 구조만 생성하고 불필요한 구조는 제거
- 더 작은 파라미터 수와 더 낮은 FLOPs 달성
- 고정 구조 대비 더 나은 accuracy–efficiency trade-off 확보

---

## 6. 차별점
HDN은 기존의 다음 연구 줄기들과 관련이 있다.

- constructive / growing neural networks
- network morphism
- adaptive width / dynamic depth
- neuron splitting
- grow-and-prune frameworks
- load balancing in modular / MoE architectures

그러나 HDN의 차별점은 위 요소들을 개별 기법으로 쓰는 것이 아니라, 다음을 **하나의 통합 발달 알고리즘**으로 묶는 데 있다.

- residual-driven representation birth
- gradient heterogeneity 기반 node split
- demand-driven depth formation
- usage/redundancy/contribution 기반 pruning
- functional load homeostasis

즉, HDN의 기여는 “구조를 키운다” 자체가 아니라, **언제 생성하고, 언제 분화하고, 언제 깊게 만들고, 언제 줄일 것인가를 하나의 원리로 통합한다는 점**에 있다.

---

## 7. 핵심 개념

### 7.1 Homeostasis
모든 노드가 완전히 동일한 역할을 하도록 만드는 것이 아니라, **소수 노드에 처리 부담이 과도하게 집중되지 않도록 조절**하는 것을 의미한다.

### 7.2 Specialization
각 노드는 서로 다른 기능적 역할을 담당할 수 있어야 한다. 따라서 HDN은 균형화를 추구하되, 모든 노드를 똑같이 만드는 것이 아니라 **전문화된 표현의 분화**를 유지한다.

### 7.3 Development
신경망의 은닉 구조를 미리 완성된 형태로 두지 않고, 학습 중에 점진적으로 형성되는 것으로 본다.

---

## 8. 주요 연산
HDN은 학습 도중 다음 네 가지 구조 변화를 수행한다.

### Birth
새로운 hidden node 또는 representation을 생성한다.

### Split
과부하된 노드를 둘 이상의 자식 노드로 분화시켜 기능을 분리한다.

### Deepen
기존 병렬 확장으로 해결되지 않는 경우 새로운 hidden stage를 추가한다.

### Prune
거의 사용되지 않거나, 성능 기여가 작거나, 다른 노드와 지나치게 유사한 구조를 제거한다.

---

## 9. 검증할 연구 가설

### H1
고정 구조 모델 대비, HDN은 더 작은 모델 크기로 유사하거나 더 높은 성능을 낼 수 있다.

### H2
단순 grow-prune 방식보다 HDN은 노드 간 처리 부담 편중을 더 잘 줄인다.

### H3
이러한 homeostatic balancing은 특정 노드 제거나 입력 corruption 상황에서 더 높은 강건성으로 이어진다.

### H4
Birth, Split, Deepen, Prune의 통합은 각각을 단독으로 사용하는 것보다 더 좋은 구조-성능 균형을 만든다.

---

## 10. 실험 계획

### 10.1 메커니즘 검증
작은 toy dataset에서 구조 생성과 분화가 실제로 일어나는지 시각화한다.

예시:
- XOR
- Two Moons
- Concentric Circles

### 10.2 메인 벤치마크
다음 데이터셋에서 정확도와 효율을 검증한다.

- MNIST / Fashion-MNIST
- CIFAR-10
- CIFAR-100
- 필요 시 Tiny-ImageNet

### 10.3 강건성 평가
다음 시나리오에서 robustness를 측정한다.

- Common corruption benchmark
- busiest-node ablation
- random-node ablation
- route / layer perturbation

### 10.4 어블레이션
다음 구성 요소를 제거해 비교한다.

- no balance
- no split
- no deepen
- no prune
- split criterion 변경
- prune criterion 변경

---

## 11. 주요 평가 지표

### 성능
- Accuracy
- Loss
- Calibration

### 효율
- Total parameters
- Active parameters
- FLOPs
- Average used depth

### 균형성
- Utilization coefficient of variation
- Gini coefficient
- Dead-node ratio
- Busiest-node load share

### 강건성
- Corruption accuracy
- Node ablation drop
- Seed variance

### 구조 해석성
- 학습 중 노드 수 변화
- 층 수 변화
- split / prune 이벤트 수
- specialization map

---

## 12. 구현 방향
초기 구현은 **PyTorch 기반**으로 진행한다.

추천 운영 순서:
1. toy dataset에서 birth/split/prune 메커니즘 검증
2. MNIST에서 전체 파이프라인 통합
3. CIFAR-10에서 baseline 비교
4. CIFAR-100 및 robustness 실험 확장

코드 구조는 다음과 같이 관리한다.

```text
hdn/
  README.md
  configs/
  src/
  notebooks/
  checkpoints/
  logs/
  results/
```

---

## 13. 하드웨어 및 실행 전략
초기 프로토타입은 **Colab**에서도 가능하다.

- toy / MNIST / CIFAR-10 초기 실험: Colab 가능
- CIFAR-100 / corruption / 반복 실험: 유료 Colab 또는 외부 GPU 권장
- 최종 논문표 재현: 로컬 GPU 또는 클라우드 VM 권장

권장 VRAM 기준:
- 최소: 16GB
- 권장: 24GB
- 여유: 32GB

---

## 14. 기대 효과

### 학술적 기대 효과
- 성장형 구조, neuron split, adaptive depth, pruning, load balancing을 하나의 프레임으로 통합
- 구조 학습과 강건성 사이의 관계를 새로운 방식으로 제시
- 발달형 신경망이라는 해석 가능한 연구 서사 제공

### 실용적 기대 효과
- 작은 seed network에서 시작해 자동으로 구조를 찾는 효율적 학습
- 특정 노드 과의존 감소를 통한 강건성 향상
- compact하고 adaptive한 모델 설계 가능성

---

## 15. 현재 단계와 다음 할 일
현재 프로젝트는 **아이디어 정리와 실험 설계가 완료된 상태**이며, 다음 단계는 첫 구현을 만드는 것이다.

### 바로 해야 할 일
1. toy dataset용 baseline 코드 작성
2. birth / split / prune 이벤트 로깅 시스템 구축
3. MNIST 통합 실험 실행
4. CIFAR-10 baseline 및 어블레이션 비교
5. README와 실험 로그 포맷 정리

---

## 16. 요약
HDN은 신경망을 처음부터 완성된 구조로 두지 않고, 학습 과정에서 내부 표현이 생성·분화·심화·소거되도록 만드는 **발달형 신경망 프레임워크**이다. 이 프로젝트의 핵심 목표는 **노드 간 처리 부담의 편중을 줄이면서도 기능적 분화를 유지하여**, 더 작고 더 강건한 모델을 만드는 데 있다.
