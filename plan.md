# HDN Implementation Plan

## 1. Current Status

- Repository currently contains concept documents only.
- Core research idea and experiment direction are defined in `HDN_project_overview.md`.
- Implementation, configs, training pipeline, and experiment assets are not created yet.

## 2. Project Goal

Build a PyTorch-based HDN (Homeostatic Developmental Network) prototype that:

- starts from a minimal input-to-output structure,
- grows hidden representations only when needed,
- splits overloaded nodes,
- adds depth when width expansion is insufficient,
- prunes redundant or low-utility structure,
- and validates whether this developmental process improves efficiency and robustness.

## 3. Development Principles

- Implement the smallest end-to-end version first.
- Verify each structural operation independently before integrating all of them.
- Treat logging and visualization as first-class features, not optional extras.
- Compare against fixed-architecture baselines at every stage.

## 4. Recommended Repository Structure

```text
HDN/
  configs/
  src/
    data/
    models/
    training/
    metrics/
    utils/
  scripts/
  notebooks/
  tests/
  logs/
  results/
```

## 5. Phase Plan

### Phase 1. Project Bootstrap

Objective: create a runnable training scaffold.

Tasks:
- initialize package structure under `src/`
- add environment/dependency definition
- add base training loop, config loader, logging utilities
- add simple MLP baseline for comparison
- define common experiment output format for logs and results

Deliverables:
- minimal train script
- baseline model training on a toy dataset
- reproducible config-driven run flow

### Phase 2. Toy Dataset Mechanism Validation

Objective: confirm that growth events occur and can be observed.

Datasets:
- XOR
- Two Moons
- Concentric Circles

Tasks:
- implement toy dataset loaders
- implement Birth mechanism
- implement Split mechanism
- implement initial Prune mechanism
- log structural events with timestamps/steps
- visualize node count and decision boundary changes

Deliverables:
- successful toy runs
- plots for structure evolution
- event logs showing birth/split/prune behavior

### Phase 3. Core HDN Integration

Objective: unify the developmental loop into one training framework.

Tasks:
- define HDN model state representation
- implement structural scheduler for `Birth -> Split -> Deepen -> Prune`
- add criteria modules for:
  - residual-driven birth
  - gradient/load heterogeneity split
  - demand-driven deepen
  - usage/redundancy/contribution prune
- ensure optimizer state can survive structural mutation
- add checkpoint save/load support for dynamic architectures

Deliverables:
- integrated HDN training pipeline
- stable dynamic-structure checkpointing
- documented event criteria and thresholds

### Phase 4. MNIST End-to-End Validation

Objective: validate the full pipeline on the first real benchmark.

Tasks:
- add MNIST and Fashion-MNIST data pipeline
- compare against fixed MLP/CNN baselines
- collect metrics for accuracy, params, active depth, utilization balance
- evaluate whether dynamic growth remains stable across seeds

Deliverables:
- baseline vs HDN comparison table
- seed variance summary
- first usable default configuration

### Phase 5. CIFAR-10 Expansion

Objective: test whether HDN scales beyond toy/simple image tasks.

Tasks:
- adapt backbone and structural ops for higher-dimensional inputs
- compare with fixed-capacity baselines on CIFAR-10
- measure efficiency/accuracy trade-offs
- inspect whether deepen events become more important than width-only growth

Deliverables:
- CIFAR-10 benchmark results
- growth pattern analysis by training stage

### Phase 6. Robustness and Ablation

Objective: validate the homeostasis claim.

Tasks:
- implement busiest-node ablation
- implement random-node ablation
- evaluate corruption robustness
- run ablations:
  - no balance
  - no split
  - no deepen
  - no prune
  - alternative split/prune criteria

Deliverables:
- robustness report
- ablation result table
- evidence for or against homeostatic balancing benefits

## 6. Technical Priorities

Priority order:
1. Baseline training scaffold
2. Toy dataset support
3. Birth/Split/Prune event logging
4. Integrated HDN state mutation logic
5. MNIST benchmark
6. CIFAR-10 and robustness expansion

## 7. Key Metrics

- Performance: accuracy, loss, calibration
- Efficiency: total parameters, active parameters, FLOPs, average used depth
- Balance: utilization CV, Gini coefficient, dead-node ratio, busiest-node load share
- Robustness: corruption accuracy, node ablation drop, seed variance
- Structure: node count over time, depth over time, event counts, specialization map

## 8. Immediate Next Actions

1. Create base project directories and dependency setup.
2. Implement a minimal training loop plus toy dataset loaders.
3. Add an MLP baseline and experiment logging format.
4. Implement the first dynamic operation set: Birth, Split, Prune.
5. Run toy experiments and verify that structure changes are observable.

## 9. Risks and Open Questions

- Dynamic architecture mutation may break optimizer state handling.
- Split/deepen/prune criteria may be sensitive to thresholds and training noise.
- Robustness gains are still a hypothesis and must be validated experimentally.
- CIFAR-scale experiments may require architecture constraints to keep growth stable.
- Checkpoint compatibility for changing structures should be designed early.

## 10. Definition of Initial Success

The first milestone is complete when:

- toy datasets train end-to-end,
- at least one run shows meaningful structural growth,
- event logs and plots explain when structure changed,
- and MNIST integration can be started on top of the same training framework.
