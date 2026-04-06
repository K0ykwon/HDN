# Mamba Comparison Plan

## What Mamba Evaluated

Based on the official Mamba paper and repository, the evaluation strategy was broader than a single benchmark family.

### 1. Large-scale language modeling

The Mamba paper presents itself as a general sequence backbone and highlights large-scale language-model pretraining as a primary evaluation axis. The abstract states that:

- Mamba achieves strong language modeling performance
- the `3B` model outperforms same-size Transformers
- it matches Transformers roughly twice its size in pretraining and downstream evaluation

Source:

- Mamba paper abstract: https://arxiv.org/abs/2312.00752

### 2. Zero-shot downstream language evaluation

The official repository documents the zero-shot downstream tasks used for the paper's language evaluation table. The README points to `lm-evaluation-harness` and lists:

- `lambada_openai`
- `hellaswag`
- `piqa`
- `arc_easy`
- `arc_challenge`
- `winogrande`
- `openbookqa`

Source:

- Official repository evaluation section: https://github.com/state-spaces/mamba

### 3. Throughput / inference efficiency

The paper abstract explicitly claims fast inference and reports about `5x` higher throughput than Transformers together with linear scaling in sequence length.

Source:

- Mamba paper abstract: https://arxiv.org/abs/2312.00752

### 4. Very long-sequence scaling

The abstract also states that performance improves on real data up to million-length sequences. This matters because Mamba’s core claim is not just accuracy at one context length, but practical scaling as length increases.

Source:

- Mamba paper abstract: https://arxiv.org/abs/2312.00752

### 5. Cross-modality validation

The abstract says Mamba demonstrates strong results across:

- language
- audio
- genomics

This is important because the Mamba paper does not argue only from synthetic long-context benchmarks; it argues from multiple modalities.

Source:

- Mamba paper abstract: https://arxiv.org/abs/2312.00752

## What This Means For TWR-LM

If TWR-LM is intended to be compared seriously against Transformer and Mamba in a paper, then a four-benchmark snapshot is not enough.

Mamba’s evaluation logic was roughly:

1. show backbone quality on large-scale sequence modeling
2. show downstream transfer / zero-shot utility
3. show throughput and scaling advantages
4. show robustness across more than one modality or task family

TWR-LM currently has partial evidence for only part of this story:

- strong small-model benchmark competitiveness on several long-context classification tasks
- parameter efficiency
- a clean `token-free`, `attention-free` identity after compression

But it does not yet have:

- language-model style pretraining evaluation
- broad downstream transfer evaluation
- convincing runtime advantage
- multimodal evidence

So the benchmark plan for TWR should be designed around proving the advantages that TWR can honestly claim now.

## TWR Advantages That Need To Be Tested

The useful practical claims for TWR-LM are:

1. it can remain competitive without persistent token states after compression
2. it can stay parameter-efficient relative to Transformer baselines
3. it may preserve long-context reasoning or retrieval performance with a smaller persistent backbone state
4. it may offer better length scaling headroom if the latent hierarchy is tuned well

The benchmark design should test exactly those claims, not generic SOTA aspirations.

## Recommended Benchmark Design

### Track A: Mainline Backbone Paper

This is the first paper track that best matches the current codebase.

Benchmarks:

1. `ListOps`
Why:
- tests compositional structure under compressed latent processing
- sensitive to early compression damage

2. `RULER`
Why:
- tests long-context retrieval and synthetic long-range reasoning
- directly relevant to whether the latent backbone preserves sparse important information

Source:
- https://github.com/NVIDIA/RULER
- https://arxiv.org/abs/2404.06654

3. `LongBench`
Why:
- tests realistic long-context NLP tasks
- should be expanded beyond only `trec`

Source:
- https://github.com/THUDM/LongBench
- https://aclanthology.org/2024.acl-long.172

4. `Hyperpartisan`
Why:
- real long-document classification
- tests whether the architecture works outside synthetic settings

### Track B: Long-Context Stress Tests

These are necessary if the paper wants to argue that the latent backbone has headroom beyond the current four tasks.

1. `BABILong`
Why:
- stronger reasoning-in-context test than simple retrieval
- useful for checking whether compression destroys multistep evidence chains

Source:
- https://github.com/booydar/babilong
- https://arxiv.org/abs/2406.10149

2. `InfiniteBench`
Why:
- designed for very long contexts
- useful for length extrapolation and scaling claims

Source:
- https://github.com/OpenBMB/InfiniteBench
- https://arxiv.org/abs/2402.13718

### Track C: Retrieval / Precision Preservation

This track tests a practical strength TWR should have if latent compression is working well: preserving the right facts, not all tokens.

1. `RepoQA`
Why:
- tests code retrieval / precise long-context lookup
- a strong fit for “compressed persistent state but preserved salient information”

Source:
- https://github.com/evalplus/repoqa
- https://arxiv.org/abs/2406.06025

2. `BRIGHT`
Why:
- retrieval with more reasoning pressure than simple needle tasks

Source:
- https://arxiv.org/abs/2407.12883

## Experimental Matrix

For every benchmark, compare:

- `TWR lean`
- `TWR`
- `Transformer`
- `Mamba`

Report:

- task score
- parameter count
- throughput
- peak memory
- effective depth
- active latent count

## Critical Ablations

These ablations are necessary because they connect directly to TWR’s claimed advantages.

### Compression Study

Measure:

- window size
- stride
- latent dimension
- readout query count

Question:

`How much information is lost at compression, and what settings recover the most signal without destroying efficiency?`

### Hierarchy Study

Measure:

- think step count
- multiscale readout vs final-level-only readout

Question:

`Is the latent hierarchy actually helping, or is the model mostly winning from the encoder and classifier?`

### Length Extrapolation Study

Protocol:

- train on one context length
- evaluate on longer lengths

Suggested:

- train at `256`
- test at `512`, `1k`, `2k`, and higher if feasible

Question:

`Does the latent backbone degrade more gracefully than Transformer and Mamba as sequence length grows?`

### Efficiency Study

Measure:

- examples/sec
- memory
- approximate FLOPs
- performance under equal-parameter and equal-throughput budgets

Question:

`Does TWR’s smaller persistent latent state translate into practical efficiency, or only into parameter savings?`

## Concrete Paper Sections

### Section 1: Backbone Competitiveness

Use the current four-task suite plus at least one stronger long-context benchmark.

### Section 2: Compression Bottleneck

This should be a major section, not a small appendix, because it is the central architectural risk in TWR.

### Section 3: Long-Context Scaling

This section is the closest analogue to Mamba’s long-sequence claim.

### Section 4: Efficiency

TWR should be evaluated honestly here.
Right now, TWR’s strongest advantage is parameter efficiency, not throughput.
That distinction should be explicit in the paper.

## Recommended Paper Claim Versus Mamba

Do not frame the paper as:

`TWR is better than Mamba everywhere.`

Frame it as:

`TWR explores a different scaling strategy: instead of linear recurrent token processing, it compresses into a persistent latent backbone and performs token-free, attention-free hierarchical refinement.`

Then the empirical question becomes:

- when is compressed latent persistence better than token persistence?
- when is it more parameter-efficient?
- when does it fail?

That is a much stronger and more defensible paper framing.
