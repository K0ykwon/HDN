# Paper Benchmark And Experiment Plan

## Goal

Position TWR-LM as a `token-free`, `attention-free`, latent-backbone alternative to Transformer baselines after compression.

The paper should avoid over-claiming. The current codebase is strongest as a long-context classification and retrieval-style backbone, not yet as a general generative LM. The benchmark plan should therefore be split into:

1. a main paper track that the current repository can support with limited architectural changes
2. an extension track for generation and broader long-context reasoning once task heads are expanded

## Recommended Paper Claim

The safest paper claim is:

`A token-free, attention-free latent backbone can approach or exceed compact Transformer baselines on selected long-context classification and retrieval-style benchmarks, with competitive parameter efficiency, provided compression and latent hierarchy are tuned carefully.`

This is a better fit than a broad claim of replacing Transformers for all long-context generation tasks.

## Benchmark Tiers

### Tier 1: Main Paper Benchmarks

These should be the core benchmark suite for the first paper version because they align with the current codebase.

1. `ListOps`
Purpose: compositional structure sensitivity under compressed latent processing.
Status: already implemented locally.

2. `RULER`
Purpose: long-context recall plus synthetic long-range reasoning beyond trivial needle retrieval.
Source: [NVIDIA/RULER](https://github.com/NVIDIA/RULER), [paper](https://arxiv.org/abs/2404.06654)
Status: already implemented in the repo in simplified form.

3. `LongBench`
Purpose: realistic long-context multitask evaluation.
Source: [THUDM/LongBench repo](https://github.com/THUDM/LongBench), [ACL 2024 paper](https://aclanthology.org/2024.acl-long.172)
Status: repo currently uses `trec`; paper version should expand beyond a single subset.

4. `Hyperpartisan`
Purpose: long-document classification with real text.
Status: already implemented locally.

### Tier 2: Strong Additions For Long-Context Paper Credibility

These are high-value additions because they directly test long-range reasoning headroom.

1. `BABILong`
Purpose: reasoning-in-a-haystack rather than simple retrieval.
Source: [paper](https://arxiv.org/abs/2406.10149), [repo](https://github.com/booydar/babilong)
Why it matters: this is a strong fit for the repository's stated concern about long-context headroom.

2. `InfiniteBench`
Purpose: evaluation beyond 100K tokens with standardized long-context tasks.
Source: [OpenBMB/InfiniteBench](https://github.com/OpenBMB/InfiniteBench), [paper](https://arxiv.org/abs/2402.13718)
Why it matters: this is a direct stress test for whether the latent backbone scales better than token-persistent baselines.

3. `LongBench v2`
Purpose: harder realistic long-context reasoning across real-world tasks.
Source: [LongBench v2 repo](https://github.com/THUDM/LongBench), [paper](https://arxiv.org/abs/2412.15204)
Why it matters: better fit than original LongBench when the paper wants to argue deeper reasoning rather than shallow benchmark matching.

### Tier 3: Retrieval / Code Understanding

These are useful if the paper wants to broaden the claim beyond classification.

1. `RepoQA`
Purpose: long-context code understanding and retrieval.
Source: [RepoQA repo](https://github.com/evalplus/repoqa), [paper](https://arxiv.org/abs/2406.06025)
Why it matters: useful for showing the latent backbone can search and preserve precise information in large contexts.

2. `BRIGHT` long-context retrieval setting
Purpose: reasoning-intensive retrieval with long documents.
Source: [paper](https://arxiv.org/abs/2407.12883)
Why it matters: stronger retrieval benchmark than simple needle tests.

### Tier 4: Generation Extension Track

These should not be in the first main paper unless the repository gains a true generation head and decoding pipeline.

1. `LongBench-Write`
2. `LongWrite-Ruler`
Source: [THUDM/LongWriter repo](https://github.com/THUDM/LongWriter)

These are valuable once TWR-LM has a generation-oriented readout rather than only a classifier.

## Proposed Experimental Structure

### Section A: Core Benchmark Comparison

Compare:

- `TWR-LM lean`
- `TWR-LM`
- `Transformer baseline`
- `Mamba baseline`

For each benchmark report:

- validation / test accuracy or task score
- parameter count
- throughput
- peak memory
- effective latent depth
- active latent count

### Section B: Compression Bottleneck Study

This is the most important architecture study for the current repo.

Ablations:

- window size sweep
- stride / overlap sweep
- number of readout queries
- latent dimension sweep
- think step sweep

Primary question:

`How much accuracy is lost when tokens are compressed into latent windows, and which compression settings recover the most signal without breaking parameter efficiency?`

### Section C: Merge / Hierarchy Study

Ablations:

- fixed-depth hierarchy vs adaptive latent hierarchy
- merge-depth sweep
- readout from final level only vs multiscale readout

Primary question:

`Is the current forced hierarchical reduction too destructive, and how much does multiscale latent reading compensate for it?`

### Section D: Length Generalization Study

Train at one context length and test at longer lengths.

Suggested protocol:

- train at `256`
- test at `512`, `1k`, `2k`, and if feasible higher

Run this on:

- RULER
- BABILong
- InfiniteBench-style synthetic subsets

Primary question:

`Does a token-free latent backbone extrapolate in length better than standard token-persistent baselines at similar parameter budgets?`

### Section E: Efficiency Study

Report:

- examples/sec
- tokens/sec if possible
- peak memory
- parameter count
- approximate FLOPs

And compare:

- short-context throughput
- long-context throughput
- performance under equal-parameter and equal-throughput budgets

This section is necessary because current results show that smaller parameter count alone does not guarantee practical speed.

## Main Paper Table Plan

Recommended core tables:

1. `Main benchmark table`
Columns:
benchmark, TWR-lean, TWR, Transformer, Mamba, params, throughput

2. `Compression ablation table`
Columns:
window size, stride, latent count, accuracy, throughput

3. `Length extrapolation table`
Columns:
train length, test length, TWR score, Transformer score, Mamba score

4. `Efficiency table`
Columns:
model, params, throughput, memory, score

## Must-Have Figures

1. accuracy vs throughput
2. accuracy vs parameter count
3. performance vs context length
4. effective depth / active latent usage by task

## Suggested Acceptance Criteria

The first paper version is strong enough if it can show most of the following:

- TWR-lean is competitive with Transformer on the current four-task suite
- TWR-lean wins at least one long-context benchmark at equal or smaller parameter count
- TWR retains reasonable performance as context grows
- the compression and hierarchy ablations explain where gains and losses come from
- efficiency is honestly reported, even when TWR loses on runtime

## Practical Next Steps In This Repo

1. Expand `LongBench` from only `trec` to a broader subset.
2. Add one stronger long-range reasoning benchmark: `BABILong` or `InfiniteBench`.
3. Add one retrieval-oriented benchmark: `RepoQA` or `BRIGHT`.
4. Keep the first paper classification-and-retrieval centered.
5. Defer generation claims until a generation-specific readout is implemented.

## Sources

- LongBench: [ACL 2024 paper](https://aclanthology.org/2024.acl-long.172), [repo](https://github.com/THUDM/LongBench)
- LongBench v2: [paper](https://arxiv.org/abs/2412.15204), [repo](https://github.com/THUDM/LongBench)
- RULER: [paper](https://arxiv.org/abs/2404.06654), [repo](https://github.com/NVIDIA/RULER)
- BABILong: [paper](https://arxiv.org/abs/2406.10149), [repo](https://github.com/booydar/babilong)
- InfiniteBench: [paper](https://arxiv.org/abs/2402.13718), [repo](https://github.com/OpenBMB/InfiniteBench)
- RepoQA: [paper](https://arxiv.org/abs/2406.06025), [repo](https://github.com/evalplus/repoqa)
- BRIGHT: [paper](https://arxiv.org/abs/2407.12883)
- LongWriter / LongBench-Write: [repo](https://github.com/THUDM/LongWriter)
