# Experiment Summary

This report reflects the current benchmark snapshot after moving the mainline TWR implementation to a lean latent-backbone architecture.

## Three-Way Comparison

| task | model | acc | params | throughput |
| --- | --- | ---: | ---: | ---: |
| ListOps | TWR lean | 0.2285 | 82,587 | 11,837.5 |
| ListOps | Transformer | 0.2383 | 86,026 | 63,946.9 |
| ListOps | Mamba | 0.1289 | 95,530 | 58,591.7 |
| RULER needle | TWR lean | 0.0840 | 100,065 | 11,611.2 |
| RULER needle | Transformer | 0.0801 | 117,136 | 63,654.0 |
| RULER needle | Mamba | 0.0781 | 252,048 | 52,911.8 |
| LongBench TREC | TWR lean | 0.1800 | 494,083 | 2,221.4 |
| LongBench TREC | Transformer | 0.1800 | 627,250 | 14,109.4 |
| LongBench TREC | Mamba | 0.1800 | 1,272,242 | 11,649.8 |
| Hyperpartisan | TWR lean | 0.5400 | 475,283 | 2,923.3 |
| Hyperpartisan | Transformer | 0.5450 | 624,130 | 16,139.0 |
| Hyperpartisan | Mamba | 0.5150 | 1,266,050 | 13,140.7 |

## Takeaways

- Transformer is still the strongest overall baseline.
- Lean TWR is close on `ListOps` and `Hyperpartisan` while using fewer parameters than Transformer.
- Lean TWR keeps a small edge on `RULER needle`, but not the larger advantage seen in the earlier large TWR backbone.
- On `LongBench TREC`, all three models are tied in the current setup.
- Mamba is not competitive on any of the four current benchmarks in this repository.

## TWR Full vs Lean

| task | TWR lean acc | TWR full acc | comment |
| --- | ---: | ---: | --- |
| ListOps | 0.2285 | 0.2246 | lean is slightly better with far fewer parameters |
| RULER needle | 0.0840 | 0.1133 | lean loses long-context headroom |
| LongBench TREC | 0.1800 | 0.2000 | lean gives up the earlier TWR lead |
| Hyperpartisan | 0.5400 | 0.5175 | lean is better and more efficient |

## Current Recommendation

Use the lean TWR backbone as the default research branch for parameter-efficient comparison, and treat the large TWR configs as long-context reference points rather than the default mainline.
