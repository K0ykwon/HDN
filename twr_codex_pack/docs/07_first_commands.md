# First Commands to Ask Codex to Run

## 1. Bootstrap the repo
"Read AGENTS.md and codex_master_prompt.md, then scaffold the repository skeleton, config files, and a single training entrypoint."

## 2. Implement core model
"Implement the smallest trainable TWR-LM with sequential soft write, a simple think loop, and pooled readout. Preserve strict tokenless behavior after the write phase."

## 3. Add trainer
"Add a config-driven trainer with JSON logging, checkpointing, seed control, and basic accuracy/loss metrics."

## 4. Add analysis hooks
"Log average effective depth, step-gate statistics, and average active slot usage for every validation run."

## 5. Add baselines
"Implement Transformer encoder and Perceiver-style latent baselines under the same trainer and config interface."

## 6. Add ablations
"Add config-driven ablations for no_think, fixed_think, adaptive_depth, and no_slot_gate with minimal code duplication."
