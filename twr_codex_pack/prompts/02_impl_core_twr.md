# Prompt 02 — Implement Core TWR-LM

Implement the smallest end-to-end TWR-LM model that can train on a classification task.

## Required modules
- event encoder
- latent memory initializer
- sequential soft write
- refine block
- think loop
- readout head
- full TWR model wrapper

## Required behavior
- Input tokens -> event vectors
- Events sequentially update memory
- Token representations are not used after write
- Think phase operates only on memory
- Readout uses pooled final memory

## v1 defaults
- embed dim 128
- memory slots 16
- slot dim 128
- max think steps 4
- low-rank slot mixer + GLU MLP
- scalar step gate
- slot-wise gate

## Also do
- Add shape assertions.
- Add docstrings.
- Add a tiny smoke test or example forward pass.

## Deliverable
A trainable model implementation with a clean forward API and config-driven construction.
