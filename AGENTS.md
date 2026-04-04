# AGENTS.md

## Project
TWR-LM

## Current Goal
Build and evaluate a `tokenless latent backbone` that can compete with Transformer baselines while keeping the persistent post-compression state in latent space rather than token space.

## Current Mainline Architecture
1. Encode tokens into overlapping latent windows.
2. Discard token-wise persistent states after compression.
3. Run shared latent refinement and hierarchical merge only on latent representations.
4. Read outputs with multiscale learned queries.
5. Optimize for parameter efficiency, reproducible benchmarking, and baseline comparison.

## Non-Negotiable Constraints
- Do not keep token-wise hidden states as the persistent backbone state after compression.
- Keep the codebase experiment-first.
- Prefer small, composable PyTorch modules.
- Favor reproducible configs, benchmark comparability, and simple ablations over speculative complexity.
- Treat the old write/think/slot-gate design as historical, not as the default implementation target.
