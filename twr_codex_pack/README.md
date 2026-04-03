# TWR-LM Codex Pack

This folder contains a ready-to-use prompt and markdown planning pack for using Codex to build the TWR-LM experiment codebase.

## Files
- `AGENTS.md` — repository rules and non-negotiable architecture constraints
- `codex_master_prompt.md` — main prompt to paste into Codex
- `prompts/` — phase-by-phase prompts for incremental implementation
- `docs/` — architecture, experiment, ablation, evaluation, and execution docs

## Recommended usage
1. Give Codex `AGENTS.md` and `codex_master_prompt.md` first.
2. Then use the numbered prompts in `prompts/` one by one.
3. Keep the docs in `docs/` inside the repo so Codex has stable context.

## Suggested workflow
- Start from `prompts/01_bootstrap_repo.md`
- Then `prompts/02_impl_core_twr.md`
- Then `prompts/03_add_training_and_logging.md`
- Then baselines and ablations
