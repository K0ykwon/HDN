# AGENTS.md

This file is preserved as part of the historical `twr_codex_pack`.

## Important Note

The instructions in this folder were written for the original `write -> think -> read` architecture with slot-gated latent memory.

That is no longer the current mainline implementation in this repository.

## Current Reality

- The repository now uses a lean tokenless latent backbone.
- Persistent post-compression state lives in latent space, not token space.
- The active model is built around `latent_encoder.py` and `latent_backbone.py`.

Treat the rest of `twr_codex_pack/` as archived design material.
