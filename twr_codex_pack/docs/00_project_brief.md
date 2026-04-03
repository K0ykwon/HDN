# TWR-LM Project Brief

## Name
TWR-LM: Tokenless Write-Think-Read Latent Machine

## One-sentence definition
A tokenless sequence backbone that writes input events into a small latent memory, discards token representations after the write phase, performs iterative computation only inside the memory, and reads outputs from the refined memory.

## Core idea
The model should not treat token-wise hidden states as the long-lived computational substrate. Instead, it should compress incoming information into a small persistent latent memory and conduct downstream reasoning within that bottleneck.

## Why this matters
This changes the default unit of persistence from token states to memory slots. That shift makes it possible to study:
- compressed long-context reasoning
- adaptive compute inside the bottleneck
- tokenless state transition dynamics
- efficiency/quality trade-offs against standard token-persistent models

## v1 success criteria
- End-to-end training works.
- Tokenless write-think-read separation is preserved in code.
- Adaptive think depth can be measured.
- Slot usage can be measured.
- Baseline comparison is possible through one training stack.
