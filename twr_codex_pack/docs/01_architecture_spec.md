# TWR-LM Architecture Spec

## Pipeline
Input Tokens -> Event Encoder -> Sequential Write -> Token Removal -> Think Phase -> Readout

## Event Encoder
Each token x_t with positional signal p_t is transformed into an event vector e_t.

Suggested v1:
- token embedding
- positional embedding or positional signal
- 1-layer MLP or linear projection + activation

## Latent Memory
Persistent model state after write:
- memory tensor shape: `[batch, slots, dim]`
- default slots: 16
- default dim: 128

## Sequential Soft Write
For each event:
1. Compute write attention over slots.
2. Project event to slot update value.
3. Add update into memory.

Design intent:
- simple
- differentiable
- easy to inspect

## Tokenless Transition
After the final event is written:
- original token representations should not be used in the TWR forward path
- only the memory tensor may continue to the think phase

## Think Phase
Repeatedly refine latent memory.

Update template:
- normalize memory
- mix across slots
- apply gated MLP
- gate residual update

## Adaptive Compute
### Step gate
A scalar gate per think step determines effective depth.

### Slot gate
A slot-wise gate determines which memory slots receive stronger updates.

## Readout
Pool over final memory, then apply a linear prediction head.

## v1 default spec
- embed_dim: 128
- slots: 16
- slot_dim: 128
- think_steps_max: 4
- refine: low-rank slot mixer + GLU MLP
- readout: mean-pool + linear
