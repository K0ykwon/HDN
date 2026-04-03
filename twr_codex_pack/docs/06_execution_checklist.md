# Execution Checklist

## Stage 0
- [ ] Create repo skeleton
- [ ] Add dependencies
- [ ] Add config loader
- [ ] Add training entrypoint

## Stage 1
- [ ] Implement event encoder
- [ ] Implement latent memory init
- [ ] Implement sequential soft write
- [ ] Implement refine block
- [ ] Implement think loop
- [ ] Implement readout head
- [ ] Implement TWR wrapper model
- [ ] Verify single forward pass

## Stage 2
- [ ] Implement trainer
- [ ] Implement evaluator
- [ ] Log loss and accuracy
- [ ] Log throughput and GPU memory
- [ ] Log effective depth and slot usage
- [ ] Run one debug experiment

## Stage 3
- [ ] Add Transformer baseline
- [ ] Add Perceiver-style baseline
- [ ] Unify trainer across models

## Stage 4
- [ ] Add ablation configs
- [ ] Add no_think
- [ ] Add no_slot_gate
- [ ] Add fixed_think vs adaptive_depth
- [ ] Add capacity sweeps

## Stage 5
- [ ] Add tests
- [ ] Clean README
- [ ] Save reproducible run commands
- [ ] Export final summary tables
