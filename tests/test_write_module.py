import torch

from src.twr.modules.sequential_write import SequentialSoftWrite


def test_sequential_write_preserves_shapes() -> None:
    writer = SequentialSoftWrite(event_dim=16, slot_dim=16, slots=4)
    memory = torch.zeros(2, 4, 16)
    events = torch.randn(2, 6, 16)
    updated, stats = writer(memory, events)
    assert updated.shape == memory.shape
    assert stats.attention.shape == (2, 6, 4)
    assert stats.slot_histogram.shape == (4,)
