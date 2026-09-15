"""Helper functions for BaseDataModule"""
import torch


def find_duplicates(x: torch.Tensor):
    """Return indices of duplicate rows, keeping the first occurrence of each."""
    _, inv, counts = torch.unique(x, return_inverse=True, return_counts=True, dim=0)
    indices_to_delete = [
        index
        for i, c in enumerate(counts)
        if c > 1
        for index in torch.where(inv == i)[0][1:].tolist()
    ]
    return indices_to_delete


def find_nan_rows(x: torch.Tensor):
    """Return indices of rows containing any NaN value."""
    mask = torch.isnan(x).any(dim=1)
    indices_to_delete = mask.nonzero().flatten().tolist()
    return indices_to_delete
