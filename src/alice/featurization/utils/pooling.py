"""Module for pooling embeddings for LLMFeaturizer"""
import torch


def average_pool(
    last_hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """    
    This function pools a sequence of contextualized token embeddings into a single
    vector by averaging all non-padding token embeddings.
    """
    last_hidden = last_hidden_states.masked_fill(
        ~attention_mask[..., None].bool(), 0.0
    )
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def last_token_pool(
    last_hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    left_padding: bool = False,
) -> torch.Tensor:
    """
    Extract the embedding of the last real token in each sequence.
    """
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    return last_hidden_states[
        torch.arange(last_hidden_states.size(0)), sequence_lengths
    ]


def weighted_average_pool(
    last_hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """    
    This function pools a sequence of contextualized token embeddings by computing
    a weighted average where tokens at later positions receive linearly increasing
    weights. Padding tokens are excluded from the computation.
    """
    seq_length = last_hidden_states.size(1)
    weights = (
        torch.arange(1, seq_length + 1, dtype=torch.float32)
        .unsqueeze(0)
        .to(last_hidden_states.device)
    )

    weighted_mask = weights * attention_mask.float()
    weighted_hidden_states = last_hidden_states * weighted_mask.unsqueeze(-1)

    sum_weighted_embeddings = torch.sum(weighted_hidden_states, dim=1)
    sum_weights = torch.sum(weighted_mask, dim=1, keepdim=True).clamp(min=1)

    weighted_average_embeddings = sum_weighted_embeddings / sum_weights

    return weighted_average_embeddings