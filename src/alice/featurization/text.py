"""Module for getting huggingface models and text embeddings"""
from dataclasses import dataclass
import os
from typing import Optional, Any
import torch

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
torch.cuda.empty_cache()

from torch.nn.utils.rnn import pad_sequence

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    T5EncoderModel,
    T5Config,
    Qwen2Model,
    Qwen2Config,
    BartModel,
    BartConfig
)


@dataclass
class ModelConfig:
    """Configuration entry for a supported transformer model.

    Maps a model name to its corresponding HuggingFace config class,
    model class, and dropout.
    """
    name: str
    config_class: Optional[Any] = None
    model_class: Optional[Any] = None
    dropout_field: str = "dropout_rate"


class BartEncoderModel:
    """Factory class that loads only the encoder from a pretrained BART model."""
    @classmethod
    def from_pretrained(cls, model_name, config=None):
        if config is not None:
            full_model = BartModel.from_pretrained(model_name, config=config)
        else:
            full_model = BartModel.from_pretrained(model_name)
        return full_model.encoder


MODEL_CONFIGS = {
    "t5-base": ModelConfig("t5-base", T5Config, T5EncoderModel),
    "t5-small": ModelConfig("t5-small", T5Config, T5EncoderModel),
    "GT4SD/multitask-text-and-chemistry-t5-base-augm": ModelConfig(
        "GT4SD/multitask-text-and-chemistry-t5-base-augm",
        T5Config,
        T5EncoderModel,
    ),
    "Qwen/Qwen2.5-0.5B": ModelConfig(
        "Qwen/Qwen2.5-0.5B", Qwen2Config, Qwen2Model, "attention_dropout"
    ),
    "facebook/bart-base": ModelConfig(
        "facebook/bart-base", BartConfig, BartEncoderModel, "attention_dropout"
    ),
}


def get_model_and_tokenizer(model_name: str, device: str = 'cuda'):
    """Load a pretrained model and tokenizer.

    Uses MODEL_CONFIGS for architecture-specific loading when available,
    otherwise falls back to AutoModel/AutoConfig.

    Args:
        model_name: HuggingFace model identifier.
        device: Device to load the model on.

    Returns:
        (model, tokenizer) tuple, model already on `device`.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if model_config := MODEL_CONFIGS.get(model_name):
        config = model_config.config_class.from_pretrained(model_name)

        if hasattr(config, model_config.dropout_field):
            setattr(config, model_config.dropout_field, 0)

        model = model_config.model_class.from_pretrained(
            model_name, config=config
        ).to(device)

    else:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        possible_dropout_keys = [
            "dropout", "dropout_rate", "attn_dropout", "attention_dropout", "summary_last_dropout"
        ]
        for key in possible_dropout_keys:
            if hasattr(config, key):
                setattr(config, key, 0)

        model = AutoModel.from_pretrained(
            model_name, config=config, device_map=device, trust_remote_code=True
        )

    return model, tokenizer


def get_tokens(
    texts,
    model_name="t5-base",
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    """Tokenize texts and return [ids | attention_mask] as a numpy array.

    Output shape is (n_texts, 2 * max_len_in_batch), with the first half
    being token IDs and the second half attention masks. Sequences are
    truncated to 512 tokens and padded to the longest in the batch.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    encoded_input = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to(device)

    input_ids_padded = pad_sequence(
        [torch.tensor(ids) for ids in encoded_input.input_ids],
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
    )
    attention_masks_padded = pad_sequence(
        [torch.tensor(mask) for mask in encoded_input.attention_mask],
        batch_first=True,
        padding_value=0,
    )
    all_encoded_inputs = torch.cat(
        [input_ids_padded, attention_masks_padded], dim=1
    )
    return all_encoded_inputs.cpu().numpy()
