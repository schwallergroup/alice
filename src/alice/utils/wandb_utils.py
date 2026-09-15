"""Helpers for building a descriptive W&B run name from a run config."""
import re


def last_token(x: str) -> str:
    """Return the final dotted component of a class path (e.g. 'a.b.C' -> 'C')."""
    return x.split(".")[-1] if x else "UNK"


def clean_token(x: str) -> str:
    """Strip characters that are awkward in run names / filenames."""
    return re.sub(r"[^A-Za-z0-9._+-]+", "", x) or "NA"


def shorten(x: str, n: int) -> str:
    """Truncate a string to at most n characters."""
    return x if len(x) <= n else x[:n]


def build_run_name(config: dict, prefix: str, max_len: int = 128) -> str:
    """Build the descriptive W&B run name from a run config.

    The name encodes surrogate/acquisition method, featurizer model, initializer
    method, batch size, and seed, e.g.::

        <prefix>-<model>-<surrogate>-<acqf>-<init>-batch_q<N>-seed<S>

    Args:
        config: The full run configuration dictionary.
        prefix: Leading token (typically the auto-generated W&B run name).
        max_len: Maximum length; the name is truncated to this many characters.

    Returns:
        The constructed run name string.
    """
    acqf_path = config.get("acquisition", {}).get("class_path", "")
    surrogate_path = config.get("surrogate_model", {}).get("class_path", "")

    acqf_short = clean_token(last_token(acqf_path))
    surrogate_short = clean_token(last_token(surrogate_path))

    featurizer_cfg = (
        config.get("data", {}).get("init_args", {}).get("featurizer", {}).get("init_args", {})
    )
    model_name = featurizer_cfg.get("model_name", None)
    if model_name:
        model_tok = model_name.split("/")[-1]
    else:
        model_tok = "numeric_descriptors"
    model_tok = clean_token(shorten(model_tok, 24))

    seed = config.get("seed", 0)  # set seed to 0 if not specified
    method = f"{surrogate_short}-{acqf_short}"

    bo_init = config.get("bo", {}).get("init_args", {})
    bsize = bo_init.get("batch_size", 1)  # batch size defaults to 1 if not set
    batch_tag = f"batch_q{bsize}"

    initializer_args = (
        config.get("data", {}).get("init_args", {}).get("initializer", {}).get("init_args", {})
    )
    init_method = initializer_args.get("method", "unknown_init")
    init_method_tag = clean_token(init_method)

    new_name = f"{prefix}-{model_tok}-{method}-{init_method_tag}-{batch_tag}-seed{seed}"
    if len(new_name) > max_len:
        new_name = new_name[:max_len]
    return new_name
