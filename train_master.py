"""Entry point for a multi-objective Bayesian optimization campaign.

Usage::

    python train_master.py --config configs/<dataset>/<config>.yaml

The YAML supplies four component blocks, each a ``{class_path, init_args}`` pair
built by ``alice.utils.config.instantiate_class``:

    data             ``BaseDataModule``     -- CSV loading, featurization, initial split
    surrogate_model  ``SurrogateModel``     -- GP refitted every iteration
    acquisition      ``AcquisitionFunction``-- resolved through ``ACQF_REGISTRY``
    bo               ``BotorchOptimizer``   -- greedy batch selection

alongside top-level ``seed``, ``n_iters``, ``out_root``, ``output_model_dir`` and the
``wandb_*`` keys.

``--seed``, ``--batch_size``, ``--n_clusters`` and ``--n_iters`` may also be passed on
the command line (the W&B sweep does this). jsonargparse applies ``--config`` at its
position in argv, so these must come AFTER ``--config`` to take effect; the resolved
seed is logged at startup.

Flow: validate the config -> build the data module and its initial Sobol split -> log
iteration 0 -> then per iteration, refit the surrogate, select ``batch_size`` points,
move them from heldout to train, log metrics, and checkpoint any LLM featurizer.

Outputs::

    <out_root>/<surrogate>-<acqf>-<representation>-<model>/seed=<N>/batch_q<M>/
        hypervolume_seed<N>_batch_q<M>.csv      iteration, n_experiments,
                                                hypervolume, hv_percent
        hv_percent_vs_iteration_*.png           HV% against BO iteration
        hv_percent_vs_experiments_*.png         HV% against cumulative experiments
        front_map_multiobj_*.png                objective-space / Pareto front

    <output_model_dir>/iteration_<i>/           LoRA adapters and projection heads,
                                                written only for LLM surrogates
"""
import warnings
import logging
from functools import partial

import numpy as np
import torch
from tqdm import tqdm

from botorch.exceptions import InputDataWarning
from botorch.acquisition import AcquisitionFunction
from botorch.utils.multi_objective.pareto import is_non_dominated

from pytorch_lightning import seed_everything

from jsonargparse import ArgumentParser, ActionConfigFile

import wandb

from alice.data.module import BaseDataModule
from alice.utils.config import instantiate_class, flatten
from alice.bo.optimizer import BotorchOptimizer, setup_bo_optimizer
from alice.surrogate_models.gp_mo import SurrogateModel
from alice.utils.model_utils import save_checkpoint
from alice.utils.wandb_utils import build_run_name, last_token

from alice.metrics import (
    MobboCsvLogger
)

from alice.utils.diagnostics import (
    print_diagnostic_summary,
    print_bo_diagnostic,
)

from alice.utils.model_embd_sizes import MODEL_EMBEDDING_SIZES


logger = logging.getLogger(__name__)

def _silence_known_warnings():
    """Suppress third-party warnings that are expected and not actionable here."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # logger.setLevel(logging.DEBUG)
    warnings.filterwarnings("ignore", category=InputDataWarning)

torch.set_float32_matmul_precision("high")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("Using device: %s", device)

BO_COMPUTE_DTYPE = torch.float32


############################## Setting up the helper functions ##############################
def validate_configuration(config):
    """Validate that the configuration is consistent."""
    surrogate_class = config["surrogate_model"]["class_path"]

    featurizer_config = config["data"]["init_args"]["featurizer"]["init_args"]
    representation = featurizer_config.get("representation")

    if surrogate_class == "alice.surrogate_models.gp_mo.MultiGP" and representation == "get_tokens":
        raise ValueError("Standard GP shouldn't use 'get_tokens'. This is for trainable LLM models only.")

    # Ensure the finetuning model's input_dim matches the known embedding size
    model_name = featurizer_config.get("model_name")
    if model_name in MODEL_EMBEDDING_SIZES:
        embedding_size = MODEL_EMBEDDING_SIZES[model_name]

        if "surrogate_model" in config and "init_args" in config["surrogate_model"]:
            if "finetuning_model" in config["surrogate_model"]["init_args"]:
                current_dim = config["surrogate_model"]["init_args"]["finetuning_model"]["init_args"].get("input_dim")
                if current_dim != embedding_size:
                    raise ValueError(
                        f"input_dim mismatch for model '{model_name}': config specifies "
                        f"{current_dim}, but '{model_name}' produces embeddings of size "
                        f"{embedding_size}. Update finetuning_model.init_args.input_dim "
                        f"to {embedding_size} in your config."
                    )


def setup_data(config):
    """
    Initialize and configure the BaseDataModule object for Bayesian optimization.

    This function instantiates the featurizer, initializer, and data module from
    the configuration dictionary.

    Args:
        config (dict): Configuration dictionary containing:
            - config["data"]["init_args"]["data_path"]: Path to dataset CSV
            - config["data"]["init_args"]["input_column"]: Name of input column(s)
            - config["data"]["init_args"]["initializer"]: Initializer configuration
            - config["data"]["init_args"]["featurizer"]: Featurizer configuration
            - config["data"]["init_args"]["normalize_input"]: Normalization strategy
            - config["data"]["init_args"]["maximize"]: Optimization direction(s)
            - config["seed"]: Random seed for reproducibility

    Returns:
        BaseDataModule: Configured data module instance with:
            - Loaded dataset from CSV
            - Initialized featurizer for input transformation
            - Initializer for selecting initial training points
            - Split into training and heldout sets

    Notes:
        - The data module automatically performs featurization, preprocessing,
          normalization, and train/heldout splitting during instantiation
        - The seed is passed to the initializer to ensure reproducible splits

    Example:
        >>> config = {
        ...     "data": {
        ...         "class_path": "alice.data.module.BaseDataModule",
        ...         "init_args": {
        ...             "data_path": "data/reactions.csv",
        ...             "input_column": "smiles",
        ...             "initializer": {"class_path": "...", "init_args": {"method": "sobol"}},
        ...             "featurizer": {"class_path": "...", "init_args": {...}},
        ...             "normalize_input": "original",
        ...             "maximize": True
        ...         }
        ...     },
        ...     "seed": 42
        ... }
        >>> data_module = setup_data(config)
        >>> print(f"Training samples: {len(data_module.train_x)}")
    """
    data_cfg = config["data"]["init_args"]
    tkwargs = {"device": device, "dtype": BO_COMPUTE_DTYPE}

    finetuning_model_config = config.get("surrogate_model", {}).get("init_args", {}).get("finetuning_model", None)
    if finetuning_model_config:
        logger.info("[setup_data] Instantiating featurizer nn.Module for BOInitializer")
        embedding_model_instance = instantiate_class(finetuning_model_config)
    else:
        logger.info("[setup_data] No 'finetuning_model' config found. BOInitializer will use raw data.")
        embedding_model_instance = None

    initializer = instantiate_class(
        data_cfg["initializer"],
        seed=config["seed"],
        featurizer=embedding_model_instance, # llm_embeddings for initialization
        tkwargs=tkwargs
    )

    featurizer = instantiate_class(data_cfg["featurizer"])

    data_module = instantiate_class(
        config["data"], # instantiate BaseDataModule object
        initializer=initializer,
        featurizer=featurizer, # tokenizer
    ) # Every other init_arg (data_path, target_column, normalize_input, maximize, dtype) read from config

    return data_module


def clean_and_update_indices(data_module, orig_ids_to_add, pos_in_design):
    """
    Ensures new indices are unique and not already in the training set,
    then updates the training and heldout index arrays.

    Args:
        - data_module (BaseDataModule): Data module object with updated train/heldout attributes
        - orig_ids_to_add: indexes of selected points in the global search space
        - pos_in_design: indexes of selected points in design space (not global search space!)

    Note:
        - The processing may order train indices differently, leading to floating point
          differences in results
    """
    # contains the global indices of the train and heldout dataset
    train_indexes_np = np.asarray(data_module.train_indexes)
    heldout_indices_np = np.asarray(data_module.heldout_indices)

    # sorted list
    unique_new_ids, unique_pos = np.unique(orig_ids_to_add, return_index=True)
    if len(unique_new_ids) != len(orig_ids_to_add):
        num_duplicates = len(orig_ids_to_add) - len(unique_new_ids)
        raise ValueError(
            f"Batch contains {num_duplicates} duplicate indices."
            f"This indicates a bug in the acquisition function loop!"
        ) # can simplify filtering in a later version

    keep = ~np.isin(unique_new_ids, train_indexes_np)
    if not np.all(keep):
        removed_count = np.sum(~keep)
        raise ValueError(
            f"Batch contains {removed_count} indices already in training set."
            f"Check that design_space excludes training points."
        )

    final_ids_to_add = unique_new_ids[keep] # global_id to add
    final_pos_to_remove = pos_in_design[unique_pos][keep] # positional_id to remove in heldout_indices

    data_module.train_indexes = np.concatenate([train_indexes_np, final_ids_to_add]).astype(int)
    data_module.heldout_indices = np.delete(heldout_indices_np, final_pos_to_remove).astype(int)

    return final_ids_to_add


def train(config, my_logger):
    "Main training loop, including initialisation, optimisation, and logging of results."
    validate_configuration(config)
    wandb_config = flatten(config)

    with wandb.init(
        project=config.get("wandb_project", "default-project-name"),
        name=config.get("wandb_run_name"),
        group=config.get("group"),
        config=wandb_config
    ) as run:

        # ---- Pull config bits used in this scope ----
        featurizer_cfg = config.get("data", {}).get("init_args", {}).get("featurizer", {}).get("init_args", {})
        model_name = featurizer_cfg.get("model_name", None)
        representation = featurizer_cfg.get("representation", None)
        trainable = featurizer_cfg.get("trainable", True) # used for diagnostics

        if representation is not None:
            rep_token = str(representation)
            rep_short = "tokens" if rep_token == "get_tokens" else rep_token # used for diagnostics
        else:
            rep_short = "embeddings" if model_name else "descriptors"

        acqf_name = last_token(config.get("acquisition", {}).get("class_path", ""))
        seed = config["seed"]

        # ---- W&B run name ----
        run.name = build_run_name(config, prefix=run.name or "run")
        run.notes = "Auto-generated short run name"
        wandb.run.name = run.name

        # ---- Data setup ----
        data_module = setup_data(config)
        out_root = config.get("out_root", "./results")

        # optional logging of runs
        print_diagnostic_summary(data_module, rep_short, trainable, model_name)

        # ---- Create BoTorchOptimizer object ----
        bo = setup_bo_optimizer(config, dtype=BO_COMPUTE_DTYPE)
        print_bo_diagnostic(data_module, config, bo, model_name, acqf_name)

        # ---- Ground truth Y and Pareto front for HV calculation ----
        Y_full = data_module.y.clone().cpu()
        mask_full = is_non_dominated(Y_full)
        pareto_full = Y_full[mask_full]

        # ---- Shows the iteration each point was collected ----
        sample_iters = np.zeros(data_module.train_y.size(0), dtype=int)
        sample_iters_t = torch.from_numpy(sample_iters).to(torch.int)

        # ---- Ref point (0, 0) for HV metric comptuation as yield/selectivity 0-100%  ----
        rp_tuple = tuple(config.get("acquisition", {}).get("init_args", {}).get("ref_point", (0.0, 0.0)))

        # ---- Log metrics before BO and after each iteration ----
        log_iteration = partial(
            my_logger.log_iteration,
            Pareto_full=pareto_full,
            out_root=out_root,
            model_name=model_name if model_name else "numeric_descriptors",
            seed=seed,
            ref_point=rp_tuple,
            log_wandb=True,
            acqf_name=acqf_name,
            config=config,
        )

        log_iteration(
            current_y=data_module.train_y,
            sample_iters=sample_iters_t,
            iteration=0,
        )

        final_model_directory = config.get("output_model_dir", "./default_model_output_dir")

        for i in tqdm(range(config["n_iters"]), colour="blue"):
            logger.debug("Train indices: %s", np.asarray(data_module.train_indexes))
            logger.debug("Heldout indices: %s", np.asarray(data_module.heldout_indices))

            # ---- Dynamically updating train_x & train_y ----
            train_x = data_module.train_x.clone().to(device, dtype=BO_COMPUTE_DTYPE)
            train_y = data_module.train_y.clone().to(device, dtype=BO_COMPUTE_DTYPE)
            design_space = data_module.heldout_x.clone().to(device, dtype=BO_COMPUTE_DTYPE)

            # ---- Trains GP and obtains next experiments ----
            returned_indices = bo.suggest_next_experiments(train_x, train_y, design_space)
            # indices of candidates within design_space, not global

            # Map the indices to global IDs
            pos_in_design = np.asarray(returned_indices, dtype=int)
            orig_ids_to_add = data_module.heldout_indices[pos_in_design] # heldout indices ordered with heldout_x, which design_space uses

            # Use the helper function to clean and update the indices
            final_ids_to_add = clean_and_update_indices(data_module, orig_ids_to_add, pos_in_design)
            # returns idxs of the original design space to add, and updates BaseDataModule to take into account new train and heldout indices (global indeces)

            logger.info(
                "Added %d points. Train set size: %d, Heldout set size: %d",
                len(final_ids_to_add), len(data_module.train_indexes), len(data_module.heldout_indices),
            )

            # Update the data tensors based on the new index arrays
            data_module.train_x = data_module.x[data_module.train_indexes]
            data_module.train_y = data_module.y[data_module.train_indexes]
            data_module.heldout_x = data_module.x[data_module.heldout_indices]
            data_module.heldout_y = data_module.y[data_module.heldout_indices]

            new_iters = np.full((len(final_ids_to_add),), i + 1, dtype=int)
            sample_iters = np.concatenate([sample_iters, new_iters]).astype(int)

            sample_iters_t = torch.from_numpy(sample_iters).to(dtype=torch.int)

            log_iteration(
                current_y=data_module.train_y,
                sample_iters=sample_iters_t,
                iteration=i + 1,
            )
            if hasattr(getattr(bo.surrogate_model, "finetuning_model", None), "llm"):
                save_checkpoint(bo_optimizer=bo, iteration=i + 1, base_dir=final_model_directory)


def main():
    _silence_known_warnings()
    parser = ArgumentParser(
        description="Training script",
        default_config_files=[],
    )
    parser.add_argument("--config", action=ActionConfigFile)
    parser.add_argument("--seed", type=int, help="Random seeds to use")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--n_clusters", type=int, help="Override the number of initial points (n_clusters).")
    parser.add_argument("--n_iters", type=int, help="How many iterations to run")
    parser.add_argument("--group", type=str, help="Wandb group runs")
    parser.add_argument('--wandb_project', type=str, help='Name of the W&B project.')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='Optional name for the specific W&B run.')

    parser.add_argument(
        '--out_root',
        type=str,
        help='Root directory to save experiment results.',
        default='./results'
    )

    parser.add_argument(
        '--output_model_dir',
        type=str,
        help='Root directory to save experiment models.',
        default='results/models'
    )

    parser.add_subclass_arguments(BaseDataModule, "data", instantiate=False)
    parser.add_subclass_arguments(
        SurrogateModel,
        "surrogate_model",
        instantiate=False,
        skip=["train_x", "train_y"],
    )

    parser.add_subclass_arguments(
        AcquisitionFunction,
        "acquisition",
        instantiate=False,
        skip=[
            "model",
            "best_f",
            "partitioning",
            "sampler",
            "objective",
            "constraints",
            "X_pending",
            "X_baseline",
        ],
    )
    parser.add_subclass_arguments(BotorchOptimizer, "bo", instantiate=False)

    # parse arguments
    args = parser.parse_args()
    config = args.as_dict()
    my_logger = MobboCsvLogger()

    logger.debug("Parsed config: %s", config)
    if config["seed"] is None:
        raise ValueError(
            "No seed resolved. Pass --seed, or set a top-level 'seed' in the config."
        )
    logger.info("Seed: %s", config["seed"])

    # Top-level CLI overrides (set by the wandb sweep) take precedence over the
    # corresponding nested config values. Automatically overrided for seed.
    if 'batch_size' in config and config['batch_size'] is not None:
        logger.info("[CONFIG SYNC] Syncing top-level batch_size (%s) to bo.init_args.", config['batch_size'])
        config.setdefault('bo', {}).setdefault('init_args', {})['batch_size'] = config['batch_size']

    if 'n_clusters' in config and config['n_clusters'] is not None:
        logger.info("[CONFIG SYNC] Syncing top-level n_clusters (%s) to initializer.init_args.", config['n_clusters'])
        config.setdefault('data', {}).setdefault('init_args', {}).setdefault('initializer', {}).setdefault('init_args', {})['n_clusters'] = config['n_clusters']

    # config["seed"] is the CLI --seed supplied by the wandb sweep.
    seed_everything(config["seed"], workers=True)

    train(config, my_logger)


if __name__ == "__main__":
    main()
