"Module for running acquisition functions and obtaining suggestions"
from typing import Any, Dict, Optional
import logging

from tqdm import tqdm

import torch
from torch import Tensor
from botorch.sampling.normal import SobolQMCNormalSampler

from alice.utils.config import instantiate_class
from alice.bo.acqf_helper import ACQF_REGISTRY, batched_acqf_forward


logger = logging.getLogger(__name__)


class BotorchOptimizer:
    """Surrogate-model + acquisition-function optimizer for the BO loop.

    Instantiated once per run by ``setup_bo_optimizer`` in ``train_master.py``;
    ``suggest_next_experiments`` is then called once per BO iteration.

    Constructor arguments (source -> internal use):
        surrogate_model_config: ``config["surrogate_model"]``. Instantiated and
            fitted in ``train_surrogate_model`` (called per iteration).
        acq_function_config: ``config["acquisition"]``. Resolved to an acqf class
        via ``ACQF_REGISTRY``. Only ``init_args.ref_point`` is read from it, and
            only for the Noisy EHVI variants; all other ``init_args`` keys are
            ignored, since the constructor kwargs are set explicitly in ``suggest_next_experiments``.
        batch_size: ``config["bo"]["init_args"]["batch_size"]``. Number of greedy
            selection steps per iteration (``steps = min(batch_size, n_candidates)``).
        max_eval_chunk: chunk size for ``batched_acqf_forward``. Not set by any
            config or ``setup_bo_optimizer`` -> always the default (2048). Kept as a
            tuning knob, but effectively constant in the shipped pipeline.
        tkwargs: device/dtype dict, built by ``setup_bo_optimizer`` from ``dtype``.
            Used throughout for tensor placement.

    Attributes read OUTSIDE this class:
        surrogate_model: read by ``save_checkpoint`` (``utils/model_utils.py``) to
            persist the fitted featurizer/projection layer.

    Internal-only attributes:
        _shared_sampler: lazily-created ``SobolQMCNormalSampler`` reused across steps.
    """
    def __init__(
        self,
        surrogate_model_config: Optional[Dict[str, Any]] = None,
        acq_function_config: Optional[Dict[str, Any]] = None,
        batch_size: int = 1,
        max_eval_chunk: int = 2048,
        tkwargs: Optional[Dict[str, Any]] = None,
    ):
        if tkwargs is None:
            raise ValueError(
                "tkwargs is required; setup_bo_optimizer supplies device and dtype."
            )

        if not acq_function_config:
            raise ValueError(
                "acq_function_config is required."
            )

        self.surrogate_model_config = surrogate_model_config
        self.acq_function_config = acq_function_config
        self.batch_size = batch_size
        self.tkwargs = tkwargs
        self.max_eval_chunk = max_eval_chunk

        self._shared_sampler = None
        self.surrogate_model = None
        logger.info("BotorchOptimizer using device %s.", self.tkwargs["device"])

    def train_surrogate_model(self, train_x, train_y):
        "Instantiate surrogate and fit it"
        self.surrogate_model = instantiate_class(
            self.surrogate_model_config,
            train_x=train_x,
            train_y=train_y,
        )

        self.surrogate_model.model.to(**self.tkwargs)
        try:
            logger.debug("Fitting surrogate model.")
            self.surrogate_model.fit()
        except Exception:
            logger.exception("Surrogate model fit failed at this BO iteration.")
            raise

    def suggest_next_experiments(self, train_x: Tensor, train_y: Tensor, design_space: Tensor):
        """Select the next batch of experiments from ``design_space``.

        Refits the surrogate on the current data, then greedily picks
        ``min(batch_size, len(design_space))`` points, adding each selection to
        ``X_pending`` so later picks account for earlier ones.

        Args:
            train_x: Observed inputs, shape (n_train, d).
            train_y: Observed objectives, shape (n_train, n_objectives).
            design_space: Candidate pool to select from, shape (n_candidates, d).

        Returns:
            list[int]: Row indices into ``design_space`` of the selected points,
            in selection order. ``train_master`` maps these to global dataset
            indices via ``dm.heldout_indices``.
        """
        # getting the acquisition function class from the config file argument
        acqf_class_path = str(self.acq_function_config.get("class_path", ""))
        acqf_class = ACQF_REGISTRY.get(acqf_class_path) or ACQF_REGISTRY.get(acqf_class_path.split(".")[-1])
        if acqf_class is None:
            raise ValueError(f"Unknown acquisition class_path {acqf_class_path!r}. "
                            f"Known: {sorted(ACQF_REGISTRY)}")

        logger.info("Re-training surrogate model with new data...")
        self.train_surrogate_model(train_x, train_y)

        if self._shared_sampler is None:
            self._shared_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([256]))

        # getting featurizer from finetuned model, defining featurizer, selected X, and possible choices
        featurizer = getattr(self.surrogate_model, "finetuning_model", None)
        if featurizer is None:
            z_baseline = train_x.to(**self.tkwargs)
            z_choices  = design_space.to(**self.tkwargs)
        else:
            featurizer = featurizer.eval()
            with torch.no_grad():
                z_baseline = featurizer(train_x.to(**self.tkwargs))
                z_choices  = featurizer(design_space.to(**self.tkwargs))

        acq_name = acqf_class.__name__
        acq_args = {}

        if "Hypervolume" in acq_name or "EHVI" in acq_name:
            ref_point_arg = (self.acq_function_config.get("init_args") or {}).get("ref_point")
            if ref_point_arg is not None:
                ref_point = torch.as_tensor(ref_point_arg, **self.tkwargs)
            else:
                # Component-wise nadir of the observations so far
                ref_point = train_y.to(**self.tkwargs).min(dim=0).values - 1e-6
            acq_args["ref_point"] = ref_point
            acq_args["incremental_nehvi"] = True

        selected_indices = []

        # train data points are dropped before evaluation in train_master.py
        choices_idx = torch.arange(design_space.size(0), device=design_space.device)

        eval_matrix = z_choices.clone() # features of the design space
        eval2global = choices_idx.clone() # index of design space from 0 to n

        steps = min(self.batch_size, eval_matrix.size(0))
        pending = torch.empty(0, eval_matrix.size(-1), **self.tkwargs)

        for step in tqdm(range(steps), desc="Greedy Batch Selection"):
            logger.debug(
                "[step %d/%d] scoring %d remaining candidates.",
                step + 1, steps, eval_matrix.size(0),
            )

            acqf_step = acqf_class(
                model=self.surrogate_model.model,
                X_baseline=z_baseline,
                X_pending=pending,
                sampler=self._shared_sampler,
                prune_baseline=True,
                **acq_args # contains ref_point if needed
            ).to(**self.tkwargs)
            acqf_step.eval()

            acq_vals = batched_acqf_forward(
                acqf=acqf_step,
                X=eval_matrix,
                max_eval_batch=self.max_eval_chunk
            ).to(**self.tkwargs)

            best_idx_local = int(acq_vals.argmax().item()) # best idx within eval_matrix
            global_idx = int(eval2global[best_idx_local].item()) # get idx of candidate within global design_space

            selected_indices.append(global_idx)

            # update search space and indexing to drop selected point
            keep_mask = (eval2global != global_idx)
            eval_matrix = eval_matrix[keep_mask]
            eval2global = eval2global[keep_mask]

            best_candidate_z = z_choices[global_idx]
            pending = torch.cat([pending, best_candidate_z.unsqueeze(0)], dim=0)

        return selected_indices


def setup_bo_optimizer(config, dtype):
    """
    Initialize a BotorchOptimizer object with a provided config file.

    Args:
        config (dict): Configuration dictionary containing:
            - config["surrogate_model"]: Surrogate model configuration
            - config["acquisition"]: Acquisition function configuration
            - config["bo"]["init_args"]: BO settings (batch_size)
        dtype (torch.dtype): Data type

    Returns:
        BotorchOptimizer: Configured optimizer instance ready for suggesting next
            experiments in the BO loop.

    Example:
        >>> config = {
        ...     "bo": {"init_args": {"batch_size": 4}},
        ...     "surrogate_model": {...},
        ...     "acquisition": {...}
        ... }
        >>> optimizer = setup_bo_optimizer(config, torch.float64)
    """
    tkwargs = {
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "dtype": dtype
    }

    bo = BotorchOptimizer(
        surrogate_model_config=config["surrogate_model"],
        acq_function_config=config["acquisition"],
        batch_size=config["bo"]["init_args"].get("batch_size", 1),
        tkwargs=tkwargs
    )
    return bo
