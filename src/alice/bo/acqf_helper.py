"Helper file for all acquisition function computations"
import torch
from torch import Tensor
import gpytorch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.multi_objective import (
    qLogNoisyExpectedHypervolumeImprovement,
)

from botorch.acquisition.multi_objective.parego import qLogNParEGO


ACQF_REGISTRY = {
    "botorch.acquisition.multi_objective.qLogNoisyExpectedHypervolumeImprovement": qLogNoisyExpectedHypervolumeImprovement,
    "qLogNoisyExpectedHypervolumeImprovement": qLogNoisyExpectedHypervolumeImprovement,
    "botorch.acquisition.multi_objective.parego.qLogNParEGO": qLogNParEGO,
    "qLogNParEGO": qLogNParEGO,
}


def batched_acqf_forward(
    acqf: AcquisitionFunction,
    X: Tensor,
    max_eval_batch: int=2048
) -> Tensor:
    """
    Evaluates an acquisition function on a set of candidates X in memory-safe batches.

    Splits a large tensor of candidate points `X` into smaller mini-batches,
    scoring each batch sequentially, and then concatenating the results.

    Args:
        acqf: The acquisition function to evaluate. Must be a callable
              that accepts a tensor of candidates.
        X: A tensor of candidate points to be scored.
        max_eval_batch: The number of candidates to evaluate in each batch.

    Returns:
        A tensor containing the acquisition score for each candidate in `X`
    """
    scores = []
    n_total = X.shape[0]
    n_batches = (n_total + max_eval_batch - 1) // max_eval_batch
    device = X.device
    for i in range(n_batches):
        start = i * max_eval_batch
        end = min((i + 1) * max_eval_batch, n_total)
        x_batch = X[start:end].unsqueeze(1)

        with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.fast_pred_samples():
            s = acqf(x_batch).squeeze(-1).detach().cpu()
        scores.append(s)

    return torch.cat(scores, dim=0).to(device)
