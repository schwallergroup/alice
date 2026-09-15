"Module for default GP models"
import os
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch._dynamo

from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_mll

from gpytorch.mlls import SumMarginalLogLikelihood
from gpytorch.constraints import GreaterThan
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.priors import GammaPrior
from gpytorch.settings import cholesky_jitter, max_root_decomposition_size

torch._dynamo.disable()

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

class SurrogateModel(ABC):
    "GP surrogate model base class."
    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self, x):
        pass

class MultiGP(SurrogateModel):
    """
    Static Gaussian Process model optimized for fixed descriptor.

    Adapted from doi.org/10.1021/jacs.2c08592 and github.com/doyle-lab-ucla/edboplus.git
    & github.com/schwallergroup/minerva.git
    """
    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        standardize: bool = True,
        normalize: bool = True,
        noise_constraint: float = 1e-3,
        ls_prior1: float = 2.0,
        ls_prior2: float = 0.2,
        out_prior1: float = 5.0,
        out_prior2: float = 0.5,
        noise_prior1: float = 1.5,
        noise_prior2: float = 0.1,
        device: str = "cuda",
        dtype: torch.dtype = torch.float64,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.tkwargs = {"device": self.device, "dtype": dtype}

        X = train_x.to(**self.tkwargs)
        Y = train_y.to(**self.tkwargs)
        if Y.ndim == 1:
            Y = Y.unsqueeze(-1)

        x_dim = X.shape[-1]
        models = []

        for i in range(Y.shape[-1]):
            yi = Y[:, i:i+1]
            covar = ScaleKernel(
                MaternKernel(
                    nu=2.5,
                    ard_num_dims=x_dim,
                    lengthscale_prior=GammaPrior(ls_prior1, ls_prior2)
                ),
                outputscale_prior=GammaPrior(out_prior1, out_prior2),
            )

            likelihood = GaussianLikelihood(
                noise_prior=GammaPrior(noise_prior1, noise_prior2),
                noise_constraint=GreaterThan(noise_constraint),
            )

            m = SingleTaskGP(
                train_X=X,
                train_Y=yi,
                covar_module=covar,
                likelihood=likelihood,
                input_transform=Normalize(x_dim) if normalize else None,
                outcome_transform=Standardize(1) if standardize else None,
            )
            models.append(m)

        self.model: ModelListGP = ModelListGP(*models).to(self.device)

        # Placeholder for duck-typing with MultiDeepGP; no LLM is fine-tuned here.
        self.finetuning_model = nn.Identity().to(self.device)

    def fit(self, scipy_options: dict | None = None):
        self.model.train()
        for gp in self.model.models:
            gp.train()
            gp.likelihood.train()

        mll = SumMarginalLogLikelihood(self.model.likelihood, self.model)
        with cholesky_jitter(1e-3): # add jitter to stabilise training
            fit_gpytorch_mll(mll, options=(scipy_options or {"maxiter": 500}))

    @torch.no_grad()
    def posterior(self, x: torch.Tensor):
        self.model.eval()
        for gp in self.model.models:
            gp.eval()
            gp.likelihood.eval()
        with cholesky_jitter(1e-3), max_root_decomposition_size(0): # stabilise inference
            return self.model.posterior(x.to(**self.tkwargs))

    @torch.no_grad()
    def predict(self, x: torch.Tensor, return_var: bool = True):
        posterior = self.posterior(x)
        means = torch.cat([d.mean.unsqueeze(-1) for d in posterior.distributions], dim=-1)
        if return_var:
            vars_ = torch.cat([d.variance.unsqueeze(-1) for d in posterior.distributions], dim=-1)
            return means, vars_
        return means

    @property
    def num_outputs(self) -> int:
        return len(self.model.models)

