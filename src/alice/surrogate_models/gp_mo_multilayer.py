"""Module for DeepGP implementation"""
import os
from typing import Optional

from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.input import Normalize, InputTransform, ChainedInputTransform
from botorch.models.transforms.outcome import Standardize

from gpytorch.kernels import ScaleKernel, MaternKernel, RBFKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.constraints import GreaterThan
from gpytorch.likelihoods import GaussianLikelihood

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW
import torch._dynamo
torch._dynamo.disable()

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from alice.surrogate_models.gp_mo import SurrogateModel
from alice.featurization.deep_multilayer import LLMFeaturizer


class SlicingTransform(InputTransform, nn.Module):
    def __init__(self, start_idx: int, end_idx: int, transform_on_train: bool = True, transform_on_eval: bool = True, transform_on_fantasize: bool = True):
        """
        # Input tensor with 5 features
        X = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])

        # Keep only features at indices 1, 2, 3
        slicer = SlicingTransform(start_idx=1, end_idx=4)
        X_sliced = slicer.transform(X)  # Returns [[2.0, 3.0, 4.0]]
        """
        super().__init__()
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.transform_on_train = transform_on_train
        self.transform_on_eval = transform_on_eval
        self.transform_on_fantasize = transform_on_fantasize

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        return X[..., self.start_idx : self.end_idx]

    def untransform(self, X: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Slicing is not reversible.")


class MultiDeepGP(SurrogateModel):
    def __init__(
        self,
        finetuning_model: LLMFeaturizer,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        ft_lr: float = 0.02,
        gp_lr: float = 0.1,
        wd_llm: float = 0.01,
        wd_gp: float = 0.0,
        initial_noise: float = 1e-2,
        noise_constraint: float = 1e-4,
        initial_lengthscale: float = 1.0,
        initial_outputscale: float = 1.0,
        kernel_class: str = "gpytorch.kernels.MaternKernel",
        device: str = "cuda",
        dtype: Optional[torch.dtype] = None
    ):
        """
        Applies slicing strategy within each GPs input transform if independent projection layers are used. 
        """
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.tkwargs = {"device": self.device, "dtype": dtype}
        raw_x = train_x.to(**self.tkwargs)
        raw_y = train_y.to(**self.tkwargs)
        self.raw_x = raw_x
        self.raw_y = raw_y

        featurizer = finetuning_model.to(**self.tkwargs)

        num_objs = raw_y.shape[-1]

        self.projection_dim = featurizer.projection_dim if featurizer.projection_dim is not None else featurizer.embedding_dim
        self.use_slicing_strategy = getattr(featurizer, 'independent_projection_heads', True)

        if self.use_slicing_strategy:
            print(f"[MODEL] Initializing Multi-Head DeepGP (Independent) with {num_objs} heads.")
            if self.projection_dim * num_objs != featurizer.output_dim:
                raise ValueError(f"Dimension Mismatch: Expected {self.projection_dim * num_objs} but got {featurizer.output_dim}")
        else:
            print("[MODEL] Initializing Single-Head DeepGP (Shared Representation).")

        with torch.no_grad():
            z_full = featurizer(raw_x)

        gp_list = []
        for i in range(num_objs):
            if self.use_slicing_strategy:
                start = i * self.projection_dim
                end = (i + 1) * self.projection_dim

                # slice out this objective's projection block, then normalize it per GP
                slicing_t = SlicingTransform(start, end)
                norm_t = Normalize(d=self.projection_dim)
                input_transform = ChainedInputTransform(slice=slicing_t, norm=norm_t)
            else:
                input_transform = Normalize(d=z_full.shape[-1])
            
            yi = raw_y[:, i].unsqueeze(-1)
            
            likelihood = GaussianLikelihood()
            likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(noise_constraint))

            if "Matern" in kernel_class:
                base_kernel = MaternKernel(nu=2.5)
            else:
                base_kernel = RBFKernel()

            gp = SingleTaskGP(
                train_X=z_full,
                train_Y=yi,
                likelihood=likelihood,
                outcome_transform=Standardize(1), # standardize labels
                input_transform=input_transform, # applies SlicingTransform and normalization
                covar_module=ScaleKernel(base_kernel),
            )

            gp.likelihood.noise_covar.initialize(raw_noise=torch.tensor(initial_noise).to(self.device))
            gp.covar_module.base_kernel.initialize(raw_lengthscale=torch.tensor(initial_lengthscale).to(self.device))
            gp.covar_module.initialize(raw_outputscale=torch.tensor(initial_outputscale).to(self.device))
            gp_list.append(gp)

        self.model = ModelListGP(*gp_list)
        self.finetuning_model = featurizer

        self.mlls = nn.ModuleList([ExactMarginalLogLikelihood(gp.likelihood, gp) for gp in self.model.models])

        # train params
        llm_params = [p for p in self.finetuning_model.parameters() if p.requires_grad]
        gp_params = list(self.model.parameters())

        self.optimizer = AdamW(
            [
                {"params": llm_params, "lr": ft_lr, "weight_decay": wd_llm},
                {"params": gp_params, "lr": gp_lr, "weight_decay": wd_gp},
            ]
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=10, eta_min=1e-6)

        self.model.to(self.device)
        self.finetuning_model.to(self.device)

    def fit(self, num_epochs: int = 10):
        for epoch in range(num_epochs):
            print(f"Starting Epoch {epoch}")

            self.finetuning_model.train()
            for gp in self.model.models:
                gp.train()
                gp.likelihood.train()

            z_full = self.finetuning_model(self.raw_x)

            self.optimizer.zero_grad()

            for gp_idx, gp in enumerate(self.model.models):

                gp.set_train_data(inputs=z_full, targets=gp.train_targets, strict=False)

                output = gp(z_full)

                loss_i = -self.mlls[gp_idx](output, gp.train_targets)

                loss_i.backward(retain_graph=True)

            self.optimizer.step()
            self.scheduler.step()

    @torch.no_grad()
    def posterior(self, x: torch.Tensor):
        self.model.eval()
        self.finetuning_model.eval()

        z_full = self.finetuning_model(x.to(self.device))

        return self.model.posterior(z_full)

    @torch.no_grad()
    def predict(self, x: torch.Tensor, return_var: bool = True):
        post = self.posterior(x)
        means = torch.cat([d.mean.unsqueeze(-1) for d in post.distributions], dim=-1)
        if return_var:
            vars = torch.cat([d.variance.unsqueeze(-1) for d in post.distributions], dim=-1)
            return means, vars
        return means

    @property
    def num_outputs(self):
        return len(self.model.models)
