"Module for initializing BO campaign"
import logging
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import torch
from torch.quasirandom import SobolEngine
import torch.nn as nn

logger = logging.getLogger(__name__)

class Initializer(ABC):
    @abstractmethod
    def fit(self, x):
        pass


class SobolInitializer(Initializer):
    def __init__(self, n_clusters=96, seed=None, **kwargs):
        self.n_clusters = n_clusters
        self.seed = seed

    def fit(self, x):
        x_init = x
        device = x_init.device
        x_numpy = x_init.cpu().numpy()

        scaler = MinMaxScaler()
        x_normalized_numpy = scaler.fit_transform(x_numpy)
        
        x_scaled = torch.tensor(x_normalized_numpy, dtype=torch.float64).to(device)

        sobol = SobolEngine(dimension=x_scaled.shape[1], scramble=True, seed=self.seed)
        sobol_samples = sobol.draw(self.n_clusters).to(device)
        
        # Calculate distances 
        distances = torch.cdist(sobol_samples.float(), x_scaled.float())
        
        # Find the best unique matches
        selected_indices = []
        temp_distances = distances.clone()
        num_to_select = self.n_clusters

        for _ in range(num_to_select):
            min_distances, min_indices = temp_distances.min(dim=1) # gets the minimum distances of all clusters to all poitns

            best_sobol_idx = torch.argmin(min_distances) # gets the minimum of those minimum distances
            data_idx = min_indices[best_sobol_idx.item()]

            selected_indices.append(data_idx.item())

            temp_distances[:, data_idx] = float('inf')        # Column: mark data point used
            temp_distances[best_sobol_idx, :] = float('inf')  # Row: mark Sobol point done

        return selected_indices, {}


class BOInitializer:
    def __init__(
        self,
        method: str = "sobol",
        n_clusters: int = None,
        pca_dim: int = None,
        seed: Optional[int] = None,
        featurizer: Optional[nn.Module] = None,
        tkwargs: Optional[dict] = None,
        **kwargs,
    ):
        """
        Dispatcher that builds one concrete initializer (from ``self.methods``) and
        runs it in ``fit`` to choose the initial training points for the BO campaign.

        method: selects the concrete initializer class. Configs use 'sobol'
            -> ``SobolInitializer``.
        n_clusters: number of initial points to select.
        seed: Random seed for the Sobol draw; set from ``config["seed"]``.
        pca_dim: if set, PCA-reduces features in ``fit`` before
            the initializer runs. Applied to ``StandardScaler``-scaled features.
        featurizer: optional ``LLMFeaturizer``; when present, ``fit`` embeds the
            raw inputs through it (multilayer path) before selection.
        tkwargs: device/dtype for the feature tensor in ``fit``.


        Returns (from ``fit``): ``(selected_reactions, clusters)`` where
        ``selected_reactions`` are global row indices of the chosen initial points.
        """
        self.seed = seed
        self.n_clusters = n_clusters
        self.featurizer = featurizer
        self.pca_dim = pca_dim

        if tkwargs is None:
            self.tkwargs = {
                "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                "dtype": torch.float64, 
            }
        else:
            self.tkwargs = tkwargs

        if self.featurizer:
            logger.info(
                "BOInitializer: embedding inputs with %s before selection.",
                type(self.featurizer).__name__,
            )
            self.featurizer.eval()
        else:
            logger.info("BOInitializer: no featurizer, selecting on raw inputs.")

        self.methods = {
            "sobol": SobolInitializer,
        }
        if method not in self.methods:
            raise ValueError(f"Unknown init_method: {method}")

        init_cls = self.methods[method]
        init_params = dict(
            n_clusters=n_clusters,
            seed=seed,
            **kwargs,
        )

        self.initializer = init_cls(**init_params)
        self.selected_reactions = None
        self.clusters = None

    def fit(self, x):
        x_init = x

        if self.featurizer:
            logger.debug("Embedding inputs (shape %s) with the featurizer.", tuple(x_init.shape))
            with torch.no_grad():
                x_features_tensor = self.featurizer.get_embeddings(x_init.to(self.tkwargs['device']))
        else:
            logger.debug("Using inputs (shape %s) directly.", tuple(x_init.shape))
            x_features_tensor = x_init

        x_features_numpy = x_features_tensor.cpu().numpy()

        if self.pca_dim is not None:
            if isinstance(self.pca_dim, int) and self.pca_dim >= 1:
                scaler = StandardScaler()
                x_scaled = scaler.fit_transform(x_features_numpy)

                pca_fixed = PCA(n_components=self.pca_dim)
                x_processed = pca_fixed.fit_transform(x_scaled)
                logger.info(
                    "Reduced features %d -> %d dims (StandardScaler + PCA).",
                    x_features_numpy.shape[1],
                    self.pca_dim,
                )

            else:
                logger.warning("pca_dim=%r is not a positive int, skipping PCA.", self.pca_dim)
                x_processed = x_features_numpy
        else:
            logger.debug("No pca_dim set, skipping PCA.")
            x_processed = x_features_numpy

        x_for_init = torch.from_numpy(x_processed).to(**self.tkwargs)

        result = self.initializer.fit(x_for_init)

        if isinstance(result, tuple):
            selected_indices = result[0]
            clusters = result[1]
        else:
            selected_indices = result
            clusters = {}

        selected = np.asarray(selected_indices, dtype=int)
        if len(np.unique(selected)) != len(selected):
            raise RuntimeError(f"Initializer returned duplicate indices: {selected}")

        self.selected_reactions = np.sort(selected)
        self.clusters = clusters

        return self.selected_reactions, self.clusters
