"""Featurizer dispatching to a registry of representation functions."""
from typing import Optional
import inspect

import numpy as np


class Featurizer:
    """
    Main featurizer class that handles different featurization methods.
    Uses a factory pattern to create the appropriate featurization function.
    """
    def __init__(
        self,
        representation: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        self.representation = representation
        self.params = {
            "model_name": model_name,
        }
        self._featurization_registry = self._build_registry()

    def _build_registry(self):
        """
        Build a registry of featurization functions.
        This centralizes the import logic and makes it easier to add new methods.
        """
        from alice.featurization.text import get_tokens

        return {
            "get_tokens": get_tokens,
        }

    def featurize(self, data) -> np.ndarray:
        """
        Transform input data into feature vectors based on the configured representation.
        
        Args:
            data: Input data to featurize (list, pandas Series, etc.)
            
        Returns:
            np.ndarray: Featurized data
        """
        data_list = data.tolist() if hasattr(data, 'tolist') else data

        if self.representation not in self._featurization_registry:
            raise ValueError(f"Unsupported representation: {self.representation}")

        featurize_func = self._featurization_registry[self.representation]

        sig = inspect.signature(featurize_func)
        valid_params = {k: v for k, v in self.params.items() if k in sig.parameters}

        return featurize_func(data_list, **valid_params)
