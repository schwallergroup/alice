"Module for handling data loading of descriptors, prompts, etc. from .csv files."
import logging
from typing import List, Union, Optional

import numpy as np
import pandas as pd
import torch

from alice.data.utils import find_duplicates, find_nan_rows
from alice.initialization.initializers import BOInitializer
from alice.featurization.base import Featurizer


logger = logging.getLogger(__name__)


class BaseDataModule:
    """
    Data container for (multi-)objective Bayesian optimization.

    On construction, ``setup()`` runs the full pipeline:
    load CSV -> featurize -> preprocess -> normalize -> split into train/heldout.

    Constructor arguments (source -> use):
        data_path: ``config[...]["data_path"]``. CSV read in ``load_data``.
        input_column: ``config[...]["input_column"]``. For 'descriptors' a list of
            feature columns (selected directly); otherwise the text column(s) fed to
            the featurizer in ``featurize_data``.
        target_column: ``config[...]["target_column"]``. Objective column(s); read by
            ``load_data``/``featurize_data`` and exposed as ``self.target_column``.
        maximize: ``config[...]["maximize"]``. Per-objective direction; minimize
            targets are sign-flipped in ``load_data``. Length must match
            ``target_column``.
        normalize_input: ``config[...]["normalize_input"]``. Input scaling strategy in ``normalize_data``.
        featurizer: instantiated by ``setup_data`` from
            ``config[...]["featurizer"]``. Drives ``featurize_data``.
        initializer: instantiated by ``setup_data`` from
            ``config[...]["initializer"]`` (a ``BOInitializer``). Selects the initial
            train set in ``split_data``.
        dtype: ``config[...]["dtype"]``. dtype of ``self.x``/``self.y``.

    Key attributes read by ``train_master``:
        x, y: full featurized inputs/targets.
        train_x/train_y, heldout_x/heldout_y: current split tensors.
        train_indexes/heldout_indices: global row indices of each split (the BO loop
            mutates these each iteration).
        data: the raw DataFrame; target_column: resolved objective names.
    """
    def __init__(
        self,
        data_path: str,
        input_column: Union[str, List[str]] = "input",
        target_column: Union[str, List[str]] = "target",
        maximize: Union[bool, List[bool]] = True,
        featurizer: Optional[Featurizer] = None,
        initializer: Optional[BOInitializer] = None,
        normalize_input: str = "original",
        dtype: torch.dtype = torch.float32
    ) -> None:
        super().__init__()
        # Paths & config
        self.data_path = data_path
        self.input_column = input_column
        self.normalize_input = normalize_input
        self.dtype = dtype

        self.target_column = (
            [target_column] # converts to list if just given as str
            if isinstance(target_column, str)
            else target_column.copy()
        )
        if isinstance(maximize, bool):
            self.maximize = [maximize] * len(self.target_column)
        else:
            if len(maximize) != len(self.target_column):
                raise ValueError("Length of 'maximize' must match number of target columns.")
            self.maximize = maximize.copy()

        # Featurizer & initializer
        self.featurizer = featurizer if featurizer is not None else Featurizer()
        self.initializer = initializer

        # Placeholders for data
        self.data: pd.DataFrame
        self.x: torch.Tensor
        self.y: torch.Tensor
        self.train_x: torch.Tensor
        self.train_y: torch.Tensor
        self.heldout_x: torch.Tensor
        self.heldout_y: torch.Tensor
        self.train_indexes: torch.Tensor
        self.heldout_indices: torch.Tensor

        self.setup()

    def load_data(self) -> None:
        """Load the CSV and optionally flip sign for minimization targets."""
        self.data = pd.read_csv(self.data_path)
        for col, do_max in zip(self.target_column, self.maximize):
            if not do_max:
                self.data[col] = -self.data[col]

    def featurize_data(self) -> None:
        """
        Converts raw data inputs (prompts/descriptors) into tensors for training models.

        Note that tokens are returned for the LLM representations.
        """
        representation = getattr(self.featurizer, 'representation', None)
        if representation == 'descriptors':
            logger.info("Featurizing in 'descriptors' mode, selecting input columns directly.")

            if not isinstance(self.input_column, list):
                raise TypeError("For 'descriptors' mode, 'input_column' must be a list of column names.")

            x_raw = self.data[self.input_column].to_numpy()
            x = x_raw

        else: # language model representations
            logger.info("Featurizing with representation %r.", representation)
            inputs = (
                [self.data[c] for c in self.input_column]
                if isinstance(self.input_column, list)
                else self.data[self.input_column]
            )
            x = self.featurizer.featurize(inputs) # obtain tokens

        y = self.data[self.target_column].to_numpy()

        self.x = torch.tensor(x, dtype=self.dtype)
        self.y = torch.tensor(y, dtype=self.dtype)

    def preprocess_data(self) -> None:
        """Raise if x has NaN or duplicate rows, or if y has NaN."""
        nan_rows = find_nan_rows(self.x)
        dup_rows = find_duplicates(self.x)
        to_remove = np.union1d(nan_rows, dup_rows)

        if len(to_remove) > 0:
            raise ValueError(
                f"Data contains {len(to_remove)} problematic rows (NaNs or duplicates). "
                f"Please clean your data at: {self.data_path}"
            )
        if torch.isnan(self.y).any():
            n = int(torch.isnan(self.y).any(dim=1).sum())
            raise ValueError(
                f"Target columns contain NaN in {n} rows. Source: {self.data_path}"
            )

    def normalize_data(self) -> None:
        """Normalize inputs according to specified strategy."""
        if self.normalize_input == "standard_scaling":
            mean, std = self.x.mean(0), self.x.std(0)
            self.x = (self.x - mean) / std
        elif self.normalize_input == "l2_max_scaling":
            self.x = self.x / self.x.norm(dim=1).max()
        elif self.normalize_input == "l2_normalize":
            self.x = self.x / self.x.norm(dim=1, keepdim=True)
        elif self.normalize_input == "original":
            pass
        else:
            raise ValueError(f"Unknown normalize_input: {self.normalize_input!r}")

    def split_data(self) -> None:
        """
        Split dataset into initial training set and heldout candidates using the initializer.

        This method uses the configured BOInitializer to select initial training points
        for Bayesian optimization. The remaining points become the heldout/candidate set
        from which the optimizer will select future experiments.

        Process:
            1. Call the initializer to select the initial training points
            2. Remaining points become the heldout set

        Sets the following attributes:
            train_indexes (torch.Tensor): Indices of initial training points (long dtype)
            heldout_indices (torch.Tensor): Indices of remaining candidate points (long dtype)
            train_x (torch.Tensor): Input features for training set
            train_y (torch.Tensor): Target values for training set
            heldout_x (torch.Tensor): Input features for heldout set
            heldout_y (torch.Tensor): Target values for heldout set

        Raises:
            AttributeError: If initializer doesn't implement fit()
        """
        if not hasattr(self.initializer, 'fit'):
            raise AttributeError(
                f"Initializer {self.initializer.__class__.__name__} must implement fit() method"
            )

        method = self.initializer.fit
        init_res = method(self.x) # fit and draw samples from self.x, llm already loaded into initializers

        if isinstance(init_res, tuple):
            init_idxs = init_res[0]
        else:
            init_idxs = init_res

        self.train_indexes = torch.as_tensor(init_idxs, dtype=torch.long) # training points collected by init method

        # set of all indices
        all_idx = set(range(self.x.shape[0]))
        held_idx = sorted(all_idx - set(init_idxs)) # set of all indexes minus set of train/init indices
        self.heldout_indices = torch.as_tensor(held_idx, dtype=torch.long)
        # Row uniqueness is a precondition for index-based splitting in split_data().

        self.train_x = self.x[self.train_indexes]
        self.train_y = self.y[self.train_indexes]
        self.heldout_x = self.x[self.heldout_indices]
        self.heldout_y = self.y[self.heldout_indices]

    def setup(self) -> None:
        self.load_data()
        self.featurize_data()
        self.preprocess_data()
        self.normalize_data()
        self.split_data()
