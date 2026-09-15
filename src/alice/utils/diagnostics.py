"Diagnostic prints for BaseDataModule and BotorchOptimizer"
import torch
import numpy as np

from alice.data.module import BaseDataModule


def print_diagnostic_summary(dm: BaseDataModule, rep_short, trainable, model_name):
    """
    Prints a detailed diagnostic summary of the data and configuration.

    Args:
        dm (object): The data manager object holding the splits.
        rep_short (str): Short representation identifier (e.g., "embeddings").
        trainable (bool): Flag indicating if the model is trainable.
        model_name (str): The name of the model being used.
    """

    train_x_shape = tuple(getattr(dm, 'train_x', torch.empty(0)).shape)
    train_y_shape = tuple(getattr(dm, 'train_y', torch.empty(0)).shape)
    train_indices_size = getattr(dm, 'train_indexes', np.array([])).size()
    train_indices_sample = getattr(dm, 'train_indexes', np.array([]))[:5]

    heldout_x_shape = tuple(getattr(dm, 'heldout_x', torch.empty(0)).shape)
    heldout_y_shape = tuple(getattr(dm, 'heldout_y', torch.empty(0)).shape)
    heldout_indices_size = getattr(dm, 'heldout_indices', np.array([])).size()

    y_min_obj1 = dm.y[:, 0].min().item() # hard coded for two objectives for now
    y_max_obj1 = dm.y[:, 0].max().item()
    y_min_obj2 = dm.y[:, 1].min().item()
    y_max_obj2 = dm.y[:, 1].max().item()

    if not np.array_equal(dm.data.index.values, np.arange(len(dm.data))):
        raise ValueError(
            f"DataFrame index is not a contiguous 0..N-1 range. Positional indices "
            f"used by the BO loop would not align with DataFrame rows. "
            f"Source: {dm.data_path}"
        )

    # Reported, not validated: for LLM representations x holds tokens, so its width
    # is the sequence length rather than the model's embedding size.
    feature_dim_status = f"Actual: {dm.x.shape[1]}"
    objective_status = f"Actual: {dm.y.shape[1]}"

    if rep_short == "descriptors":
        mode_info = "DESCRIPTORS"
    else:
        mode_info = f"EMBEDDINGS (rep={rep_short}, model_name={model_name}, trainable={trainable})"

    summary_table = f"""
    +-------------------------------------------------------+
    |           Experiment Diagnostic Summary               |
    +=======================================================+
    | OVERALL SUMMARY                                       |
    +-------------------------------------------------------+
    | Total X (features) Shape: {tuple(dm.x.shape)}
    | Total Y (targets) Shape:  {tuple(dm.y.shape)}
    | Y (obj 1) Range: {y_min_obj1:.3f} to {y_max_obj1:.3f}
    | Y (obj 2) Range: {y_min_obj2:.3f} to {y_max_obj2:.3f}
    | Total Dataset Rows: {len(dm.data)}
    | Maximize Flags: {getattr(dm, "maximize", None)}
    | Target Columns: {dm.target_column}
    +-------------------------------------------------------+
    | CONFIGURATION & VALIDATION                            |
    +-------------------------------------------------------+
    | Run Mode: {mode_info}
    | Feature Dimensions: {feature_dim_status}
    | Objective Dimensions: {objective_status}
    | Featurizer Class: {type(dm.featurizer).__name__}
    +-------------------------------------------------------+
    | INITIAL DATA SPLIT                                    |
    +-------------------------------------------------------+
    | Train X Shape:      {train_x_shape}
    | Train Y Shape:      {train_y_shape}
    | Num. Train Indices: {train_indices_size}
    | Sample Indices:     {train_indices_sample}
    |-------------------------------------------------------|
    | Held-out X Shape:   {heldout_x_shape}
    | Held-out Y Shape:   {heldout_y_shape}
    | Num. Held-out Indices: {heldout_indices_size}
    +-------------------------------------------------------+
    """
    print(summary_table)


def print_bo_diagnostic(dm: BaseDataModule, config, bo, model_name, acqf_name):
    """
    Prints a detailed diagnostic summary of the Bayesian Optimization configuration.

    Args:
        dm (object): The data manager object.
        config (object): The configuration dictionary.
        bo (object): The Bayesian Optimization optimizer object.
        model_name (str): The name of the model being used.
        acqf_name (str): The name of the acquisition function.
    """
    acq_cfg = config.get("acquisition", {})
    init_args = acq_cfg.get("init_args", {})

    ref_point_info = "Not specified (default for non-hypervolume based acquisition functions)"
    try:
        if "ref_point" in init_args:
            # as_tensor accepts lists, tuples, ndarrays and tensors alike
            rp_t = torch.as_tensor(init_args["ref_point"], dtype=dm.y.dtype)
            ref_point_info = (f"Value: {tuple(rp_t.cpu().numpy())}, "
                              f"dtype: {rp_t.dtype}, "
                              f"shape: {tuple(rp_t.shape)}")
    except Exception as e:
        ref_point_info = f"ERROR: Conversion failed ({e})"

    initial_train_size = len(dm.train_indexes)
    initial_heldout_size = len(dm.heldout_indices)

    summary_table = f"""
    +-------------------------------------------------------+
    |         Bayesian Optimization Diagnostic Summary      |
    +=======================================================+
    | BO CONFIGURATION                                      |
    +-------------------------------------------------------+
    | BO Optimizer Class: {type(bo).__name__}
    | Acquisition Function: {acqf_name}
    | Model Name:         {model_name}
    | Batch Size (q):     {config['bo']['init_args'].get('batch_size', 'N/A')}
    +-------------------------------------------------------+
    | ACQUISITION FUNCTION PARAMETERS                       |
    +-------------------------------------------------------+
    | ACQF reference Point:    {ref_point_info}
    +-------------------------------------------------------+
    | INITIAL DATA STATE                                    |
    +-------------------------------------------------------+
    | Initial Training Size: {initial_train_size}
    | Initial Held-out Size: {initial_heldout_size}
    +-------------------------------------------------------+
    """
    print(summary_table)
