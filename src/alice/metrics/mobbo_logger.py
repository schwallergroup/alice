"Module for logging BO runs"
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import re
import warnings
import logging

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated
try:
    import wandb
except ImportError:
    wandb = None
    warnings.warn("wandb not installed. To log to WandB, run 'pip install wandb'")

logger = logging.getLogger(__name__)

Array = np.ndarray
Tensor = torch.Tensor
Pathlike = str | Path

ACQF_SHORT_NAMES = {
    "qLogNoisyExpectedHypervolumeImprovement": "qLogNEHVI",
}


def plot_objective_space(
    Y_cur: np.ndarray,
    PF: np.ndarray,
    iters: np.ndarray,
    obj_names: list[str],
    out_path: str,
    iteration: int,
) -> Pathlike | None:
    """
    Generating pareto front plots for each seed

    Args:
        Y_cur (np.darray): Collected Y up to current iteration
        PF: true/total pareto front Y
        iters: Iteration at which each point in Y_cur was collected
        obj_names: Names of objective for obj space plot
        out_path: Output file path
        iteration: Current BO iteration (0 = initial design, then 1..n_iters)
    """
    m = Y_cur.shape[1]
    if m < 2 or m > 3:
        logger.info("Skipping objective space plot for %d dimensions (only 2D/3D supported).", m)
        return None

    fig = plt.figure(figsize=(8.5, 6))

    # ---- Plot pareto front only ----
    if m == 2:
        ax = fig.add_subplot(111)
        ax.scatter(PF[:, 0], PF[:, 1], color="black", marker='x', s=25, label="Global Pareto Front")
    else:  # m == 3
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(PF[:, 0], PF[:, 1], PF[:, 2], color="black", marker='x', s=25, label="Global Pareto Front")
        ax.set_zlabel(obj_names[2])

    ax.set_xlabel(obj_names[0])
    ax.set_ylabel(obj_names[1])
    ax.set_title(f"Bayesian Optimization History (Iteration {iteration})")

    # ---- Plot initial samples only with grey colourmap ----
    initial_points = Y_cur[iters == 0]
    if initial_points.shape[0] > 0:
        ax.scatter(*initial_points.T, color="lightgray", s=35, label="Initial Samples")

    # ---- Plot intermediate points with plasma colourmap ----
    unique_iters = np.unique(iters)
    iters_nonzero = unique_iters[unique_iters > 0]

    if iters_nonzero.size > 0:
        cmap_full = cm.get_cmap("plasma_r")
        cmap = mcolors.LinearSegmentedColormap.from_list("trunc_plasma_r", cmap_full(np.linspace(0.15, 1.0, 256)))

        if iters_nonzero.size == 1:
            color_map = {iters_nonzero[0]: cmap(0.7)}
        else:
            norm = mcolors.Normalize(vmin=iters_nonzero.min(), vmax=iters_nonzero.max())
            color_map = {t: cmap(norm(t)) for t in iters_nonzero}

        for t in iters_nonzero:
            color = color_map[t]
            batch_points = Y_cur[iters == t]

            if batch_points.size == 0:
                continue

            ax.scatter(*batch_points.T, color=color, label=f"iter {t}", s=35)

            for pt in PF:
                if np.any(np.all(np.isclose(batch_points, pt, atol=1e-8), axis=1)):
                    ax.scatter(
                        *pt.T, color=color, marker="*", s=120,
                        edgecolor="black", linewidth=0.6
                    )

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="center left", bbox_to_anchor=(1.05, 0.5), fontsize="small", title="Batches")

    fig.subplots_adjust(right=0.75)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

    return out_path


def _ensure_dir(path: Pathlike):
    os.makedirs(path, exist_ok = True)


def _to_np(tensor: Tensor) -> Array:
    return tensor.detach().cpu().numpy()


def _clean_name(name: str) -> str:
    return re.sub(r'[^\w\-_\. ]', '_', str(name))


class MobboCsvLogger:
    def __init__(self):
        self._metrics_defined: bool = False
        self._last_n_experiments_logged: int = -1
   
    def _parse_config(self, config: Optional[Dict]) -> Dict[str, Any]:
        if config is None:
            config = {}

        parsed = {
            "surrogate_name": "UNKNOWN_SURROGATE",
            "representation": "unknown_rep",
            "featurizer_model": "unknown_model",
            "batch_size": 0,
            "obj_names": [],
        }

        try:
            bo_init = config.get("bo", {}).get("init_args", {})
            parsed["batch_size"] = int(bo_init.get("batch_size", 0))
        except Exception:
            pass

        try:
            s = config.get("surrogate_model", {}).get("class_path", "")
            if s:
                parsed["surrogate_name"] = s.split(".")[-1]
        except Exception:
            pass

        try:
            data_init = config.get("data", {}).get("init_args", {})
            feat_init = data_init.get("featurizer", {}).get("init_args", {})
            rep = feat_init.get("representation", None)
            fm = feat_init.get("model_name", "")

            if rep is not None:
                parsed["representation"] = str(rep)
            elif fm:
                parsed["representation"] = "embeddings"
            else:
                parsed["representation"] = "descriptors"

            if fm:
                parsed["featurizer_model"] = _clean_name(str(fm))

        except Exception:
            pass

        data_init = config.get("data", {}).get("init_args", {})
        parsed["obj_names"] = data_init.get("target_column", [])

        return parsed

    def _save_timeseries_plot(
            self,
            df: pd.DataFrame,
            x: str,
            y: str,
            title: str,
            xlabel: str,
            ylabel: str,
            out_path: Path,
            marker: Optional[str] = None,
        ):
        """
        Standardized line plot generator for metrics tracking across iterations
        """
        plt.figure(figsize=(6, 4))
        plt.plot(df[x], df[y], marker=marker)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)

        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()

    def log_iteration(
            self,
            current_y: Tensor,
            Pareto_full: Tensor,
            sample_iters: Tensor,
            iteration: int,
            out_root: str,
            model_name: str,
            seed: int,
            ref_point: Tuple,
            acqf_name: str,
            log_wandb: bool,
            config: Dict,
        ):
        """
        Args:
            current_y (torch.Tensor): Current train_y from BaseDataModule, updated at each iteration
            Pareto_full (torch.Tensor): Ground-truth Pareto front of the full Y dataset,
                computed once in ``train_master``. Used only for ``hv_star``, the
                normalising constant behind ``hv_percent``.
            sample_iters (torch.Tensor): Stored values of each collected x data point with the iteration collected
            iteration (int): BO iteration number (0 = initial design, then 1..n_iters)
            out_root (str): Output path file
            model_name (str): LLM model name taken from configs
            seed (int): Seed taken from config
            ref_point (Tuple): Obtained from model config
            acqf_name (str): Acquisition function class name (last dotted token of
                ``acquisition.class_path``); shortened via ``ACQF_SHORT_NAMES``
            config (Dict): Read config file
        """
        m = current_y.shape[1] # num_objectives

        if len(ref_point) != m:
            raise ValueError(
                f"Dimension of ref_point ({len(ref_point)}) must match objectives ({m})."
            )

        # parse config run file
        cfg = self._parse_config(config)
        cfg_model_name = cfg["featurizer_model"]

        featurizer_model = _clean_name(model_name if model_name != "unknown_model" else cfg_model_name)

        acqf_short = ACQF_SHORT_NAMES.get(acqf_name, _clean_name(acqf_name))
        surrogate_short = cfg["surrogate_name"].split(".")[-1]
        rep_short = _clean_name(cfg["representation"])
        batch_tag = f"batch_q{cfg['batch_size']}"

        out_dir = (
            Path(out_root)
            / f"{surrogate_short}-{acqf_short}-{rep_short}-{featurizer_model}"
            / f"seed={seed}"
            / batch_tag
        )
        _ensure_dir(out_dir)

        # Converting data to numpy - we have access to current y selected + full, PF full and n_iter of BO
        Y_cur_np = _to_np(current_y)
        PF_np = _to_np(Pareto_full)
        iters_np = _to_np(sample_iters).astype(int)

        # Generic per-objective column names used in the saved CSVs.
        obj_cols = [f"obj{i}" for i in range(m)]
        obj_names = cfg["obj_names"] if len(cfg["obj_names"]) == m else obj_cols

        # Plotting the objective space
        front_map_fname = out_dir / f"front_map_multiobj_seed{seed}_{batch_tag}.png"
        plot_objective_space(
            Y_cur=Y_cur_np,
            PF=PF_np,
            iters=iters_np,
            obj_names=obj_names,
            out_path=front_map_fname,
            iteration=iteration,
        )

        Y_t = torch.as_tensor(Y_cur_np, dtype=torch.float64)
        PF_t = torch.as_tensor(PF_np, dtype=torch.float64)
        st_t = torch.as_tensor(iters_np, dtype=torch.long)
        ref_t = torch.as_tensor(ref_point, dtype=torch.float64)
        hv_calc = Hypervolume(ref_point=ref_t) # calculate hypervolume

        # ---- Hypervolume of true ground truth ----
        hv_star = float(hv_calc.compute(PF_t)) if PF_t.numel() > 0 else 0.0

        tmin, tmax = int(st_t.min().item()), int(st_t.max().item()) # minimum iteration, max iterations
        rows_hv = []
        cum_count = 0

        for t in range(tmin, tmax + 1):
            cum_count += int((st_t == t).sum()) # cumulative experiments sampled

            pts = Y_t[st_t <= t]
            hv_t = 0.0
            if pts.numel() > 0:
                pf_t = pts[is_non_dominated(pts)]
                if pf_t.numel() > 0:
                    hv_t = float(hv_calc.compute(pf_t))

            rows_hv.append({
                "iteration": t,
                "n_experiments": cum_count,
                "hypervolume": hv_t,
                "hv_percent": (100.0 * hv_t / hv_star) if hv_star > 0 else 0.0,
            })

        hv_df = pd.DataFrame(rows_hv)
        hv_df.to_csv(out_dir / f"hypervolume_seed{seed}_{batch_tag}.csv", index=False)

        final_metrics = {
            "hypervolume": hv_df["hypervolume"].iloc[-1],
            "hv_percent": hv_df["hv_percent"].iloc[-1],
        }

        title_suffix = f"— iteration={iteration}"
        plots_to_make = [
            ("hv_percent_vs_iteration", "iteration", f"HV% of Pareto front {title_suffix}",
             "BO iteration", "o"),
            ("hv_percent_vs_experiments", "n_experiments", f"HV% vs # experiments {title_suffix}",
             "# experiments", "o"),
        ]

        plot_paths = {}
        for out_name, x, title, xlabel, marker in plots_to_make:
            out_path = out_dir / f"{out_name}_seed{seed}_{batch_tag}.png"
            plot_paths[out_name] = out_path
            try:
                self._save_timeseries_plot(
                    hv_df, x, "hv_percent", title, xlabel, "% of max hypervolume", out_path, marker
                )
            except Exception:
                logger.exception("Failed to generate plot '%s'", title)

        if log_wandb:
            self._log_to_wandb(
                iteration=iteration,
                acqf_short=acqf_short,
                cfg=cfg,
                seed=seed,
                hv_df=hv_df,
                final_metrics=final_metrics,
                plot_paths=plot_paths,
                front_map_path=front_map_fname
            )

        logger.info("Iteration %d metrics saved to %s", iteration, out_dir)

    def _log_to_wandb(
            self,
            iteration: int,
            acqf_short: str,
            cfg: Dict,
            seed: int,
            hv_df: pd.DataFrame,
            final_metrics: Dict,
            plot_paths: Dict[str, Path],
            front_map_path: Path,
        ):

        if wandb is None or wandb.run is None:
            logger.warning("wandb.run is None or module not imported. Skipping WandB logging.")
            return
        
        try:
            wandb_metrics = [
                "hv_percent",
                "hypervolume",
            ]

            if not self._metrics_defined:
                wandb.define_metric("n_experiments")
                for m_name in wandb_metrics:
                    wandb.define_metric(m_name, step_metric="n_experiments")
                self._metrics_defined = True

            timeline_df = hv_df.sort_values("n_experiments")

            metrics_to_log = [
                m for m in wandb_metrics if m in timeline_df.columns
            ]

            new_rows = timeline_df[timeline_df["n_experiments"] > self._last_n_experiments_logged]
            
            for _, row in new_rows.iterrows():
                payload = {"n_experiments": int(row.n_experiments)}
                for col in metrics_to_log: 
                    if col in row and not pd.isna(row[col]):
                        payload[col] = float(row[col])
                
                wandb.log(payload)
                self._last_n_experiments_logged = int(row.n_experiments)

            summary_log = {
                "iteration": iteration,
                "batch_size": cfg["batch_size"],
            }

            for name, value in final_metrics.items():
                if not pd.isna(value):
                    summary_log[f"{acqf_short}/{name}_final"] = float(value)
            
            if front_map_path.exists():
                summary_log[f"{acqf_short}/front_map_plot"] = wandb.Image(str(front_map_path))
            
            for name, path in plot_paths.items():
                if path.exists():
                    summary_log[f"{acqf_short}/{name}_plot"] = wandb.Image(str(path))
            
            wandb.log(summary_log, commit=False)

            try:
                wandb.config.update({
                    "acqf_name": acqf_short,
                    "surrogate": cfg["surrogate_name"],
                    "batch_size": cfg["batch_size"],
                    "representation": cfg["representation"],
                    "featurizer_model": cfg["featurizer_model"],
                    "seed": seed,
                }, allow_val_change=True)
            except Exception:
                logger.warning("Failed to update wandb config", exc_info=True)

            wandb.log({}, commit=True)

        except Exception:
            logger.exception("WandB logging failed; skipping")
