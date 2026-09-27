"""Training entry point for the GRACE adaptive downscaling baseline."""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
import xarray as xr
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common import configure_logging, ensure_dir, write_dataset  # noqa: E402
from scripts.train.dataset import GraceTileDataset  # noqa: E402
from scripts.train.losses import build_total_loss  # noqa: E402
from utils.mass_closure import project_to_coarse_constraint_torch  # noqa: E402
from scripts.train.model import build_model_from_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "downscaling_baseline.yaml",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume-last", action="store_true", help="Resume from training/output_dir/checkpoints/last.pt if present.")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    """Load YAML config."""
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_determinism(enabled: bool) -> None:
    """Request reproducible kernels for matched-ablation training runs."""
    if not enabled:
        return
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def resolve_device(requested: str) -> torch.device:
    """Resolve user-selected device."""
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_loader(dataset: GraceTileDataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    """Create DataLoader with conservative defaults for Zarr-backed tiles."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    """Move tensor values in a batch dictionary to the training device."""
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def build_datasets(config: dict) -> tuple[GraceTileDataset, GraceTileDataset, GraceTileDataset]:
    """Instantiate train/val/test datasets."""
    dataset_cfg = config["dataset"]
    channel_overrides = dataset_cfg.get("channel_overrides")
    weight_files = dataset_cfg.get("weight_files")
    regional_weights = config.get("regional_weights")
    train_ds = GraceTileDataset(
        dataset_cfg["zarr_path"],
        split="train",
        tile_size=dataset_cfg["tile_size"],
        stride=dataset_cfg["stride_train"],
        coarse_factor=dataset_cfg["coarse_factor"],
        min_valid_fraction=dataset_cfg["min_valid_fraction"],
        min_land_fraction=dataset_cfg["min_land_fraction"],
        channel_overrides=channel_overrides,
        weight_files=weight_files,
        regional_weights=regional_weights,
    )
    val_ds = GraceTileDataset(
        dataset_cfg["zarr_path"],
        split="val",
        tile_size=dataset_cfg["tile_size"],
        stride=dataset_cfg["stride_eval"],
        coarse_factor=dataset_cfg["coarse_factor"],
        min_valid_fraction=dataset_cfg["min_valid_fraction"],
        min_land_fraction=dataset_cfg["min_land_fraction"],
        channel_overrides=channel_overrides,
        weight_files=weight_files,
        regional_weights=regional_weights,
    )
    test_ds = GraceTileDataset(
        dataset_cfg["zarr_path"],
        split="test",
        tile_size=dataset_cfg["tile_size"],
        stride=dataset_cfg["stride_eval"],
        coarse_factor=dataset_cfg["coarse_factor"],
        min_valid_fraction=dataset_cfg["min_valid_fraction"],
        min_land_fraction=dataset_cfg["min_land_fraction"],
        channel_overrides=channel_overrides,
        weight_files=weight_files,
        regional_weights=regional_weights,
    )
    return train_ds, val_ds, test_ds


def save_weight_diagnostics(dataset: GraceTileDataset, output_dir: Path) -> None:
    """Save alpha/beta regional diagnostics before training starts."""
    diagnostics_dir = ensure_dir(output_dir / "diagnostics")
    weight_maps = dataset.get_weight_diagnostics()
    glacier_mask = (weight_maps["glacier_fraction"] >= 0.01).astype(np.float32)
    non_glacier_mask = ((weight_maps["land_mask"] > 0) & (weight_maps["glacier_fraction"] < 0.001)).astype(np.float32)
    arid_mask = ((weight_maps["land_mask"] > 0) & (weight_maps["aridity_mask"] >= 0.5)).astype(np.float32)
    humid_mask = ((weight_maps["land_mask"] > 0) & (weight_maps["aridity_mask"] <= 0.1)).astype(np.float32)
    high_human_mask = ((weight_maps["land_mask"] > 0) & (weight_maps["human_activity_index"] >= 0.05)).astype(np.float32)
    low_human_mask = ((weight_maps["land_mask"] > 0) & (weight_maps["human_activity_index"] <= 0.001)).astype(np.float32)

    def masked_mean_np(field: np.ndarray, mask: np.ndarray) -> float:
        valid = np.isfinite(field) & mask.astype(bool)
        if not np.any(valid):
            return float("nan")
        return float(np.nanmean(field[valid]))

    summary_rows = [
        {
            "weight_name": "alpha_original",
            "glacier_mean": masked_mean_np(weight_maps["alpha_original"], glacier_mask),
            "non_glacier_mean": masked_mean_np(weight_maps["alpha_original"], non_glacier_mask),
            "arid_mean": masked_mean_np(weight_maps["alpha_original"], arid_mask),
            "humid_mean": masked_mean_np(weight_maps["alpha_original"], humid_mask),
            "high_human_mean": masked_mean_np(weight_maps["alpha_original"], high_human_mask),
            "low_human_mean": masked_mean_np(weight_maps["alpha_original"], low_human_mask),
        },
        {
            "weight_name": "alpha_glacier_preserved",
            "glacier_mean": masked_mean_np(weight_maps["alpha_adjusted"], glacier_mask),
            "non_glacier_mean": masked_mean_np(weight_maps["alpha_adjusted"], non_glacier_mask),
            "arid_mean": masked_mean_np(weight_maps["alpha_adjusted"], arid_mask),
            "humid_mean": masked_mean_np(weight_maps["alpha_adjusted"], humid_mask),
            "high_human_mean": masked_mean_np(weight_maps["alpha_adjusted"], high_human_mask),
            "low_human_mean": masked_mean_np(weight_maps["alpha_adjusted"], low_human_mask),
        },
        {
            "weight_name": "beta_original",
            "glacier_mean": masked_mean_np(weight_maps["beta_original"], glacier_mask),
            "non_glacier_mean": masked_mean_np(weight_maps["beta_original"], non_glacier_mask),
            "arid_mean": masked_mean_np(weight_maps["beta_original"], arid_mask),
            "humid_mean": masked_mean_np(weight_maps["beta_original"], humid_mask),
            "high_human_mean": masked_mean_np(weight_maps["beta_original"], high_human_mask),
            "low_human_mean": masked_mean_np(weight_maps["beta_original"], low_human_mask),
        },
        {
            "weight_name": "beta_glacier_preserved",
            "glacier_mean": masked_mean_np(weight_maps["beta_adjusted"], glacier_mask),
            "non_glacier_mean": masked_mean_np(weight_maps["beta_adjusted"], non_glacier_mask),
            "arid_mean": masked_mean_np(weight_maps["beta_adjusted"], arid_mask),
            "humid_mean": masked_mean_np(weight_maps["beta_adjusted"], humid_mask),
            "high_human_mean": masked_mean_np(weight_maps["beta_adjusted"], high_human_mask),
            "low_human_mean": masked_mean_np(weight_maps["beta_adjusted"], low_human_mask),
        },
    ]

    summary_csv = diagnostics_dir / "alpha_beta_summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    maps_ds = xr.Dataset(
        {
            "alpha_original": (("lat", "lon"), weight_maps["alpha_original"]),
            "alpha_glacier_preserved": (("lat", "lon"), weight_maps["alpha_adjusted"]),
            "beta_original": (("lat", "lon"), weight_maps["beta_original"]),
            "beta_glacier_preserved": (("lat", "lon"), weight_maps["beta_adjusted"]),
            "glacier_mask": (("lat", "lon"), weight_maps["glacier_fraction"]),
            "aridity_mask": (("lat", "lon"), weight_maps["aridity_mask"]),
            "human_activity_index": (("lat", "lon"), weight_maps["human_activity_index"]),
        },
        coords={
            "lat": dataset.ds["lat"].values,
            "lon": dataset.ds["lon"].values,
        },
    )
    write_dataset(maps_ds, diagnostics_dir / "alpha_beta_maps.nc")

    fig, axes = plt.subplots(2, 4, figsize=(16, 7), constrained_layout=True)
    panel_specs = [
        ("alpha_original", "alpha_original"),
        ("alpha_adjusted", "alpha_glacier_preserved"),
        ("beta_original", "beta_original"),
        ("beta_adjusted", "beta_glacier_preserved"),
        ("glacier_fraction", "glacier_mask"),
        ("aridity_mask", "aridity_mask"),
        ("human_activity_index", "human_activity_index"),
        ("land_mask", "land_mask"),
    ]
    for ax, (key, title) in zip(axes.flat, panel_specs, strict=True):
        image = ax.imshow(weight_maps[key], cmap="viridis")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(diagnostics_dir / "alpha_beta_glacier_diagnostics.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_training_diagnostics(metrics_csv: Path, output_dir: Path) -> None:
    """Render training-loss diagnostics from the epoch metrics CSV."""
    with metrics_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["phase"] == "epoch"]
    if not rows:
        return

    diagnostics_dir = ensure_dir(output_dir / "diagnostics")
    epochs = [int(row["epoch"]) for row in rows]

    def as_float(column: str) -> tuple[list[float], list[float]]:
        train_vals = [float(row.get(f"train_{column}", 0.0) or 0.0) for row in rows]
        val_vals = [float(row.get(f"val_{column}", 0.0) or 0.0) for row in rows]
        return train_vals, val_vals

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    panels = [
        ("loss_total", "total_loss"),
        ("loss_mass", "loss_mass"),
        ("loss_value_adaptive", "loss_value_adaptive"),
        ("loss_gradient_adaptive", "loss_gradient_adaptive"),
    ]
    for ax, (column, title) in zip(axes.flat, panels, strict=True):
        train_vals, val_vals = as_float(column)
        ax.plot(epochs, train_vals, label="train")
        ax.plot(epochs, val_vals, label="val")
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.2)
    axes.flat[0].legend()
    fig.savefig(diagnostics_dir / "loss_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), constrained_layout=True)
    spectral_panels = [
        ("loss_spectral", "loss_spectral"),
        ("loss_low_grace", "loss_low_grace"),
        ("loss_high_wghm", "loss_high_wghm"),
        ("loss_basin", "loss_basin"),
    ]
    for ax, (column, title) in zip(axes.flat, spectral_panels, strict=True):
        train_vals, val_vals = as_float(column)
        ax.plot(epochs, train_vals, label="train")
        ax.plot(epochs, val_vals, label="val")
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.2)
    axes.flat[0].legend()
    fig.savefig(diagnostics_dir / "spectral_loss_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def epoch_pass(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    loss_cfg: dict,
    max_batches: int | None = None,
    log_interval_batches: int | None = None,
    logger=None,
) -> dict[str, float]:
    """Run one train or eval epoch."""
    is_train = optimizer is not None
    model.train(is_train)
    totals: dict[str, float] | None = None
    batches = 0
    start_time = time.time()

    for batch in loader:
        batch = move_batch(batch, device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train):
            forward_kwargs = {
                "coarse_context": batch["coarse_context"],
                "valid_mask": batch.get("valid_mask"),
                "area_weight": batch.get("area_weight"),
            }
            if getattr(model, "gate_mode", None) == "oracle":
                forward_kwargs["oracle_gate"] = batch.get("reliability_truth")
            pred = model(batch["inputs"], **forward_kwargs)
            model_kl = getattr(model, "last_kl_loss", None)
            if model_kl is not None:
                batch["model_kl_loss"] = model_kl
            model_gate = getattr(model, "last_gate", None)
            if model_gate is not None:
                batch["model_gate"] = model_gate
            projection_cfg = loss_cfg.get("strict_projection", {})
            projection_diagnostics = None
            if bool(projection_cfg.get("enabled", False)):
                pred, projection_diagnostics = project_to_coarse_constraint_torch(
                    pred,
                    batch["target_coarse"],
                    batch["valid_mask"],
                    factor=int(loss_cfg["coarse_factor"]),
                    area_weights=batch.get("area_weight"),
                    refinement_steps=int(projection_cfg.get("refinement_steps", 2)),
                )
            total_loss, components = build_total_loss(pred, batch, loss_cfg=loss_cfg)
            if projection_diagnostics is not None:
                components["projection_mean_abs_after"] = projection_diagnostics[
                    "mass_closure_error_after"
                ].detach()
                components["projection_max_abs_after"] = projection_diagnostics[
                    "mass_closure_max_abs_after"
                ].detach()
            if is_train:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        if totals is None:
            totals = {key: 0.0 for key in components}
        for key in totals:
            totals[key] += float(components[key].detach().cpu())
        batches += 1
        if logger is not None and log_interval_batches and batches % log_interval_batches == 0:
            elapsed = time.time() - start_time
            logger.info(
                "Batch progress | mode=%s | batches=%d | avg_seconds_per_batch=%.3f | latest_total=%.5f",
                "train" if is_train else "eval",
                batches,
                elapsed / batches,
                float(components["loss_total"].detach().cpu()),
            )
        if max_batches is not None and batches >= max_batches:
            break

    if batches == 0:
        raise ValueError("No batches were produced by the DataLoader.")
    assert totals is not None
    return {key: value / batches for key, value in totals.items()}


def save_metrics_row(csv_path: Path, row: dict[str, object], fieldnames: list[str]) -> None:
    """Append one row of metrics to CSV."""
    ensure_dir(csv_path.parent)
    write_header = not csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def save_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: AdamW,
    scheduler: CosineAnnealingLR,
    metrics: dict[str, float],
    config: dict,
) -> None:
    """Save a training checkpoint."""
    ensure_dir(path.parent)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "metrics": metrics,
            "config": config,
        },
        path,
    )


def load_resume_state(
    checkpoint_path: Path,
    best_checkpoint_path: Path,
    model: nn.Module,
    optimizer: AdamW,
    scheduler: CosineAnnealingLR,
    device: torch.device,
    logger,
) -> tuple[int, float, int]:
    """Restore model/optimizer/scheduler state and derive the next epoch."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    else:
        completed_epochs = int(checkpoint.get("epoch", 0))
        for _ in range(completed_epochs):
            scheduler.step()

    start_epoch = int(checkpoint.get("epoch", 0)) + 1
    best_val = float("inf")
    best_epoch = -1
    if best_checkpoint_path.exists():
        best_checkpoint = torch.load(best_checkpoint_path, map_location=device)
        best_epoch = int(best_checkpoint.get("epoch", -1))
        best_metrics = best_checkpoint.get("metrics", {}).get("val", {})
        best_val = float(best_metrics.get("loss_total", float("inf")))
    else:
        val_metrics = checkpoint.get("metrics", {}).get("val", {})
        best_val = float(val_metrics.get("loss_total", float("inf")))
        best_epoch = int(checkpoint.get("epoch", -1))

    logger.info(
        "Resuming training from %s | next_epoch=%d | best_epoch=%d | best_val=%.5f",
        checkpoint_path,
        start_epoch,
        best_epoch,
        best_val,
    )
    return start_epoch, best_val, best_epoch


def load_init_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    device: torch.device,
    logger,
) -> None:
    """Warm-start model weights from a prior checkpoint without resuming optimizer state."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info("Initialized model weights from %s", checkpoint_path)


def main() -> None:
    """Entry point."""
    args = parse_args()
    logger = configure_logging()
    config = load_config(args.config)
    training_cfg = config["training"]
    if "num_threads" in training_cfg:
        torch.set_num_threads(int(training_cfg["num_threads"]))
    if "interop_threads" in training_cfg:
        torch.set_num_interop_threads(int(training_cfg["interop_threads"]))

    def metric_or_zero(metrics: dict[str, float], key: str) -> float:
        return float(metrics.get(key, 0.0))
    seed = int(config["training"]["seed"])
    set_seed(seed)
    configure_determinism(bool(training_cfg.get("deterministic", False)))
    device = resolve_device(args.device)
    logger.info("Using device: %s", device)

    output_dir = Path(config["training"]["output_dir"])
    ensure_dir(output_dir)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, test_ds = build_datasets(config)
    save_weight_diagnostics(train_ds, output_dir)
    logger.info(
        "Dataset tiles - train: %d, val: %d, test: %d",
        len(train_ds),
        len(val_ds),
        len(test_ds),
    )

    loader_cfg = training_cfg
    train_loader = make_loader(train_ds, batch_size=loader_cfg["batch_size"], shuffle=True, num_workers=loader_cfg["num_workers"])
    val_loader = make_loader(val_ds, batch_size=loader_cfg["batch_size"], shuffle=False, num_workers=loader_cfg["num_workers"])
    test_loader = make_loader(test_ds, batch_size=loader_cfg["batch_size"], shuffle=False, num_workers=loader_cfg["num_workers"])
    max_train_batches = loader_cfg.get("max_train_batches")
    max_eval_batches = loader_cfg.get("max_eval_batches")
    log_interval_batches = loader_cfg.get("log_interval_batches")

    # Dataset construction must not make initialization differ across ablations.
    # Reset immediately before building the shared architecture so seed-matched
    # M0/M1/M2/M3/M5 start from exactly the same parameter tensors.
    set_seed(seed)
    model_cfg = config["model"]
    model = build_model_from_config(train_ds.in_channels, model_cfg).to(device)
    optimizer = AdamW(model.parameters(), lr=loader_cfg["learning_rate"], weight_decay=loader_cfg["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=loader_cfg["epochs"], eta_min=loader_cfg.get("min_learning_rate", 1e-5))

    metrics_csv = output_dir / "metrics.csv"
    metrics_fieldnames = [
        "phase",
        "epoch",
        "best_epoch",
        "learning_rate",
        "train_loss_total",
        "train_loss_mass",
        "train_loss_value_adaptive",
        "train_loss_gradient_adaptive",
        "train_loss_fine",
        "train_loss_spectral",
        "train_loss_low_grace",
        "train_loss_high_wghm",
        "train_loss_basin",
        "train_loss_gate_supervision",
        "train_loss_gate_variance",
        "train_gate_mae",
        "train_gate_std",
        "train_gate_saturation_fraction",
        "train_loss_kl",
        "train_dynamic_lambda",
        "train_projection_mean_abs_after",
        "train_projection_max_abs_after",
        "val_loss_total",
        "val_loss_mass",
        "val_loss_value_adaptive",
        "val_loss_gradient_adaptive",
        "val_loss_fine",
        "val_loss_spectral",
        "val_loss_low_grace",
        "val_loss_high_wghm",
        "val_loss_basin",
        "val_loss_gate_supervision",
        "val_loss_gate_variance",
        "val_gate_mae",
        "val_gate_std",
        "val_gate_saturation_fraction",
        "val_loss_kl",
        "val_dynamic_lambda",
        "val_projection_mean_abs_after",
        "val_projection_max_abs_after",
        "test_loss_total",
        "test_loss_mass",
        "test_loss_value_adaptive",
        "test_loss_gradient_adaptive",
        "test_loss_fine",
        "test_loss_spectral",
        "test_loss_low_grace",
        "test_loss_high_wghm",
        "test_loss_basin",
        "test_loss_gate_supervision",
        "test_loss_gate_variance",
        "test_gate_mae",
        "test_gate_std",
        "test_gate_saturation_fraction",
        "test_loss_kl",
        "test_dynamic_lambda",
        "test_projection_mean_abs_after",
        "test_projection_max_abs_after",
    ]
    best_val = float("inf")
    best_epoch = -1
    best_checkpoint = output_dir / "checkpoints" / "best.pt"
    last_checkpoint = output_dir / "checkpoints" / "last.pt"
    start_epoch = 1

    if args.resume_last and last_checkpoint.exists():
        start_epoch, best_val, best_epoch = load_resume_state(
            checkpoint_path=last_checkpoint,
            best_checkpoint_path=best_checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            logger=logger,
        )
    elif args.resume_last:
        logger.warning("Requested --resume-last but checkpoint does not exist: %s", last_checkpoint)
    else:
        init_checkpoint = training_cfg.get("init_checkpoint")
        if init_checkpoint:
            init_path = Path(init_checkpoint)
            if not init_path.is_absolute():
                init_path = (PROJECT_ROOT / init_path).resolve()
            if init_path.exists():
                load_init_checkpoint(init_path, model=model, device=device, logger=logger)
            else:
                logger.warning("Requested init_checkpoint but file does not exist: %s", init_path)

    for epoch in range(start_epoch, loader_cfg["epochs"] + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        train_metrics = epoch_pass(
            model,
            train_loader,
            optimizer,
            device,
            config["loss"],
            max_batches=max_train_batches,
            log_interval_batches=log_interval_batches,
            logger=logger,
        )
        val_metrics = epoch_pass(
            model,
            val_loader,
            None,
            device,
            config["loss"],
            max_batches=max_eval_batches,
            log_interval_batches=log_interval_batches,
            logger=logger,
        )
        row = {
            "phase": "epoch",
            "epoch": epoch,
            "best_epoch": "",
            "learning_rate": current_lr,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        save_metrics_row(metrics_csv, row, metrics_fieldnames)
        logger.info(
            "Epoch %d | train_total=%.5f | val_total=%.5f | val_mass=%.5f | val_value=%.5f | val_gradient=%.5f | val_spectral=%.5f | val_basin=%.5f",
            epoch,
            metric_or_zero(train_metrics, "loss_total"),
            metric_or_zero(val_metrics, "loss_total"),
            metric_or_zero(val_metrics, "loss_mass"),
            metric_or_zero(val_metrics, "loss_value_adaptive"),
            metric_or_zero(val_metrics, "loss_gradient_adaptive"),
            metric_or_zero(val_metrics, "loss_spectral"),
            metric_or_zero(val_metrics, "loss_basin"),
        )

        save_checkpoint(last_checkpoint, epoch, model, optimizer, scheduler, {"train": train_metrics, "val": val_metrics}, config)
        if val_metrics["loss_total"] < best_val:
            best_val = val_metrics["loss_total"]
            best_epoch = epoch
            save_checkpoint(best_checkpoint, epoch, model, optimizer, scheduler, {"train": train_metrics, "val": val_metrics}, config)
            logger.info("Saved new best checkpoint at epoch %d", epoch)
        scheduler.step()

    if bool(training_cfg.get("evaluate_test_after_training", True)):
        checkpoint = torch.load(best_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        test_metrics = epoch_pass(
            model,
            test_loader,
            None,
            device,
            config["loss"],
            max_batches=max_eval_batches,
            log_interval_batches=log_interval_batches,
            logger=logger,
        )
        save_metrics_row(
            metrics_csv,
            {
                "phase": "test_best",
                "epoch": "test_best",
                "best_epoch": best_epoch,
                "learning_rate": 0.0,
                **{f"test_{key}": value for key, value in test_metrics.items()},
            },
            metrics_fieldnames,
        )
        logger.info(
            "Test(best epoch %d) | total=%.5f | mass=%.5f | value=%.5f | gradient=%.5f | spectral=%.5f | basin=%.5f",
            best_epoch,
            metric_or_zero(test_metrics, "loss_total"),
            metric_or_zero(test_metrics, "loss_mass"),
            metric_or_zero(test_metrics, "loss_value_adaptive"),
            metric_or_zero(test_metrics, "loss_gradient_adaptive"),
            metric_or_zero(test_metrics, "loss_spectral"),
            metric_or_zero(test_metrics, "loss_basin"),
        )
    else:
        logger.info("Test evaluation skipped by protocol; validation-only selection remains frozen.")
    plot_training_diagnostics(metrics_csv, output_dir)


if __name__ == "__main__":
    main()
