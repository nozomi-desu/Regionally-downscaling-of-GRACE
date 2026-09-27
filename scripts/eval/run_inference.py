"""Run tiled inference and reconstruct full monthly prediction maps."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common import configure_logging, ensure_dir, write_dataset  # noqa: E402
from scripts.eval.common import build_tile_starts, repeat_coarse_to_fine  # noqa: E402
from scripts.train.dataset import (  # noqa: E402
    SPLIT_NAME_TO_INDEX,
    GraceTileDataset,
    apply_regional_weight_adjustments as adjust_dataset_regional_weights,
    load_external_weight_fields,
)
from scripts.train.model import build_model_from_config  # noqa: E402
from utils.mass_closure import (  # noqa: E402
    apply_mass_closure_correction,
    latitude_area_weights,
    masked_pool2d_mean_numpy,
    project_to_coarse_constraint_numpy,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--zarr-path", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--split", choices=sorted(SPLIT_NAME_TO_INDEX), default="test")
    parser.add_argument("--time-start", type=str, help="Inclusive YYYY-MM or YYYY-MM-DD filter on dataset time.")
    parser.add_argument("--time-end", type=str, help="Inclusive YYYY-MM or YYYY-MM-DD filter on dataset time.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--tile-size", type=int)
    parser.add_argument("--stride", type=int)
    parser.add_argument("--max-times", type=int)
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    """Resolve device string to torch device."""
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def infer_output_path(args: argparse.Namespace) -> Path:
    """Build a default output path when none is provided."""
    if args.output is not None:
        return args.output
    stem = args.checkpoint.parent.parent.name
    return PROJECT_ROOT / "outputs" / "inference" / f"{stem}_{args.split}.nc"


def load_config_from_checkpoint(args: argparse.Namespace) -> dict:
    """Load experiment configuration from checkpoint or explicit path."""
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if args.config is not None:
        import yaml

        with args.config.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    if "config" not in checkpoint:
        raise KeyError("Checkpoint does not contain embedded config. Pass --config.")
    return checkpoint["config"]


def apply_channel_overrides(channel_arrays: dict[str, np.ndarray], overrides: dict[str, float]) -> dict[str, np.ndarray]:
    """Apply constant overrides used by ablation configs."""
    updated = dict(channel_arrays)
    for channel_name, value in (overrides or {}).items():
        if channel_name in updated:
            updated[channel_name] = np.full_like(updated[channel_name], np.float32(value), dtype=np.float32)
    return updated


def apply_inference_regional_weight_adjustments(
    channel_arrays: dict[str, np.ndarray],
    regional_weights: dict[str, float | bool] | None,
) -> dict[str, np.ndarray]:
    """Mirror the training-time alpha/beta adjustment logic during inference."""
    updated = dict(channel_arrays)
    if not regional_weights:
        return updated

    alpha_adjusted, beta_adjusted = adjust_dataset_regional_weights(
        updated["alpha_value_weight"],
        updated["beta_gradient_weight"],
        updated["glacier_fraction"],
        updated["aridity_mask"],
        updated["human_activity_index"],
        regional_weights,
    )
    updated["alpha_value_weight"] = alpha_adjusted
    updated["beta_gradient_weight"] = beta_adjusted
    return updated


def build_model(checkpoint: dict, config: dict, device: torch.device) -> torch.nn.Module:
    """Instantiate the model and load checkpoint weights."""
    in_channels = int(checkpoint["model_state_dict"]["stem.block.0.weight"].shape[1])
    model_cfg = config["model"]
    model = build_model_from_config(in_channels, model_cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def reconstruct_time_step(
    ds: xr.Dataset,
    model: torch.nn.Module,
    time_idx: int,
    tile_size: int,
    stride: int,
    coarse_factor: int,
    batch_size: int,
    device: torch.device,
    channel_overrides: dict[str, float],
    external_weight_fields: dict[str, np.ndarray] | None,
    regional_weights: dict[str, float | bool] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Reconstruct one global raw prediction map and optional learned gate."""
    lat_size = int(ds.sizes["lat"])
    lon_size = int(ds.sizes["lon"])
    lat_starts = build_tile_starts(lat_size, tile_size, stride)
    lon_starts = build_tile_starts(lon_size, tile_size, stride)

    static_fields = {
        "alpha_value_weight": np.asarray(ds["alpha_value_weight"].values, dtype=np.float32),
        "beta_gradient_weight": np.asarray(ds["beta_gradient_weight"].values, dtype=np.float32),
        "aridity_mask": np.asarray(ds["aridity_mask"].values, dtype=np.float32),
        "human_activity_index": np.asarray(ds["human_activity_index"].values, dtype=np.float32),
        "glacier_fraction": np.asarray(ds["glacier_fraction"].values, dtype=np.float32),
        "hydrobasins_mask": np.asarray(ds["hydrobasins_mask"].values, dtype=np.float32),
        "land_mask": np.asarray(ds["land_mask"].values, dtype=np.float32),
    }
    static_fields.update(external_weight_fields or {})
    dynamic_fields = {
        "input_wghm_twsa": np.asarray(ds["input_wghm_twsa"].isel(time=time_idx).values, dtype=np.float32),
        "input_era5_twsa": np.asarray(ds["input_era5_twsa"].isel(time=time_idx).values, dtype=np.float32),
        "input_era5_cwsc": np.asarray(ds["input_era5_cwsc"].isel(time=time_idx).values, dtype=np.float32),
    }
    coarse_context_full = repeat_coarse_to_fine(
        np.asarray(ds["target_jplm_twsa_coarse"].isel(time=time_idx).values, dtype=np.float32),
        factor=coarse_factor,
    )
    month = int(np.asarray(ds["month_of_year"].isel(time=time_idx).values))
    month_angle = 2.0 * np.pi * (month - 1) / 12.0
    dynamic_fields["coarse_jpl_context"] = coarse_context_full
    dynamic_fields["month_sin"] = np.full((lat_size, lon_size), np.sin(month_angle), dtype=np.float32)
    dynamic_fields["month_cos"] = np.full((lat_size, lon_size), np.cos(month_angle), dtype=np.float32)

    all_fields = apply_channel_overrides({**dynamic_fields, **static_fields}, channel_overrides)
    all_fields = apply_inference_regional_weight_adjustments(all_fields, regional_weights)
    channel_order = GraceTileDataset.dynamic_channels + GraceTileDataset.static_channels + ["month_sin", "month_cos"]

    pred_sum = np.zeros((lat_size, lon_size), dtype=np.float32)
    pred_count = np.zeros((lat_size, lon_size), dtype=np.float32)
    gate_mode = getattr(model, "gate_mode", None)
    export_gate = gate_mode in {"learned", "oracle"}
    gate_sum = np.zeros((lat_size, lon_size), dtype=np.float32) if export_gate else None
    gate_count = np.zeros((lat_size, lon_size), dtype=np.float32) if export_gate else None
    valid_mask_full = np.asarray(ds["valid_mask"].isel(time=time_idx).values, dtype=np.float32)
    if "cell_area_weight" in ds.data_vars:
        area_weight_full = np.asarray(ds["cell_area_weight"].values, dtype=np.float32)
    else:
        area_weight_full = latitude_area_weights(
            np.asarray(ds["lat"].values),
            width=lon_size,
        ).astype(np.float32)
    tile_specs = [(lat_start, lon_start) for lat_start in lat_starts for lon_start in lon_starts]

    with torch.inference_mode():
        for batch_start in range(0, len(tile_specs), batch_size):
            batch_specs = tile_specs[batch_start : batch_start + batch_size]
            input_tiles = []
            coarse_tiles = []
            valid_tiles = []
            area_tiles = []
            oracle_tiles = []
            for lat_start, lon_start in batch_specs:
                lat_slice = slice(lat_start, lat_start + tile_size)
                lon_slice = slice(lon_start, lon_start + tile_size)
                tile = np.stack(
                    [all_fields[name][lat_slice, lon_slice] for name in channel_order],
                    axis=0,
                )
                input_tiles.append(tile)
                coarse_tiles.append(all_fields["coarse_jpl_context"][lat_slice, lon_slice][None, ...])
                valid_tiles.append(valid_mask_full[lat_slice, lon_slice][None, ...])
                area_tiles.append(area_weight_full[lat_slice, lon_slice][None, ...])
                if gate_mode == "oracle":
                    oracle_tiles.append(
                        np.asarray(
                            ds["closed_loop_reliability_truth"]
                            .isel(time=time_idx, lat=lat_slice, lon=lon_slice)
                            .values,
                            dtype=np.float32,
                        )[None, ...]
                    )
            input_tensor = torch.from_numpy(np.stack(input_tiles, axis=0)).to(device)
            coarse_tensor = torch.from_numpy(np.stack(coarse_tiles, axis=0)).to(device)
            valid_tensor = torch.from_numpy(np.stack(valid_tiles, axis=0)).to(device)
            area_tensor = torch.from_numpy(np.stack(area_tiles, axis=0)).to(device)
            forward_kwargs = {
                "coarse_context": coarse_tensor,
                "valid_mask": valid_tensor,
                "area_weight": area_tensor,
            }
            if gate_mode == "oracle":
                forward_kwargs["oracle_gate"] = torch.from_numpy(np.stack(oracle_tiles, axis=0)).to(device)
            pred_tiles = model(input_tensor, **forward_kwargs).detach().cpu().numpy()[:, 0, :, :]
            gate_tiles = None
            if export_gate:
                last_gate = getattr(model, "last_gate", None)
                if last_gate is None:
                    raise RuntimeError("Gated model did not expose last_gate during inference.")
                gate_tiles = last_gate.detach().cpu().numpy()[:, 0, :, :]

            for tile_index, ((lat_start, lon_start), pred_tile) in enumerate(
                zip(batch_specs, pred_tiles, strict=True)
            ):
                lat_slice = slice(lat_start, lat_start + tile_size)
                lon_slice = slice(lon_start, lon_start + tile_size)
                pred_sum[lat_slice, lon_slice] += pred_tile.astype(np.float32)
                pred_count[lat_slice, lon_slice] += 1.0
                if gate_tiles is not None and gate_sum is not None and gate_count is not None:
                    gate_sum[lat_slice, lon_slice] += gate_tiles[tile_index].astype(np.float32)
                    gate_count[lat_slice, lon_slice] += 1.0

    pred_count = np.clip(pred_count, a_min=1.0, a_max=None)
    prediction = pred_sum / pred_count
    valid_mask = valid_mask_full.astype(bool)
    prediction = np.where(valid_mask, prediction, np.nan).astype(np.float32)
    reliability_gate = None
    if gate_sum is not None and gate_count is not None:
        reliability_gate = gate_sum / np.clip(gate_count, a_min=1.0, a_max=None)
        reliability_gate = np.where(valid_mask, reliability_gate, np.nan).astype(np.float32)
    return prediction, valid_mask.astype(np.uint8), reliability_gate


def main() -> None:
    """Entry point."""
    args = parse_args()
    logger = configure_logging()
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = load_config_from_checkpoint(args)
    output_path = infer_output_path(args)

    dataset_cfg = config["dataset"]
    zarr_path = Path(args.zarr_path or dataset_cfg["zarr_path"])
    if not zarr_path.is_absolute():
        zarr_path = (PROJECT_ROOT / zarr_path).resolve()
    ds = xr.open_zarr(zarr_path)

    split_idx = SPLIT_NAME_TO_INDEX[args.split]
    all_time_indices = np.where(np.asarray(ds["split_index"].values) == split_idx)[0].tolist()
    if args.time_start or args.time_end:
        time_values_all = ds["time"].values
        if args.time_start:
            start = np.datetime64(args.time_start)
            all_time_indices = [idx for idx in all_time_indices if time_values_all[idx] >= start]
        if args.time_end:
            end = np.datetime64(args.time_end)
            if len(str(args.time_end)) == 7:
                end = end + np.timedelta64(31, "D")
            else:
                end = end + np.timedelta64(1, "D")
            all_time_indices = [idx for idx in all_time_indices if time_values_all[idx] < end]
    if args.max_times is not None:
        all_time_indices = all_time_indices[: args.max_times]
    if not all_time_indices:
        raise ValueError(f"No time indices found for split={args.split}.")

    tile_size = int(args.tile_size or dataset_cfg["tile_size"])
    stride = int(args.stride or dataset_cfg.get("stride_eval", tile_size))
    coarse_factor = int(dataset_cfg["coarse_factor"])
    channel_overrides = dict(dataset_cfg.get("channel_overrides") or {})
    weight_files = dict(dataset_cfg.get("weight_files") or {})
    for specification in weight_files.values():
        if isinstance(specification, dict):
            weight_path = Path(specification["path"])
            if not weight_path.is_absolute():
                specification["path"] = str((PROJECT_ROOT / weight_path).resolve())
    external_weight_fields = load_external_weight_fields(
        weight_files,
        np.asarray(ds["lat"].values),
        np.asarray(ds["lon"].values),
    )
    regional_weights = dict(config.get("regional_weights") or {})

    model = build_model(checkpoint, config, device)
    postprocess_cfg = config.get("postprocess", {})
    apply_correction = bool(postprocess_cfg.get("apply_mass_closure_correction", False))
    strict_projection_cfg = postprocess_cfg.get("strict_mass_projection", {})
    apply_strict_projection = bool(strict_projection_cfg.get("enabled", False))
    if apply_correction and apply_strict_projection:
        raise ValueError("Legacy mass-closure correction and strict projection cannot both be enabled.")
    save_raw_prediction = bool(postprocess_cfg.get("save_raw_prediction", True))
    save_corrected_prediction = bool(postprocess_cfg.get("save_corrected_prediction", apply_correction))
    correction_blend_factor = float(postprocess_cfg.get("mass_closure_blend_factor", 1.0))
    correction_max_residual_abs = postprocess_cfg.get("mass_closure_max_residual_abs")
    logger.info(
        "Running inference for split=%s on %d monthly fields using %s",
        args.split,
        len(all_time_indices),
        device,
    )

    predictions_default = []
    predictions_raw = []
    predictions_corrected = []
    predictions_projected = []
    coarse_predictions_default = []
    coarse_predictions_raw = []
    coarse_predictions_corrected = []
    coarse_predictions_projected = []
    targets = []
    wghm = []
    valid_masks = []
    mass_before = []
    mass_after = []
    mass_max_after = []
    reliability_gates = []
    if "cell_area_weight" in ds.data_vars:
        area_weights = np.asarray(ds["cell_area_weight"].values, dtype=np.float64)
    else:
        area_weights = latitude_area_weights(np.asarray(ds["lat"].values), width=int(ds.sizes["lon"]))
    time_values = ds["time"].isel(time=all_time_indices).values

    for index, time_idx in enumerate(all_time_indices, start=1):
        logger.info("Reconstructing month %d / %d (%s)", index, len(all_time_indices), str(ds["time"].values[time_idx])[:10])
        pred_raw, valid_mask, reliability_gate = reconstruct_time_step(
            ds=ds,
            model=model,
            time_idx=time_idx,
            tile_size=tile_size,
            stride=stride,
            coarse_factor=coarse_factor,
            batch_size=args.batch_size,
            device=device,
            channel_overrides=channel_overrides,
            external_weight_fields=external_weight_fields,
            regional_weights=regional_weights,
        )
        target = np.asarray(ds["target_jplm_twsa"].isel(time=time_idx).values, dtype=np.float32)
        wghm_field = np.asarray(ds["input_wghm_twsa"].isel(time=time_idx).values, dtype=np.float32)
        valid_mask_bool = valid_mask.astype(bool)

        pred_corrected = pred_raw
        pred_projected = pred_raw
        correction_stats = {
            "mass_closure_error_before": float("nan"),
            "mass_closure_error_after": float("nan"),
            "mass_closure_max_abs_after": float("nan"),
        }
        target_coarse_month = np.asarray(
            ds["target_jplm_twsa_coarse"].isel(time=time_idx).values,
            dtype=np.float64,
        )
        if apply_strict_projection:
            pred_projected, correction_stats = project_to_coarse_constraint_numpy(
                np.where(np.isfinite(pred_raw), pred_raw, 0.0),
                target_coarse_month,
                valid_mask_bool.astype(np.float64),
                factor=coarse_factor,
                area_weights=area_weights,
                refinement_steps=int(strict_projection_cfg.get("refinement_steps", 2)),
            )
            pred_projected = np.where(valid_mask_bool, pred_projected, np.nan).astype(np.float64)
            pred_corrected = pred_projected
        if apply_correction:
            pred_corrected, correction_stats = apply_mass_closure_correction(
                pred_raw,
                target,
                valid_mask_bool.astype(np.float32),
                factor=coarse_factor,
                blend_factor=correction_blend_factor,
                max_residual_abs=float(correction_max_residual_abs) if correction_max_residual_abs is not None else None,
            )
            pred_corrected = np.where(valid_mask_bool, pred_corrected, np.nan).astype(np.float32)

        aggregation_weights = valid_mask_bool.astype(np.float64) * area_weights

        coarse_pred_raw, coarse_mask_raw = masked_pool2d_mean_numpy(
            np.where(np.isfinite(pred_raw), pred_raw, 0.0),
            aggregation_weights,
            factor=coarse_factor,
        )
        coarse_pred_corrected, coarse_mask_corrected = masked_pool2d_mean_numpy(
            np.where(np.isfinite(pred_corrected), pred_corrected, 0.0),
            aggregation_weights,
            factor=coarse_factor,
        )
        coarse_pred_projected, coarse_mask_projected = masked_pool2d_mean_numpy(
            np.where(np.isfinite(pred_projected), pred_projected, 0.0),
            aggregation_weights,
            factor=coarse_factor,
        )
        coarse_pred_raw = np.where(coarse_mask_raw, coarse_pred_raw, np.nan).astype(np.float32)
        coarse_pred_corrected = np.where(coarse_mask_corrected, coarse_pred_corrected, np.nan).astype(np.float32)
        coarse_pred_projected = np.where(coarse_mask_projected, coarse_pred_projected, np.nan).astype(np.float64)
        if not apply_correction and not apply_strict_projection:
            raw_abs = np.abs(coarse_pred_raw - target_coarse_month)
            raw_mass_error = float(np.nanmean(raw_abs))
            correction_stats["mass_closure_error_before"] = raw_mass_error
            correction_stats["mass_closure_error_after"] = raw_mass_error
            correction_stats["mass_closure_max_abs_after"] = float(np.nanmax(raw_abs))

        use_projected = apply_strict_projection
        default_prediction = pred_projected if use_projected else (pred_corrected if apply_correction else pred_raw)
        default_coarse_prediction = coarse_pred_projected if use_projected else (coarse_pred_corrected if apply_correction else coarse_pred_raw)

        predictions_default.append(
            default_prediction.astype(np.float64 if apply_strict_projection else np.float32)
        )
        coarse_predictions_default.append(
            default_coarse_prediction.astype(np.float64 if apply_strict_projection else np.float32)
        )
        predictions_raw.append(pred_raw.astype(np.float32))
        coarse_predictions_raw.append(coarse_pred_raw.astype(np.float32))
        predictions_corrected.append(pred_corrected.astype(np.float32))
        coarse_predictions_corrected.append(coarse_pred_corrected.astype(np.float32))
        predictions_projected.append(pred_projected.astype(np.float64))
        coarse_predictions_projected.append(coarse_pred_projected.astype(np.float64))
        targets.append(target)
        wghm.append(wghm_field)
        valid_masks.append(valid_mask.astype(np.uint8))
        mass_before.append(float(correction_stats["mass_closure_error_before"]))
        mass_after.append(float(correction_stats["mass_closure_error_after"]))
        mass_max_after.append(float(correction_stats["mass_closure_max_abs_after"]))
        if reliability_gate is not None:
            reliability_gates.append(reliability_gate)

    data_vars = {
        "predicted_twsa": (("time", "lat", "lon"), np.stack(predictions_default, axis=0)),
        "predicted_twsa_coarse": (("time", "coarse_lat", "coarse_lon"), np.stack(coarse_predictions_default, axis=0)),
        "target_jplm_twsa": (("time", "lat", "lon"), np.stack(targets, axis=0)),
        "target_jplm_twsa_coarse": (
            ("time", "coarse_lat", "coarse_lon"),
            np.asarray(ds["target_jplm_twsa_coarse"].isel(time=all_time_indices).values, dtype=np.float32),
        ),
        "input_wghm_twsa": (("time", "lat", "lon"), np.stack(wghm, axis=0)),
        "valid_mask": (("time", "lat", "lon"), np.stack(valid_masks, axis=0)),
        "mass_closure_error_before": (("time",), np.asarray(mass_before, dtype=np.float32)),
        "mass_closure_error_after": (("time",), np.asarray(mass_after, dtype=np.float32)),
        "mass_closure_max_abs_after": (("time",), np.asarray(mass_max_after, dtype=np.float32)),
    }
    if save_raw_prediction:
        data_vars["predicted_twsa_raw"] = (("time", "lat", "lon"), np.stack(predictions_raw, axis=0))
        data_vars["predicted_twsa_coarse_raw"] = (
            ("time", "coarse_lat", "coarse_lon"),
            np.stack(coarse_predictions_raw, axis=0),
        )
    if save_corrected_prediction or apply_correction:
        data_vars["predicted_twsa_corrected"] = (("time", "lat", "lon"), np.stack(predictions_corrected, axis=0))
        data_vars["predicted_twsa_coarse_corrected"] = (
            ("time", "coarse_lat", "coarse_lon"),
            np.stack(coarse_predictions_corrected, axis=0),
        )
    if apply_strict_projection:
        data_vars["predicted_twsa_projected"] = (
            ("time", "lat", "lon"),
            np.stack(predictions_projected, axis=0),
        )
        data_vars["predicted_twsa_coarse_projected"] = (
            ("time", "coarse_lat", "coarse_lon"),
            np.stack(coarse_predictions_projected, axis=0),
        )
    if reliability_gates:
        if len(reliability_gates) != len(all_time_indices):
            raise RuntimeError("Reliability gate was only available for part of the inference period.")
        gate_stack = np.stack(reliability_gates, axis=0).astype(np.float32)
        data_vars["reliability_gate"] = (("time", "lat", "lon"), gate_stack)
        data_vars["reliability_gate_mean"] = (
            ("time",),
            np.nanmean(gate_stack, axis=(1, 2)).astype(np.float32),
        )
        data_vars["reliability_gate_std"] = (
            ("time",),
            np.nanstd(gate_stack, axis=(1, 2)).astype(np.float32),
        )
        data_vars["reliability_gate_min"] = (
            ("time",),
            np.nanmin(gate_stack, axis=(1, 2)).astype(np.float32),
        )
        data_vars["reliability_gate_max"] = (
            ("time",),
            np.nanmax(gate_stack, axis=(1, 2)).astype(np.float32),
        )

    out_ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "time": time_values,
            "lat": ds["lat"].values,
            "lon": ds["lon"].values,
            "coarse_lat": ds["coarse_lat"].values,
            "coarse_lon": ds["coarse_lon"].values,
        },
        attrs={
            "checkpoint": str(args.checkpoint),
            "split": args.split,
            "tile_size": tile_size,
            "stride": stride,
            "coarse_factor": coarse_factor,
            "default_prediction_variant": "projected" if apply_strict_projection else ("corrected" if apply_correction else "raw"),
            "mass_closure_correction_applied": str(apply_correction).lower(),
            "mass_closure_blend_factor": correction_blend_factor,
            "strict_mass_projection_applied": str(apply_strict_projection).lower(),
            "strict_mass_projection_operator": "masked_coslat_area_weighted_block_mean",
            "strict_mass_projection_refinement_steps": int(strict_projection_cfg.get("refinement_steps", 2)),
            "regional_weight_adjustment_applied": str(bool(regional_weights)).lower(),
            "reliability_gate_exported": str(bool(reliability_gates)).lower(),
            "reliability_gate_interpretation": (
                "sigmoid gate multiplying the area-centered high-frequency residual; "
                "not an independently calibrated probability"
            ),
        },
    )
    ensure_dir(output_path.parent)
    write_dataset(out_ds, output_path)
    logger.info("Wrote inference output to %s", output_path)


if __name__ == "__main__":
    main()
