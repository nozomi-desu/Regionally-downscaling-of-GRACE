"""Apply native JPL mascon observation/null-space decomposition to full fields.

This command must run after tiled predictions have been stitched into complete
monthly maps.  Fine detail is retained only where the inference product is
valid; unsupported land is explicitly observation-only.  Missing GRACE months
are dropped rather than being represented as observed constraints.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common import configure_logging, write_dataset  # noqa: E402
from utils.mass_closure import latitude_area_weights  # noqa: E402
from utils.native_mascon import (  # noqa: E402
    decompose_observation_and_nullspace_with_support_numpy,
    mascon_area_weighted_mean_numpy,
)


DEFAULT_JPL = (
    PROJECT_ROOT
    / "data_raw"
    / "jpl_mascon"
    / "GRCTellus.JPL.200204_202603.GLO.RL06.3M.MSCNv04CRI.nc"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference", type=Path, required=True)
    parser.add_argument("--jpl", type=Path, default=DEFAULT_JPL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--prediction-variable",
        default="auto",
        help="Fine prediction variable, or 'auto' to prefer predicted_twsa_raw.",
    )
    parser.add_argument(
        "--missing-months",
        choices=("drop", "error"),
        default="drop",
        help="Policy for inference months without a released JPL observation.",
    )
    parser.add_argument("--max-times", type=int, help="Structural smoke only.")
    return parser.parse_args()


def normalize_longitude(ds: xr.Dataset) -> xr.Dataset:
    lon = ((np.asarray(ds["lon"].values, dtype=np.float64) + 180.0) % 360.0) - 180.0
    return ds.assign_coords(lon=lon).sortby("lon")


def unit_factor_to_mm(units: str) -> float:
    normalized = units.strip().lower().replace(" ", "")
    if normalized in {"mm", "mmequivalentwaterheight", "mmewh", "kgm-2", "kg/m2"}:
        return 1.0
    if normalized in {"cm", "cmequivalentwaterheight", "cmewh"}:
        return 10.0
    if normalized in {"m", "meter", "metre"}:
        return 1000.0
    raise ValueError(f"Unsupported JPL water-thickness units: {units!r}.")


def select_prediction_variable(ds: xr.Dataset, requested: str) -> str:
    if requested != "auto":
        if requested not in ds:
            raise KeyError(f"Prediction variable {requested!r} is absent from inference file.")
        return requested
    for candidate in ("predicted_twsa_raw", "predicted_twsa"):
        if candidate in ds:
            return candidate
    raise KeyError("Inference file has neither predicted_twsa_raw nor predicted_twsa.")


def month_key(values: np.ndarray) -> np.ndarray:
    return np.asarray(values).astype("datetime64[M]")


def validate_grid(inference: xr.Dataset, jpl: xr.Dataset) -> None:
    for coordinate in ("lat", "lon"):
        left = np.asarray(inference[coordinate].values, dtype=np.float64)
        right = np.asarray(jpl[coordinate].values, dtype=np.float64)
        if left.shape != right.shape or not np.allclose(left, right, rtol=0.0, atol=1e-8):
            raise ValueError(
                f"Inference and JPL {coordinate} grids differ; native support cannot be applied."
            )
    if tuple(jpl["mascon_ID"].dims) != ("lat", "lon"):
        raise ValueError("JPL mascon_ID must be a 2D lat/lon raster.")


def build_month_pairs(
    inference_time: np.ndarray,
    jpl_time: np.ndarray,
    missing_policy: str,
) -> tuple[list[tuple[int, int]], list[str]]:
    jpl_lookup = {value: index for index, value in enumerate(month_key(jpl_time))}
    pairs: list[tuple[int, int]] = []
    missing: list[str] = []
    for inference_index, key in enumerate(month_key(inference_time)):
        if key in jpl_lookup:
            pairs.append((inference_index, jpl_lookup[key]))
        else:
            missing.append(str(key))
    if missing and missing_policy == "error":
        raise ValueError(f"No released JPL observation for months: {', '.join(missing)}")
    if not pairs:
        raise ValueError("No monthly overlap remains between inference and released JPL observations.")
    return pairs, missing


def main() -> None:
    args = parse_args()
    logger = configure_logging()
    inference_path = args.inference.resolve()
    jpl_path = args.jpl.resolve()
    output_path = args.output.resolve()
    inference = normalize_longitude(xr.open_dataset(inference_path))
    jpl = normalize_longitude(xr.open_dataset(jpl_path))
    validate_grid(inference, jpl)
    prediction_variable = select_prediction_variable(inference, args.prediction_variable)
    if tuple(inference[prediction_variable].dims) != ("time", "lat", "lon"):
        raise ValueError(f"{prediction_variable} must have dimensions time, lat, lon.")

    pairs, missing_months = build_month_pairs(
        inference["time"].values,
        jpl["time"].values,
        args.missing_months,
    )
    if args.max_times is not None:
        pairs = pairs[: args.max_times]
    if not pairs:
        raise ValueError("--max-times removed all overlapping observations.")

    ids = np.asarray(jpl["mascon_ID"].values, dtype=np.float64)
    output_mask = (
        np.asarray(jpl["land_mask"].values, dtype=np.float64) > 0
    ) & np.isfinite(ids) & (ids > 0)
    labels = np.unique(np.rint(ids[output_mask]).astype(np.int64))
    area_weights = latitude_area_weights(
        np.asarray(inference["lat"].values, dtype=np.float64),
        width=int(inference.sizes["lon"]),
    )
    jpl_factor = unit_factor_to_mm(str(jpl["lwe_thickness"].attrs.get("units", "")))
    uncertainty_factor = unit_factor_to_mm(str(jpl["uncertainty"].attrs.get("units", "")))

    observation_components: list[np.ndarray] = []
    nullspace_components: list[np.ndarray] = []
    reconstructions: list[np.ndarray] = []
    detail_masks: list[np.ndarray] = []
    target_mascons: list[np.ndarray] = []
    uncertainty_mascons: list[np.ndarray] = []
    detail_coverage: list[np.ndarray] = []
    closure_max: list[float] = []
    detail_supported_counts: list[int] = []
    observation_only_counts: list[int] = []
    output_times: list[np.datetime64] = []

    for position, (inference_index, jpl_index) in enumerate(pairs, start=1):
        logger.info("Native decomposition month %d / %d", position, len(pairs))
        prior = np.asarray(
            inference[prediction_variable].isel(time=inference_index).values,
            dtype=np.float64,
        )
        if "valid_mask" in inference:
            detail_mask = (
                np.asarray(inference["valid_mask"].isel(time=inference_index).values) > 0
            )
        else:
            detail_mask = np.isfinite(prior)
        detail_mask &= output_mask & np.isfinite(prior)

        observed_grid = (
            np.asarray(jpl["lwe_thickness"].isel(time=jpl_index).values, dtype=np.float64)
            * jpl_factor
        )
        target, target_valid, returned_labels = mascon_area_weighted_mean_numpy(
            observed_grid,
            ids,
            area_weights,
            output_mask,
            labels,
        )
        if not np.array_equal(labels, returned_labels) or not np.all(target_valid):
            raise ValueError("A released JPL mascon has no finite land observation.")
        uncertainty_grid = (
            np.asarray(jpl["uncertainty"].isel(time=jpl_index).values, dtype=np.float64)
            * uncertainty_factor
        )
        uncertainty, uncertainty_valid, _ = mascon_area_weighted_mean_numpy(
            uncertainty_grid,
            ids,
            area_weights,
            output_mask,
            labels,
        )
        uncertainty = np.where(uncertainty_valid, np.abs(uncertainty), np.nan)

        observation, null_component, reconstruction, diagnostics = (
            decompose_observation_and_nullspace_with_support_numpy(
                prior,
                target,
                ids,
                area_weights,
                detail_mask,
                output_mask,
                labels,
            )
        )
        observation_components.append(observation.astype(np.float32))
        nullspace_components.append(null_component.astype(np.float32))
        reconstructions.append(reconstruction.astype(np.float64))
        detail_masks.append(detail_mask.astype(np.uint8))
        target_mascons.append(target.astype(np.float64))
        uncertainty_mascons.append(uncertainty.astype(np.float32))
        detail_coverage.append(np.asarray(diagnostics["detail_area_fraction"], dtype=np.float32))
        closure_max.append(float(diagnostics["mass_closure_max_abs"]))
        detail_supported_counts.append(int(diagnostics["detail_supported_mascon_count"]))
        observation_only_counts.append(int(diagnostics["observation_only_mascon_count"]))
        output_times.append(inference["time"].values[inference_index])

    output = xr.Dataset(
        data_vars={
            "observation_component": (
                ("time", "lat", "lon"),
                np.stack(observation_components),
            ),
            "nullspace_component": (
                ("time", "lat", "lon"),
                np.stack(nullspace_components),
            ),
            "predicted_twsa_native_projected": (
                ("time", "lat", "lon"),
                np.stack(reconstructions),
            ),
            "detail_support_mask": (
                ("time", "lat", "lon"),
                np.stack(detail_masks),
            ),
            "native_output_mask": (("lat", "lon"), output_mask.astype(np.uint8)),
            "mascon_id_raster": (("lat", "lon"), np.rint(ids).astype(np.int32)),
            "target_native_mascon_twsa": (
                ("time", "mascon"),
                np.stack(target_mascons),
            ),
            "jpl_native_mascon_uncertainty": (
                ("time", "mascon"),
                np.stack(uncertainty_mascons),
            ),
            "detail_area_fraction": (
                ("time", "mascon"),
                np.stack(detail_coverage),
            ),
            "native_mass_closure_max_abs": (
                ("time",),
                np.asarray(closure_max, dtype=np.float64),
            ),
            "detail_supported_mascon_count": (
                ("time",),
                np.asarray(detail_supported_counts, dtype=np.int32),
            ),
            "observation_only_mascon_count": (
                ("time",),
                np.asarray(observation_only_counts, dtype=np.int32),
            ),
        },
        coords={
            "time": np.asarray(output_times),
            "lat": inference["lat"].values,
            "lon": inference["lon"].values,
            "mascon": labels,
        },
        attrs={
            "protocol_id": "native_mascon_observation_nullspace_v1",
            "source_inference": str(inference_path),
            "source_jpl": str(jpl_path),
            "source_prediction_variable": prediction_variable,
            "operator": "JPL mascon_ID area-weighted mean",
            "decomposition": "released observation component + prior-derived zero-mascon-mean detail",
            "unsupported_land_policy": "observation_only_nullspace_zero",
            "missing_grace_month_policy": args.missing_months,
            "strict_closure_storage_dtype": "float64",
            "claim_boundary": (
                "The constraint guarantees native-mascon closure; it does not identify "
                "or validate the fine-scale null-space component."
            ),
        },
    )
    output["observation_component"].attrs.update(units="mm EWH", role="range_of_native_observation")
    output["nullspace_component"].attrs.update(units="mm EWH", role="prior_detail_in_nullspace")
    output["predicted_twsa_native_projected"].attrs.update(units="mm EWH")
    output["target_native_mascon_twsa"].attrs.update(units="mm EWH")
    output["jpl_native_mascon_uncertainty"].attrs.update(units="mm EWH", uncertainty="1-sigma")
    write_dataset(output, output_path)

    summary = {
        "protocol_id": output.attrs["protocol_id"],
        "source_inference": str(inference_path),
        "source_jpl": str(jpl_path),
        "prediction_variable": prediction_variable,
        "overlap_month_count": len(pairs),
        "dropped_missing_grace_months": missing_months,
        "native_land_mascon_count": int(labels.size),
        "native_land_cell_count": int(output_mask.sum()),
        "closure_max_abs_mm": float(np.max(closure_max)),
        "detail_supported_mascon_count_min": int(np.min(detail_supported_counts)),
        "observation_only_mascon_count_max": int(np.max(observation_only_counts)),
        "output": str(output_path),
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote %s", output_path)
    logger.info("Wrote %s", summary_path)


if __name__ == "__main__":
    main()
