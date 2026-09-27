"""Build alpha and beta adaptive soft-constraint weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common import configure_logging, guess_data_var, open_dataset_any, write_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--consistency",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data_processed"
            / "adaptive_weights_train_only"
            / "wghm_jplm_consistency_train_only.nc"
        ),
    )
    parser.add_argument("--aridity", type=Path)
    parser.add_argument("--glacier", type=Path)
    parser.add_argument("--human", type=Path)
    parser.add_argument(
        "--land",
        type=Path,
        default=PROJECT_ROOT / "data_raw" / "jpl_mascon" / "GRCTellus.JPL.200204_202603.GLO.RL06.3M.MSCNv04CRI.nc",
        help="Raster-like file that contains a land mask variable; used to zero weights over oceans.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data_processed" / "adaptive_weights_train_only",
    )
    parser.add_argument(
        "--allow-missing-fit-metadata",
        action="store_true",
        help="Permit legacy consistency files without a recorded fit window. Never use for formal experiments.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalize_positive(da: xr.DataArray) -> xr.DataArray:
    """Scale a field to [0, 1] using finite min and max."""
    finite_min = da.min(skipna=True)
    finite_max = da.max(skipna=True)
    denom = finite_max - finite_min
    denom = xr.where(np.abs(denom) < 1e-12, 1.0, denom)
    return ((da - finite_min) / denom).clip(0, 1).fillna(0)


def normalize_absolute(da: xr.DataArray) -> xr.DataArray:
    """Scale absolute values to [0, 1]."""
    return normalize_positive(np.abs(da))


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest for one source file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_record(path: Path | None) -> dict[str, str]:
    """Return a serializable provenance record for an optional source."""
    if path is None:
        return {"path": "", "sha256": "", "status": "not_provided"}
    resolved = path.resolve()
    if not path.exists():
        return {"path": str(resolved), "sha256": "", "status": "missing"}
    return {"path": str(resolved), "sha256": sha256_file(path), "status": "available"}


def validate_fit_metadata(ds: xr.Dataset, allow_missing: bool = False) -> None:
    """Reject consistency inputs that cannot prove a frozen training-only fit."""
    required = {
        "fit_start_requested",
        "fit_end_requested",
        "fit_start_effective",
        "fit_end_effective",
        "fit_n_months",
    }
    missing = sorted(required.difference(ds.attrs))
    if missing and not allow_missing:
        raise ValueError(
            "Consistency input lacks mandatory train-only provenance attributes: "
            f"{missing}. Recompute it with --fit-start/--fit-end."
        )


def optional_penalty(path: Path | None, template: xr.DataArray) -> xr.DataArray:
    """Load an optional raster-like file and normalize it to the template grid."""
    if path is None or not path.exists():
        return xr.zeros_like(template)
    ds = open_dataset_any(path)
    variable = guess_data_var(ds)
    field = ds[variable]
    if "time" in field.dims:
        field = field.mean("time")
    field = field.interp(lat=template["lat"], lon=template["lon"], method="nearest")
    return normalize_positive(field)


def load_land_mask(path: Path | None, template: xr.DataArray) -> xr.DataArray:
    """Load a binary-like land mask on the template grid."""
    if path is None or not path.exists():
        return xr.ones_like(template)
    ds = open_dataset_any(path)
    variable = guess_data_var(ds, aliases=["land_mask", "mask", "land"])
    field = ds[variable]
    if "time" in field.dims:
        field = field.isel(time=0, drop=True)
    field = field.interp(lat=template["lat"], lon=template["lon"], method="nearest")
    return xr.where(field >= 0.5, 1.0, 0.0).fillna(0.0)


def main() -> None:
    """Entry point."""
    args = parse_args()
    logger = configure_logging()
    alpha_path = args.output_dir / "alpha_value_weight.nc"
    beta_path = args.output_dir / "beta_gradient_weight.nc"
    if alpha_path.exists() and beta_path.exists() and not args.overwrite:
        logger.info("Adaptive weight outputs already exist.")
        return

    ds = open_dataset_any(args.consistency)
    validate_fit_metadata(ds, allow_missing=args.allow_missing_fit_metadata)
    corr = ds["corr"].clip(-1, 1).fillna(0)
    corr_score = ((corr + 1.0) / 2.0).clip(0, 1)
    trend_penalty = normalize_absolute(ds["trend_diff"])
    season_penalty = normalize_absolute(ds["season_amp_diff"])
    nrmse_penalty = normalize_positive(ds["nrmse"])
    aridity_penalty = optional_penalty(args.aridity, corr)
    glacier_penalty = optional_penalty(args.glacier, corr)
    human_penalty = optional_penalty(args.human, corr)
    land_mask = load_land_mask(args.land, corr)

    alpha = (
        0.05
        + 0.95
        * corr_score
        * (1.0 - 0.60 * trend_penalty)
        * (1.0 - 0.40 * season_penalty)
        * (1.0 - 0.40 * nrmse_penalty)
        * (1.0 - 0.35 * aridity_penalty)
        * (1.0 - 0.75 * glacier_penalty)
        * (1.0 - 0.50 * human_penalty)
    ).clip(0.05, 1.0) * land_mask
    alpha.name = "alpha_value_weight"
    alpha.attrs["description"] = "Adaptive weight for WGHM value constraint, masked to land cells."
    alpha.attrs["units"] = "1"

    beta = (
        0.10
        + 0.90
        * corr_score
        * (1.0 - 0.45 * trend_penalty)
        * (1.0 - 0.35 * season_penalty)
        * (1.0 - 0.30 * nrmse_penalty)
        * (1.0 - 0.25 * aridity_penalty)
        * (1.0 - 0.45 * glacier_penalty)
        * (1.0 - 0.35 * human_penalty)
    ).clip(0.05, 1.0) * land_mask
    beta.name = "beta_gradient_weight"
    beta.attrs["description"] = "Adaptive weight for WGHM gradient constraint, masked to land cells."
    beta.attrs["units"] = "1"

    sources = {
        "consistency": source_record(args.consistency),
        "aridity": source_record(args.aridity),
        "glacier": source_record(args.glacier),
        "human": source_record(args.human),
        "land": source_record(args.land),
    }
    common_attrs = {
        "description": "Train-window-only adaptive soft-constraint weights.",
        "fit_start_requested": str(ds.attrs.get("fit_start_requested", "UNKNOWN")),
        "fit_end_requested": str(ds.attrs.get("fit_end_requested", "UNKNOWN")),
        "fit_start_effective": str(ds.attrs.get("fit_start_effective", "UNKNOWN")),
        "fit_end_effective": str(ds.attrs.get("fit_end_effective", "UNKNOWN")),
        "fit_n_months": int(ds.attrs.get("fit_n_months", -1)),
        "consistency_source_sha256": sha256_file(args.consistency),
        "generation_script": str(Path(__file__).resolve()),
        "generation_script_sha256": sha256_file(Path(__file__).resolve()),
        "generation_parameters": json.dumps(
            {
                "formula_version": "train_only_static_v1",
                "allow_missing_fit_metadata": args.allow_missing_fit_metadata,
            },
            sort_keys=True,
        ),
        "source_manifest": json.dumps(sources, sort_keys=True),
        "publication_eligible": "false_pending_data_qc_and_formal_rerun",
    }
    alpha_ds = xr.Dataset({"alpha_value_weight": alpha}, attrs=common_attrs)
    beta_ds = xr.Dataset({"beta_gradient_weight": beta}, attrs=common_attrs)
    write_dataset(alpha_ds, alpha_path)
    write_dataset(beta_ds, beta_path)
    logger.info("Wrote %s and %s", alpha_path, beta_path)


if __name__ == "__main__":
    main()
