"""PyTorch dataset for GRACE adaptive downscaling training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset


SPLIT_NAME_TO_INDEX = {"train": 0, "val": 1, "test": 2}


def load_external_weight_fields(
    weight_files: dict[str, dict[str, str] | str] | None,
    lat: np.ndarray,
    lon: np.ndarray,
) -> dict[str, np.ndarray]:
    """Load frozen alpha/beta maps without copying the full training Zarr."""
    fields: dict[str, np.ndarray] = {}
    for channel_name, specification in (weight_files or {}).items():
        if channel_name not in {"alpha_value_weight", "beta_gradient_weight"}:
            raise ValueError(f"External weight channel is not allowed: {channel_name}")
        if isinstance(specification, str):
            path = Path(specification)
            variable = "alpha" if channel_name == "alpha_value_weight" else "beta"
        else:
            path = Path(specification["path"])
            variable = str(specification.get("variable") or ("alpha" if channel_name == "alpha_value_weight" else "beta"))
        with xr.open_dataset(path) as weight_ds:
            if variable not in weight_ds:
                raise KeyError(f"{path} does not contain requested variable {variable!r}.")
            field = weight_ds[variable]
            if not np.array_equal(field["lat"].values, lat) or not np.array_equal(field["lon"].values, lon):
                raise ValueError(f"External weight grid does not match training data: {path}")
            values = np.asarray(field.values, dtype=np.float32)
        finite = values[np.isfinite(values)]
        if finite.size == 0 or finite.min() < -2e-6 or finite.max() > 1.0 + 2e-6:
            raise ValueError(f"External {channel_name} is empty or outside [0,1]: {path}")
        fields[channel_name] = np.where(np.isfinite(values), np.clip(values, 0.0, 1.0), 0.0).astype(np.float32)
    return fields


@dataclass(frozen=True)
class TileIndex:
    """One spatiotemporal sample index."""

    time_idx: int
    lat_start: int
    lon_start: int


def _positive_membership(field: np.ndarray, threshold: float) -> np.ndarray:
    """Map values above one threshold to a smooth [0, 1] regional membership."""
    clipped = np.clip(np.asarray(field, dtype=np.float32), 0.0, 1.0)
    threshold = float(np.clip(threshold, 0.0, 0.999))
    return np.clip((clipped - threshold) / max(1.0 - threshold, 1e-6), 0.0, 1.0).astype(np.float32)


def _negative_membership(field: np.ndarray, threshold: float) -> np.ndarray:
    """Map values below one threshold to a smooth [0, 1] regional membership."""
    clipped = np.clip(np.asarray(field, dtype=np.float32), 0.0, 1.0)
    threshold = float(np.clip(threshold, 1e-6, 1.0))
    return np.clip((threshold - clipped) / threshold, 0.0, 1.0).astype(np.float32)


def apply_regional_weight_adjustments(
    alpha: np.ndarray,
    beta: np.ndarray,
    glacier_fraction: np.ndarray,
    aridity_mask: np.ndarray | None,
    human_activity_index: np.ndarray | None,
    regional_weight_cfg: dict[str, float | bool] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Adjust adaptive alpha/beta maps for glacier, arid, humid, and human regions."""
    cfg = regional_weight_cfg or {}
    alpha_base = np.asarray(alpha, dtype=np.float32)
    beta_base = np.asarray(beta, dtype=np.float32)
    if not cfg:
        return alpha_base, beta_base

    glacier = np.clip(np.asarray(glacier_fraction, dtype=np.float32), 0.0, 1.0)
    aridity = np.clip(np.asarray(aridity_mask if aridity_mask is not None else np.zeros_like(alpha_base), dtype=np.float32), 0.0, 1.0)
    human = np.clip(
        np.asarray(human_activity_index if human_activity_index is not None else np.zeros_like(alpha_base), dtype=np.float32),
        0.0,
        1.0,
    )

    arid_membership = _positive_membership(aridity, float(cfg.get("arid_threshold", 0.5)))
    humid_membership = _negative_membership(aridity, float(cfg.get("humid_threshold", 0.1)))
    human_membership = _positive_membership(human, float(cfg.get("high_human_threshold", 0.05)))

    alpha_adjusted = alpha_base.copy()
    beta_adjusted = beta_base.copy()

    reductions_alpha = [
        float(cfg.get("k_alpha_glacier", 0.0)) * glacier,
        float(cfg.get("k_alpha_arid", 0.0)) * arid_membership,
        float(cfg.get("k_alpha_human", 0.0)) * human_membership,
        float(cfg.get("k_alpha_humid", 0.0)) * humid_membership,
    ]
    reductions_beta = [
        float(cfg.get("k_beta_glacier", 0.0)) * glacier,
        float(cfg.get("k_beta_arid", 0.0)) * arid_membership,
        float(cfg.get("k_beta_human", 0.0)) * human_membership,
        float(cfg.get("k_beta_humid", 0.0)) * humid_membership,
    ]

    for reduction in reductions_alpha:
        alpha_adjusted *= 1.0 - np.clip(reduction, 0.0, 0.95)
    for reduction in reductions_beta:
        beta_adjusted *= 1.0 - np.clip(reduction, 0.0, 0.95)

    if bool(cfg.get("preserve_glacier_gradient", False)):
        beta_floor = beta_base * float(cfg.get("glacier_beta_floor_ratio", 0.0))
        beta_adjusted = np.where(glacier > 0, np.maximum(beta_adjusted, beta_floor), beta_adjusted)

    return np.clip(alpha_adjusted, 0.0, 1.0).astype(np.float32), np.clip(beta_adjusted, 0.0, 1.0).astype(np.float32)


def apply_glacier_preserved_weights(
    alpha: np.ndarray,
    beta: np.ndarray,
    glacier_fraction: np.ndarray,
    regional_weight_cfg: dict[str, float | bool] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Backward-compatible wrapper for older glacier-only callers."""
    return apply_regional_weight_adjustments(
        alpha,
        beta,
        glacier_fraction,
        aridity_mask=None,
        human_activity_index=None,
        regional_weight_cfg=regional_weight_cfg,
    )


class GraceTileDataset(Dataset):
    """Tile-based dataset over the aligned training-ready Zarr store."""

    dynamic_channels = [
        "coarse_jpl_context",
        "input_wghm_twsa",
        "input_era5_twsa",
        "input_era5_cwsc",
    ]
    static_channels = [
        "alpha_value_weight",
        "beta_gradient_weight",
        "aridity_mask",
        "human_activity_index",
        "glacier_fraction",
        "hydrobasins_mask",
        "land_mask",
    ]

    def __init__(
        self,
        zarr_path: str | Path,
        split: str,
        tile_size: int = 64,
        stride: int = 32,
        coarse_factor: int = 6,
        min_valid_fraction: float = 0.05,
        min_land_fraction: float = 0.05,
        channel_overrides: dict[str, float] | None = None,
        weight_files: dict[str, dict[str, str] | str] | None = None,
        regional_weights: dict[str, float | bool] | None = None,
    ) -> None:
        self.zarr_path = Path(zarr_path)
        self.split = split
        self.tile_size = tile_size
        self.stride = stride
        self.coarse_factor = coarse_factor
        self.min_valid_fraction = min_valid_fraction
        self.min_land_fraction = min_land_fraction
        self.channel_overrides = dict(channel_overrides or {})
        self.weight_files = dict(weight_files or {})
        self.regional_weights = dict(regional_weights or {})

        if split not in SPLIT_NAME_TO_INDEX:
            raise ValueError(f"Unsupported split: {split}")
        if tile_size % coarse_factor != 0:
            raise ValueError("tile_size must be divisible by coarse_factor.")

        self.ds = xr.open_zarr(self.zarr_path)
        self.external_weight_fields = load_external_weight_fields(
            self.weight_files,
            np.asarray(self.ds["lat"].values),
            np.asarray(self.ds["lon"].values),
        )
        self.valid_mask_np = np.asarray(self.ds["valid_mask"].values).astype(bool)
        self.land_mask_np = np.asarray(self.ds["land_mask"].values).astype(bool)
        if "cell_area_weight" in self.ds.data_vars:
            self.area_weight_np = np.asarray(self.ds["cell_area_weight"].values, dtype=np.float32)
        else:
            lat_weights = np.cos(np.deg2rad(np.asarray(self.ds["lat"].values, dtype=np.float64))).astype(np.float32)
            self.area_weight_np = np.broadcast_to(
                lat_weights[:, None],
                (int(self.ds.sizes["lat"]), int(self.ds.sizes["lon"])),
            ).copy()
        self.split_index_np = np.asarray(self.ds["split_index"].values)
        self.month_of_year_np = np.asarray(self.ds["month_of_year"].values)
        self.time_values = np.asarray(self.ds["time"].values)
        self.coarse_target_np = np.asarray(self.ds["target_jplm_twsa_coarse"].values, dtype=np.float32)
        self.has_basin_id = "hydrobasins_basin_id" in self.ds.data_vars
        self.has_reliability_truth = "closed_loop_reliability_truth" in self.ds.data_vars
        self.has_corruption_type = "closed_loop_corruption_type" in self.ds.data_vars

        self.time_indices = np.where(self.split_index_np == SPLIT_NAME_TO_INDEX[split])[0].tolist()
        self.samples = self._build_samples()
        if not self.samples:
            raise ValueError(f"No samples were found for split={split}.")

    def _build_samples(self) -> list[TileIndex]:
        """Enumerate tile samples that have enough valid land coverage."""
        samples: list[TileIndex] = []
        height = int(self.ds.sizes["lat"])
        width = int(self.ds.sizes["lon"])

        lat_starts = list(range(0, height - self.tile_size + 1, self.stride))
        lon_starts = list(range(0, width - self.tile_size + 1, self.stride))
        if lat_starts[-1] != height - self.tile_size:
            lat_starts.append(height - self.tile_size)
        if lon_starts[-1] != width - self.tile_size:
            lon_starts.append(width - self.tile_size)

        for time_idx in self.time_indices:
            valid_t = self.valid_mask_np[time_idx]
            for lat_start in lat_starts:
                lat_slice = slice(lat_start, lat_start + self.tile_size)
                for lon_start in lon_starts:
                    lon_slice = slice(lon_start, lon_start + self.tile_size)
                    tile_valid = valid_t[lat_slice, lon_slice]
                    tile_land = self.land_mask_np[lat_slice, lon_slice]
                    if tile_valid.mean() < self.min_valid_fraction:
                        continue
                    if tile_land.mean() < self.min_land_fraction:
                        continue
                    samples.append(TileIndex(time_idx=time_idx, lat_start=lat_start, lon_start=lon_start))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _slice_da(self, name: str, time_idx: int, lat_slice: slice, lon_slice: slice) -> np.ndarray:
        """Read one fine-grid tile from the Zarr store."""
        da = self.ds[name].isel(time=time_idx, lat=lat_slice, lon=lon_slice)
        return np.asarray(da.values, dtype=np.float32)

    def _slice_static(self, name: str, lat_slice: slice, lon_slice: slice) -> np.ndarray:
        """Read one static fine-grid tile from the Zarr store."""
        if name in self.external_weight_fields:
            return self.external_weight_fields[name][lat_slice, lon_slice]
        da = self.ds[name].isel(lat=lat_slice, lon=lon_slice)
        return np.asarray(da.values, dtype=np.float32)

    def _coarse_context_tile(self, time_idx: int, lat_start: int, lon_start: int) -> np.ndarray:
        """Upsample one coarse JPL supervision field and crop the requested fine-grid tile."""
        coarse = self.coarse_target_np[time_idx]
        fine = np.repeat(np.repeat(coarse, self.coarse_factor, axis=0), self.coarse_factor, axis=1)
        lat_slice = slice(lat_start, lat_start + self.tile_size)
        lon_slice = slice(lon_start, lon_start + self.tile_size)
        return fine[lat_slice, lon_slice].astype(np.float32)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        lat_slice = slice(sample.lat_start, sample.lat_start + self.tile_size)
        lon_slice = slice(sample.lon_start, sample.lon_start + self.tile_size)

        coarse_context = self._coarse_context_tile(sample.time_idx, sample.lat_start, sample.lon_start)
        wghm = self._slice_da("input_wghm_twsa", sample.time_idx, lat_slice, lon_slice)
        era5_twsa = self._slice_da("input_era5_twsa", sample.time_idx, lat_slice, lon_slice)
        era5_cwsc = self._slice_da("input_era5_cwsc", sample.time_idx, lat_slice, lon_slice)
        target = self._slice_da("target_jplm_twsa", sample.time_idx, lat_slice, lon_slice)
        valid_mask = self._slice_da("valid_mask", sample.time_idx, lat_slice, lon_slice)

        alpha = self._slice_static("alpha_value_weight", lat_slice, lon_slice)
        beta = self._slice_static("beta_gradient_weight", lat_slice, lon_slice)
        aridity = self._slice_static("aridity_mask", lat_slice, lon_slice)
        human = self._slice_static("human_activity_index", lat_slice, lon_slice)
        glacier = self._slice_static("glacier_fraction", lat_slice, lon_slice)
        hydrobasins = self._slice_static("hydrobasins_mask", lat_slice, lon_slice)
        if self.has_basin_id:
            basin_id = self._slice_static("hydrobasins_basin_id", lat_slice, lon_slice).astype(np.int32)
        else:
            basin_id = np.zeros_like(hydrobasins, dtype=np.int32)
        land_mask = self._slice_static("land_mask", lat_slice, lon_slice)
        area_weight = self.area_weight_np[lat_slice, lon_slice].astype(np.float32)

        month = int(self.month_of_year_np[sample.time_idx])
        month_angle = 2.0 * np.pi * (month - 1) / 12.0
        month_sin = np.full_like(coarse_context, np.sin(month_angle), dtype=np.float32)
        month_cos = np.full_like(coarse_context, np.cos(month_angle), dtype=np.float32)

        coarse_lat_start = sample.lat_start // self.coarse_factor
        coarse_lon_start = sample.lon_start // self.coarse_factor
        coarse_lat_slice = slice(coarse_lat_start, coarse_lat_start + self.tile_size // self.coarse_factor)
        coarse_lon_slice = slice(coarse_lon_start, coarse_lon_start + self.tile_size // self.coarse_factor)

        target_coarse = np.asarray(
            self.ds["target_jplm_twsa_coarse"]
            .isel(time=sample.time_idx, coarse_lat=coarse_lat_slice, coarse_lon=coarse_lon_slice)
            .values,
            dtype=np.float32,
        )
        wghm_coarse = np.asarray(
            self.ds["input_wghm_twsa_coarse"]
            .isel(time=sample.time_idx, coarse_lat=coarse_lat_slice, coarse_lon=coarse_lon_slice)
            .values,
            dtype=np.float32,
        )

        channel_arrays = {
            "coarse_jpl_context": coarse_context,
            "input_wghm_twsa": wghm,
            "input_era5_twsa": era5_twsa,
            "input_era5_cwsc": era5_cwsc,
            "alpha_value_weight": alpha,
            "beta_gradient_weight": beta,
            "aridity_mask": aridity,
            "human_activity_index": human,
            "glacier_fraction": glacier,
            "hydrobasins_mask": hydrobasins,
            "land_mask": land_mask,
            "month_sin": month_sin,
            "month_cos": month_cos,
        }
        for channel_name, override_value in self.channel_overrides.items():
            if channel_name in channel_arrays:
                channel_arrays[channel_name] = np.full_like(
                    channel_arrays[channel_name],
                    fill_value=np.float32(override_value),
                    dtype=np.float32,
                )

        alpha_original = np.array(channel_arrays["alpha_value_weight"], dtype=np.float32, copy=True)
        beta_original = np.array(channel_arrays["beta_gradient_weight"], dtype=np.float32, copy=True)
        alpha_adjusted, beta_adjusted = apply_regional_weight_adjustments(
            alpha_original,
            beta_original,
            channel_arrays["glacier_fraction"],
            channel_arrays["aridity_mask"],
            channel_arrays["human_activity_index"],
            self.regional_weights,
        )
        channel_arrays["alpha_value_weight"] = alpha_adjusted
        channel_arrays["beta_gradient_weight"] = beta_adjusted

        coarse_context = channel_arrays["coarse_jpl_context"]
        wghm = channel_arrays["input_wghm_twsa"]
        era5_twsa = channel_arrays["input_era5_twsa"]
        era5_cwsc = channel_arrays["input_era5_cwsc"]
        alpha = channel_arrays["alpha_value_weight"]
        beta = channel_arrays["beta_gradient_weight"]
        aridity = channel_arrays["aridity_mask"]
        human = channel_arrays["human_activity_index"]
        glacier = channel_arrays["glacier_fraction"]
        hydrobasins = channel_arrays["hydrobasins_mask"]
        land_mask = channel_arrays["land_mask"]
        month_sin = channel_arrays["month_sin"]
        month_cos = channel_arrays["month_cos"]

        inputs = np.stack([channel_arrays[name] for name in self.channel_names], axis=0)

        result = {
            "inputs": torch.from_numpy(inputs),
            "target_fine": torch.from_numpy(target[None, ...]),
            "target_coarse": torch.from_numpy(target_coarse[None, ...]),
            "wghm_fine": torch.from_numpy(wghm[None, ...]),
            "wghm_coarse": torch.from_numpy(wghm_coarse[None, ...]),
            "alpha": torch.from_numpy(alpha[None, ...]),
            "beta": torch.from_numpy(beta[None, ...]),
            "alpha_original": torch.from_numpy(alpha_original[None, ...]),
            "beta_original": torch.from_numpy(beta_original[None, ...]),
            "glacier_fraction": torch.from_numpy(glacier[None, ...]),
            "basin_id": torch.from_numpy(basin_id[None, ...]),
            "land_mask": torch.from_numpy(land_mask[None, ...]),
            "area_weight": torch.from_numpy(area_weight[None, ...]),
            "valid_mask": torch.from_numpy(valid_mask[None, ...]),
            "coarse_context": torch.from_numpy(coarse_context[None, ...]),
            "time_index": torch.tensor(sample.time_idx, dtype=torch.long),
            "month_of_year": torch.tensor(month, dtype=torch.long),
            "lat_start": torch.tensor(sample.lat_start, dtype=torch.long),
            "lon_start": torch.tensor(sample.lon_start, dtype=torch.long),
        }
        # Closed-loop truth is deliberately kept outside ``inputs``.  It is only
        # available to the synthetic gate-supervision loss and oracle control.
        if self.has_reliability_truth:
            reliability_truth = self._slice_da(
                "closed_loop_reliability_truth",
                sample.time_idx,
                lat_slice,
                lon_slice,
            )
            result["reliability_truth"] = torch.from_numpy(reliability_truth[None, ...])
        if self.has_corruption_type:
            corruption_type = self._slice_da(
                "closed_loop_corruption_type",
                sample.time_idx,
                lat_slice,
                lon_slice,
            ).astype(np.int64)
            result["corruption_type"] = torch.from_numpy(corruption_type[None, ...])
        return result

    @property
    def in_channels(self) -> int:
        return len(self.channel_names)

    @property
    def channel_names(self) -> list[str]:
        return self.dynamic_channels + self.static_channels + ["month_sin", "month_cos"]

    def get_weight_diagnostics(self) -> dict[str, np.ndarray]:
        """Return full-grid original and adjusted alpha/beta fields for diagnostics."""
        alpha_original = np.asarray(
            self.external_weight_fields.get("alpha_value_weight", self.ds["alpha_value_weight"].values),
            dtype=np.float32,
        )
        beta_original = np.asarray(
            self.external_weight_fields.get("beta_gradient_weight", self.ds["beta_gradient_weight"].values),
            dtype=np.float32,
        )
        glacier_fraction = np.asarray(self.ds["glacier_fraction"].values, dtype=np.float32)
        aridity_mask = np.asarray(self.ds["aridity_mask"].values, dtype=np.float32)
        human_activity_index = np.asarray(self.ds["human_activity_index"].values, dtype=np.float32)
        override_fields = {
            "alpha_value_weight": alpha_original,
            "beta_gradient_weight": beta_original,
            "glacier_fraction": glacier_fraction,
            "aridity_mask": aridity_mask,
            "human_activity_index": human_activity_index,
        }
        for channel_name, override_value in self.channel_overrides.items():
            if channel_name in override_fields:
                override_fields[channel_name] = np.full_like(
                    override_fields[channel_name],
                    fill_value=np.float32(override_value),
                    dtype=np.float32,
                )
        alpha_original = override_fields["alpha_value_weight"]
        beta_original = override_fields["beta_gradient_weight"]
        glacier_fraction = override_fields["glacier_fraction"]
        aridity_mask = override_fields["aridity_mask"]
        human_activity_index = override_fields["human_activity_index"]
        land_mask = np.asarray(self.ds["land_mask"].values, dtype=np.float32)
        alpha_adjusted, beta_adjusted = apply_regional_weight_adjustments(
            alpha_original,
            beta_original,
            glacier_fraction,
            aridity_mask,
            human_activity_index,
            self.regional_weights,
        )
        return {
            "alpha_original": alpha_original,
            "alpha_adjusted": alpha_adjusted * land_mask,
            "beta_original": beta_original,
            "beta_adjusted": beta_adjusted * land_mask,
            "glacier_fraction": glacier_fraction,
            "aridity_mask": aridity_mask,
            "human_activity_index": human_activity_index,
            "land_mask": land_mask,
        }
