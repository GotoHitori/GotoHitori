import os
import re
import glob
import csv
import warnings
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
import xarray as xr
import torch.nn.functional as F

FIXED_WFM_C = 24
RADAR_IN_T = 5
RADAR_OUT_T = 20
PANGU_T = 5
PANGU_FIXED_H = 32
PANGU_FIXED_W = 32
RADAR_FIXED_H = 128
RADAR_FIXED_W = 128


def _pad_hw_4d(x, Ht, Wt):
    h, w = x.shape[-2], x.shape[-1]
    ph, pw = Ht - h, Wt - w
    if ph < 0 or pw < 0:
        x = x[..., :min(h, Ht), :min(w, Wt)]
        h, w = x.shape[-2], x.shape[-1]
        ph, pw = Ht - h, Wt - w
    if ph == 0 and pw == 0:
        return x
    return F.pad(x, (0, pw, 0, ph), mode="replicate")


def load_catalog(path: str) -> Dict[str, Dict[str, str]]:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    tbl: Dict[str, Dict[str, str]] = {}
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = (
                row.get("radar_id")
                or row.get("vil_id")
                or row.get("id")
                or row.get("rid")
                or row.get("station")
            )
            if not rid:
                continue
            rid = str(rid).strip().upper()

            m = re.match(r"^R?(\d+)$", rid)
            if m:
                rid_norm = f"R{m.group(1)}"
            else:
                rid_norm = rid if rid.startswith("R") else f"R{rid}"

            tbl[rid_norm] = row
    return tbl


def parse_fname(fname: str) -> Tuple[str, str]:
    m = re.match(r"(\d{8}_\d{4})_R(\d+)", fname)
    if m is None:
        raise ValueError(f"invalid filename: {fname}")
    return m.group(1), f"R{m.group(2)}"


_TS_RE = re.compile(r"(\d{4}-\d{2}-\d{2}-\d{2}-\d{2})")


def _list_pangu_pairs_in_dir(folder: str) -> List[Tuple[datetime, str, str]]:
    up_files = sorted(glob.glob(os.path.join(folder, "*upper*.nc")))
    sf_files = sorted(glob.glob(os.path.join(folder, "*surface*.nc")))
    if not up_files or not sf_files:
        raise RuntimeError(f"[Pangu] missing upper/surface files under: {folder}")

    def ts_from_name(p: str) -> Optional[datetime]:
        m = _TS_RE.search(os.path.basename(p))
        if not m:
            return None
        return datetime.strptime(m.group(1), "%Y-%m-%d-%H-%M")

    up_map: Dict[datetime, str] = {}
    for p in up_files:
        ts = ts_from_name(p)
        if ts:
            up_map[ts] = p

    pairs: List[Tuple[datetime, str, str]] = []
    for p in sf_files:
        ts = ts_from_name(p)
        if ts and ts in up_map:
            pairs.append((ts, up_map[ts], p))

    if not pairs:
        raise RuntimeError(f"[Pangu] no matched upper/surface pairs in {folder}")
    pairs.sort(key=lambda x: x[0])
    return pairs


def _choose_5_around_ts(
    pairs: List[Tuple[datetime, str, str]], ts_dt: datetime
) -> List[Tuple[datetime, str, str]]:
    if len(pairs) <= PANGU_T:
        return pairs[:PANGU_T]

    idx = -1
    for i, (t, _, _) in enumerate(pairs):
        if t <= ts_dt:
            idx = i
        else:
            break

    if idx < 0:
        return pairs[:PANGU_T]

    start = max(0, idx - (PANGU_T - 1))
    end = start + PANGU_T
    if end > len(pairs):
        end = len(pairs)
        start = end - PANGU_T
    return pairs[start:end]


def _normalize_field(a: np.ndarray) -> np.ndarray:
    arr = np.asarray(a)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3:
        return arr.astype(np.float32)
    if arr.ndim == 2:
        return arr[None].astype(np.float32)
    raise ValueError(f"unexpected var shape {arr.shape}")


def _resize_hw_to_fixed(
    arr: np.ndarray,
    target_h: int = PANGU_FIXED_H,
    target_w: int = PANGU_FIXED_W,
) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"expected 3D array [C,H,W], got {arr.shape}")
    _, h, w = arr.shape
    if h == target_h and w == target_w:
        return arr
    t = torch.from_numpy(arr).unsqueeze(0)
    t = F.interpolate(t, size=(target_h, target_w), mode="bilinear", align_corners=False)
    return t.squeeze(0).numpy()


def _select_vars(
    ds: xr.Dataset, prefer: List[str], need: int, must_have_level: bool
) -> List[str]:
    vars_all = list(ds.data_vars)
    chosen: List[str] = []

    def has_level(v: str) -> bool:
        dims = ds[v].dims
        return any(
            d.lower() in ("level", "lev", "isobaricinpa", "isobaric", "pressure", "plev")
            for d in dims
        )

    for v in prefer:
        if v in vars_all and (not must_have_level or has_level(v)):
            chosen.append(v)
        if len(chosen) >= need:
            return chosen

    for v in vars_all:
        if v in chosen:
            continue
        if must_have_level and (not has_level(v)):
            continue
        chosen.append(v)
        if len(chosen) >= need:
            break

    return chosen[:need]


def _read_upper_20ch(
    upper_path: str,
    prefer_vars: Optional[List[str]] = None,
    n_layers: int = 4,
    n_vars: int = 5,
) -> np.ndarray:
    prefer_vars = prefer_vars or [
        "u",
        "v",
        "t",
        "q",
        "z",
        "u_component_of_wind",
        "v_component_of_wind",
        "temperature",
        "specific_humidity",
        "geopotential_height",
    ]
    with xr.open_dataset(upper_path) as ds:
        var_names = _select_vars(ds, prefer_vars, need=n_vars, must_have_level=True)
        cubes = []
        for v in var_names:
            a = _normalize_field(ds[v].values)
            if a.shape[0] < n_layers:
                pad = np.zeros(
                    (n_layers - a.shape[0], a.shape[1], a.shape[2]),
                    dtype=np.float32,
                )
                a = np.concatenate([a, pad], axis=0)
            cubes.append(a[:n_layers])

        if len(cubes) == 0:
            raise RuntimeError(f"No usable upper variables in {upper_path}")

        up = np.concatenate(cubes, axis=0)
        up = _resize_hw_to_fixed(up)
        return up.astype(np.float32)


def _read_surface_4ch(
    surface_path: str,
    prefer_vars: Optional[List[str]] = None,
    n_vars: int = 4,
) -> np.ndarray:
    prefer_vars = prefer_vars or ["t2m", "u10", "v10", "msl"]
    with xr.open_dataset(surface_path) as ds:
        var_names = _select_vars(ds, prefer_vars, need=n_vars, must_have_level=False)
        cubes = []
        for v in var_names:
            a = _normalize_field(ds[v].values)
            a = a[:1]
            cubes.append(a)

        if len(cubes) == 0:
            raise RuntimeError(f"No usable surface variables in {surface_path}")

        sf = np.concatenate(cubes, axis=0)

        if sf.shape[0] < n_vars:
            h, w = sf.shape[1], sf.shape[2]
            pad = np.zeros((n_vars - sf.shape[0], h, w), dtype=np.float32)
            sf = np.concatenate([sf, pad], axis=0)
        elif sf.shape[0] > n_vars:
            sf = sf[:n_vars]

        sf = _resize_hw_to_fixed(sf)
        return sf.astype(np.float32)


def _load_pangu_5(folder: str, anchor_ts: datetime) -> np.ndarray:
    pairs = _list_pangu_pairs_in_dir(folder)
    chosen = _choose_5_around_ts(pairs, anchor_ts)
    frames = []
    for ts_dt, up_p, sf_p in chosen:
        up20 = _read_upper_20ch(up_p)
        sf4 = _read_surface_4ch(sf_p)
        if up20.shape[1:] != sf4.shape[1:]:
            raise RuntimeError(
                f"Upper/Surface spatial mismatch: {up20.shape[1:]} vs {sf4.shape[1:]}"
            )
        frame = np.concatenate([up20, sf4], axis=0)
        frames.append(frame.astype(np.float32))
    arr = np.stack(frames, axis=0)
    return arr.astype(np.float32)


class RadarPanguDataset(Dataset):
    def __init__(
        self,
        radar_dir: str,
        pangu_root: str,
        catalog_csv: str,
        split: str = "train",
    ):
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train|val|test")

        self.radar_dir = os.path.join(radar_dir, split)
        self.pangu_root = pangu_root
        self.catalog = load_catalog(catalog_csv)
        self.split = split

        all_files = glob.glob(os.path.join(self.radar_dir, "*.npy"))
        self.files: List[str] = []
        for p in all_files:
            try:
                ts, rid = parse_fname(os.path.basename(p))
                if rid in self.catalog:
                    self.files.append(p)
            except Exception:
                continue

        if not self.files:
            raise RuntimeError(f"no usable radar files in {self.radar_dir}")

        try:
            ns = np.load("norm_stats.npz")
            self.radar_mean = float(ns["radar_mean"])
            self.radar_std = float(ns["radar_std"])
        except Exception as e:
            warnings.warn(f"{e}; fall back to no-normalization for Radar.")
            self.radar_mean = 0.0
            self.radar_std = 1.0

        try:
            ps = np.load("pangu_channel_stats.npz")
            means = np.asarray(
                ps.get("channel_means", ps.get("means", np.zeros(FIXED_WFM_C))),
                dtype=np.float32,
            ).reshape(-1)
            stds = np.asarray(
                ps.get("channel_stds", ps.get("stds", np.ones(FIXED_WFM_C))),
                dtype=np.float32,
            ).reshape(-1)

            if means.size != FIXED_WFM_C:
                warnings.warn(
                    f"pangu_channel_means size {means.size} != {FIXED_WFM_C}; auto-fix."
                )
                if means.size < FIXED_WFM_C:
                    means = np.pad(means, (0, FIXED_WFM_C - means.size))
                else:
                    means = means[:FIXED_WFM_C]

            if stds.size != FIXED_WFM_C:
                warnings.warn(
                    f"pangu_channel_stds size {stds.size} != {FIXED_WFM_C}; auto-fix."
                )
                if stds.size < FIXED_WFM_C:
                    stds = np.pad(
                        stds,
                        (0, FIXED_WFM_C - stds.size),
                        constant_values=1.0,
                    )
                else:
                    stds = stds[:FIXED_WFM_C]

            stds = np.where(stds == 0, 1.0, stds)
            self.pangu_channel_means = means.astype(np.float32)
            self.pangu_channel_stds = stds.astype(np.float32)
        except Exception as e:
            warnings.warn(f"{e}; fall back to no-normalization for Pangu.")
            self.pangu_channel_means = np.zeros((FIXED_WFM_C,), dtype=np.float32)
            self.pangu_channel_stds = np.ones((FIXED_WFM_C,), dtype=np.float32)

        self._rid_dir_cache: Dict[Tuple[str, str], str] = {}

    def _pangu_dir(self, ts: str, rid: str) -> str:
        cache_key = (ts, rid)
        if cache_key in self._rid_dir_cache:
            return self._rid_dir_cache[cache_key]

        root = self.pangu_root
        pattern = os.path.join(root, "**", f"*{rid}*")
        candidates = [p for p in glob.glob(pattern, recursive=True) if os.path.isdir(p)]

        if not candidates:
            rid_digits = re.sub(r"\D", "", rid or "")
            pattern2 = os.path.join(root, "**", f"*R{rid_digits}*")
            candidates = [
                p for p in glob.glob(pattern2, recursive=True) if os.path.isdir(p)
            ]

        def has_pair(d: str) -> bool:
            ups = glob.glob(os.path.join(d, "*upper*.nc"))
            sfs = glob.glob(os.path.join(d, "*surface*.nc"))
            return len(ups) > 0 and len(sfs) > 0

        candidates = [d for d in candidates if has_pair(d)]
        if not candidates:
            raise FileNotFoundError(f"No Pangu dir found under {root} for RID {rid}")

        def parse_range_from_dirname(d: str):
            base = os.path.basename(d)
            m = re.search(
                r"(\d{4}-\d{2}-\d{2}-\d{2}-\d{2})to(\d{4}-\d{2}-\d{2}-\d{2}-\d{2})",
                base,
            )
            if not m:
                return None, None
            s = datetime.strptime(m.group(1), "%Y-%m-%d-%H-%M")
            e = datetime.strptime(m.group(2), "%Y-%m-%d-%H-%M")
            return s, e

        dt_ts = datetime.strptime(ts, "%Y%m%d_%H%M")
        within = []
        scored = []

        for d in candidates:
            s, e = parse_range_from_dirname(d)
            if s is not None and e is not None:
                if s <= dt_ts <= e:
                    within.append((abs((dt_ts - s).total_seconds()), d))
                else:
                    scored.append((abs((dt_ts - s).total_seconds()), d))
            else:
                scored.append((float("inf"), d))

        if within:
            chosen = sorted(within, key=lambda x: x[0])[0][1]
        elif scored:
            chosen = sorted(scored, key=lambda x: x[0])[0][1]
        else:
            chosen = candidates[0]

        self._rid_dir_cache[cache_key] = chosen
        return chosen

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        npy_path = self.files[idx]
        ts_str, rid = parse_fname(os.path.basename(npy_path))

        arr = np.load(npy_path).astype(np.float32)
        if arr.ndim == 4 and arr.shape[0] == 49:
            radar = arr[:, 0, :, :]
        elif arr.ndim == 3 and arr.shape[-1] == 49:
            radar = np.transpose(arr, (2, 0, 1))
        else:
            raise ValueError(f"unexpected radar array shape: {arr.shape}")

        radar_10 = radar[0:49:2]
        if radar_10.shape[0] != 25:
            raise RuntimeError(f"radar subsampling failed, got {radar_10.shape}")

        radar_10_t = torch.from_numpy(radar_10).unsqueeze(1)
        radar_10_t = F.interpolate(
            radar_10_t,
            size=(RADAR_FIXED_H, RADAR_FIXED_W),
            mode="bilinear",
            align_corners=False,
        )
        radar_10 = radar_10_t.squeeze(1).numpy()

        x_radar = radar_10[:RADAR_IN_T, None, ...]
        y_radar = radar_10[RADAR_IN_T : RADAR_IN_T + RADAR_OUT_T, None, ...]

        x_radar = (x_radar - self.radar_mean) / (self.radar_std + 1e-8)
        y_radar = (y_radar - self.radar_mean) / (self.radar_std + 1e-8)

        pangu_dir = self._pangu_dir(ts_str, rid)
        anchor_dt = datetime.strptime(ts_str, "%Y%m%d_%H%M")
        pangu = _load_pangu_5(pangu_dir, anchor_dt)
        pangu = (pangu - self.pangu_channel_means[None, :, None, None]) / (
            self.pangu_channel_stds[None, :, None, None] + 1e-8
        )

        return (
            torch.from_numpy(x_radar),
            torch.from_numpy(pangu),
            torch.from_numpy(y_radar),
        )
