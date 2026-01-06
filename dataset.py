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

# 模型输入定义的常量
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


def _resize_hw_to_fixed(
    arr: np.ndarray,
    target_h: int = PANGU_FIXED_H,
    target_w: int = PANGU_FIXED_W,
) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"expected 3D array [C,H,W], got {arr.shape}")
    
    C, h, w = arr.shape
    if h == target_h and w == target_w:
        return arr
    
    # 使用 PyTorch interpolate 进行双线性插值
    t = torch.from_numpy(arr).unsqueeze(0) # [1, C, H, W]
    t = F.interpolate(t, size=(target_h, target_w), mode="bilinear", align_corners=False)
    return t.squeeze(0).numpy()


def _read_upper_20ch(upper_path: str) -> np.ndarray:
    # 严格确定的变量名 (来自之前的调试)
    # 每个变量我们取前 4 层
    VARS = [
        'u_component_of_wind', 
        'v_component_of_wind', 
        'temperature', 
        'specific_humidity', 
        'geopotential'
    ]
    n_layers_per_var = 4

    cubes = []
    with xr.open_dataset(upper_path) as ds:
        for v in VARS:
            if v in ds:
                val = ds[v].values.astype(np.float32)
                # 处理可能存在的时间维度 (Time, Level, Lat, Lon) -> (Level, Lat, Lon)
                if val.ndim == 4:
                    val = val[0]
                # 确保有足够的层
                if val.shape[0] < n_layers_per_var:
                    # 极其罕见的情况，填充
                    pad = np.zeros((n_layers_per_var - val.shape[0], val.shape[1], val.shape[2]), dtype=np.float32)
                    val = np.concatenate([val, pad], axis=0)
                
                cubes.append(val[:n_layers_per_var])
            else:
                # 缺失变量填充 0 (保持通道对齐)
                # 假设无法获取形状，这通常会导致崩溃，但根据调试变量名是存在的
                warnings.warn(f"Missing Upper Variable {v} in {upper_path}")
                # 尝试猜测 H, W，这里假设如果缺失就无法继续，抛出错误更安全
                raise RuntimeError(f"Missing required variable {v} in {upper_path}")

    # 拼接: [C, H_raw, W_raw]
    up = np.concatenate(cubes, axis=0)
    # 缩放: [C, 32, 32]
    up = _resize_hw_to_fixed(up)
    return up


def _read_surface_4ch(surface_path: str) -> np.ndarray:
    # 严格确定的变量名
    VARS = [
        'temperature_2m', 
        'u_component_of_wind_10m', 
        'v_component_of_wind_10m', 
        'mean_sea_level_pressure'
    ]
    
    cubes = []
    with xr.open_dataset(surface_path) as ds:
        for v in VARS:
            if v in ds:
                val = ds[v].values.astype(np.float32)
                # 处理时间维度 (Time, Lat, Lon) -> (Lat, Lon)
                if val.ndim == 3:
                    val = val[0]
                
                # 增加通道维 (1, Lat, Lon)
                cubes.append(val[None, :, :])
            else:
                raise RuntimeError(f"Missing required variable {v} in {surface_path}")

    sf = np.concatenate(cubes, axis=0)
    sf = _resize_hw_to_fixed(sf)
    return sf


def _load_pangu_5(folder: str, anchor_ts: datetime) -> np.ndarray:
    pairs = _list_pangu_pairs_in_dir(folder)
    chosen = _choose_5_around_ts(pairs, anchor_ts)
    frames = []
    for ts_dt, up_p, sf_p in chosen:
        up20 = _read_upper_20ch(up_p)
        sf4 = _read_surface_4ch(sf_p)
        
        # 此时 up20 和 sf4 应该都是 32x32，可以直接拼接
        frame = np.concatenate([up20, sf4], axis=0) # [24, 32, 32]
        frames.append(frame)
        
    arr = np.stack(frames, axis=0) # [5, 24, 32, 32]
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

        # ----------------------------------------------------
        # 1. 加载雷达统计值
        # ----------------------------------------------------
        try:
            ns = np.load("norm_stats.npz")
            self.radar_mean = float(ns["radar_mean"])
            self.radar_std = float(ns["radar_std"])
            print(f"[{split}] Loaded Radar Stats: mean={self.radar_mean:.2f}, std={self.radar_std:.2f}")
        except Exception as e:
            warnings.warn(f"Could not load norm_stats.npz: {e}. Using default.")
            self.radar_mean = 0.0
            self.radar_std = 1.0

        # ----------------------------------------------------
        # 2. 加载盘古逐通道统计值
        # ----------------------------------------------------
        try:
            ps = np.load("pangu_channel_stats.npz")
            # 确保是 float32 并 reshape 为 (1, C, 1, 1) 用于广播
            means = ps["means"].astype(np.float32)
            stds = ps["stds"].astype(np.float32)
            
            if means.shape[0] != FIXED_WFM_C:
                 warnings.warn(f"Pangu stats shape mismatch: {means.shape} != {FIXED_WFM_C}")
            
            # 避免除以 0
            stds = np.where(stds < 1e-6, 1.0, stds)

            self.pangu_channel_means = torch.from_numpy(means).view(1, FIXED_WFM_C, 1, 1)
            self.pangu_channel_stds = torch.from_numpy(stds).view(1, FIXED_WFM_C, 1, 1)
            print(f"[{split}] Loaded Pangu Channel Stats.")
        except Exception as e:
            warnings.warn(f"Could not load pangu_channel_stats.npz: {e}. Using default.")
            self.pangu_channel_means = torch.zeros((1, FIXED_WFM_C, 1, 1))
            self.pangu_channel_stds = torch.ones((1, FIXED_WFM_C, 1, 1))

        self._rid_dir_cache: Dict[Tuple[str, str], str] = {}

    def _pangu_dir(self, ts: str, rid: str) -> str:
        cache_key = (ts, rid)
        if cache_key in self._rid_dir_cache:
            return self._rid_dir_cache[cache_key]

        root = self.pangu_root
        # 尝试递归查找包含该站点ID的文件夹
        pattern = os.path.join(root, "**", f"*{rid}*")
        candidates = [p for p in glob.glob(pattern, recursive=True) if os.path.isdir(p)]

        if not candidates:
            # 尝试去除 R 前缀查找
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
            # 如果找不到特定站点的文件夹，这可能是一个问题
            # 在某些数据集中，所有站点数据可能混在一起，或者按日期排列
            # 这里如果不做严格匹配，可以尝试按日期查找上一级目录，但为了安全先报错
            # 如果你的数据结构是按日期组织的，而不是按站点组织的，这里的逻辑可能需要调整
            # 假设按照之前的逻辑，找不到就报错
            raise FileNotFoundError(f"No Pangu dir found for RID {rid} (TS: {ts})")

        # 找到最接近时间戳的文件夹
        def parse_range_from_dirname(d: str):
            base = os.path.basename(d)
            # 匹配 2019-11-29-19-00to2019-11-29-19-00 这种格式
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

        # ----------------------------------------------------
        # 3. 读取并归一化雷达
        # ----------------------------------------------------
        arr = np.load(npy_path).astype(np.float32)
        if arr.ndim == 4 and arr.shape[0] == 49:
            radar = arr[:, 0, :, :]
        elif arr.ndim == 3 and arr.shape[-1] == 49:
            radar = np.transpose(arr, (2, 0, 1))
        else:
            raise ValueError(f"unexpected radar array shape: {arr.shape}")

        # 降采样
        radar_10 = radar[0:49:2]
        if radar_10.shape[0] != 25:
            raise RuntimeError(f"radar subsampling failed, got {radar_10.shape}")

        # 插值到固定大小 128x128
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

        # 应用雷达归一化
        x_radar = (x_radar - self.radar_mean) / (self.radar_std + 1e-8)
        y_radar = (y_radar - self.radar_mean) / (self.radar_std + 1e-8)

        # ----------------------------------------------------
        # 4. 读取并归一化盘古
        # ----------------------------------------------------
        pangu_dir = self._pangu_dir(ts_str, rid)
        anchor_dt = datetime.strptime(ts_str, "%Y%m%d_%H%M")
        
        # pangu shape: [T, 24, 32, 32]
        pangu = _load_pangu_5(pangu_dir, anchor_dt)
        pangu_t = torch.from_numpy(pangu)

        # 应用盘古逐通道归一化 [T, 24, 32, 32] - [1, 24, 1, 1]
        pangu_t = (pangu_t - self.pangu_channel_means) / (self.pangu_channel_stds + 1e-8)

        return (
            torch.from_numpy(x_radar),
            pangu_t,
            torch.from_numpy(y_radar),
        )