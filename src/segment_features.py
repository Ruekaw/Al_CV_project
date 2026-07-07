"""4.3 边界段切分与特征提取（向量化，适配 6M 像素大图）。

关键：直接对骨架网做 connectedComponents 只会得到 1 段（整网连通），
无法做长度/支晶判定。本实现按"网络节点"切分：
  - 节点 = 交叉点(邻域数>=3) 或 端点(邻域数==1)；
  - 在交叉点处把骨架切开，每条连通分支 = 一段"边界弧"(两个节点之间)；
  - 闭环分支(无节点)单独成段。

每段提取：长度/端点(节点侧端)/端点外向切向/曲率/到最近长段距离。
"""
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import cv2
from skimage import morphology
from scipy.ndimage import convolve, binary_dilation
from scipy.spatial import cKDTree

_NB_KERNEL = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)


@dataclass
class Segment:
    sid: int
    coords: np.ndarray              # (N,2) 该段骨架点（无序）
    length: int                     # 骨架点数（≈像素长度）
    endpoints: np.ndarray           # (E,2) 段端点（节点侧端，E 通常为 0 或 2）
    tangent: list                   # 每个端点的外向单位切向
    curvature: float                # 1 - chord/length，越大越弯
    iso_dist: float = -1.0          # 到最近"长段"的最近像素距离；无长段时 -1


def _local_inward_point(cut_bool: np.ndarray, ep, n_fit: int):
    """从端点 ep 沿分支(已去交叉点)向内走 n_fit 步，返回内点。"""
    H, W = cut_bool.shape
    prev = None
    cur = tuple(ep)
    for _ in range(n_fit):
        nxt = None
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = cur[0] + dr, cur[1] + dc
                if 0 <= nr < H and 0 <= nc < W and cut_bool[nr, nc] and (nr, nc) != prev:
                    nxt = (nr, nc)
                    break
            if nxt is not None:
                break
        if nxt is None:
            break
        prev, cur = cur, nxt
    return np.array(cur, dtype=float)


def extract_segments(boundary_bin: np.ndarray, cfg: dict) -> Tuple[List[Segment], np.ndarray]:
    """返回 (segments, junction_mask)。junction_mask 供 clean_boundary 重建时保留交叉点。"""
    skel = morphology.skeletonize(boundary_bin > 0, method=cfg["segment"]["skeleton_method"])
    skel_bool = skel > 0
    if not skel_bool.any():
        return [], np.zeros_like(skel_bool, dtype=bool)

    nb = convolve(skel_bool.astype(np.uint8), _NB_KERNEL, mode="constant", cval=0)
    junction = skel_bool & (nb >= 3)
    endpoint = skel_bool & (nb == 1)
    cut_bool = skel_bool & ~junction

    num, labels = cv2.connectedComponents(np.uint8(cut_bool), connectivity=8)
    if num <= 1:
        # 无分支，只有交叉点
        return [], junction

    # 分支按标签分组（一次 argwhere + argsort + split）
    coords_all = np.argwhere(labels > 0)
    labs = labels[coords_all[:, 0], coords_all[:, 1]]
    order = np.argsort(labs, kind="stable")
    coords_all = coords_all[order]
    labs = labs[order]
    bounds = np.where(np.diff(labs))[0] + 1
    groups = np.split(coords_all, bounds)  # 每组对应一个标签 1..num-1

    # 段端点：分支像素中 (a) 与交叉点相邻 或 (b) 本身是全局端点
    jun_adj = binary_dilation(junction, _NB_KERNEL) & cut_bool
    terminal_mask = jun_adj | endpoint
    term_coords = np.argwhere(terminal_mask)
    if len(term_coords):
        term_labs = labels[term_coords[:, 0], term_coords[:, 1]]
    else:
        term_labs = np.empty(0, dtype=labels.dtype)
    term_dict = {}
    for tc, tl in zip(term_coords, term_labs):
        term_dict.setdefault(int(tl), []).append(tc)

    n_fit = int(cfg["segment"]["endpoint_fit_n"])
    L_short = int(cfg["boundary_clean"]["L_short"])

    segs: List[Segment] = []
    long_coords_list = []
    for sid, g in enumerate(groups, start=1):
        length = len(g)
        eps = np.array(term_dict.get(sid, []), dtype=int).reshape(-1, 2)
        tans = []
        for ep in eps:
            inner = _local_inward_point(cut_bool, ep, n_fit)
            t = np.array([ep[0] - inner[0], ep[1] - inner[1]], dtype=float)
            n = np.hypot(t[0], t[1])
            tans.append(t / n if n > 1e-6 else np.array([0.0, 0.0]))
        if len(eps) >= 2 and length > 2:
            d = np.hypot(eps[:, None, 0] - eps[None, :, 0],
                         eps[:, None, 1] - eps[None, :, 1])
            chord = float(d.max())
            curvature = float(1.0 - chord / length)
        else:
            curvature = 0.0
        seg = Segment(sid=sid, coords=g, length=length,
                      endpoints=eps, tangent=tans, curvature=curvature)
        segs.append(seg)
        if length >= L_short:
            long_coords_list.append(g)

    # iso_dist：非长段到最近长段像素的距离
    if long_coords_list:
        long_pts = np.vstack(long_coords_list)
        tree = cKDTree(long_pts)
        for seg in segs:
            if seg.length >= L_short:
                seg.iso_dist = 0.0
                continue
            if len(seg.coords):
                d, _ = tree.query(seg.coords, k=1)
                seg.iso_dist = float(np.min(d))
            else:
                seg.iso_dist = -1.0
    else:
        for seg in segs:
            seg.iso_dist = -1.0
    return segs, junction
