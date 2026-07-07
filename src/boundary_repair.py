"""4.5 晶界断口修复。

对每个端点 p：在 R_connect 内找候选端点 q，要求
  p->q 方向与 p 处外向切向夹角 < angle_thresh，
  q->p 方向与 q 处外向切向夹角 < angle_thresh，
满足则直线连接 p、q（双向约束防误连）。
修复后再骨架化保证单像素宽。补连段单独存 added_mask 供蓝色高亮。
"""
import numpy as np
import cv2
from skimage import morphology


def _skeleton_endpoints(skel_bin: np.ndarray):
    """返回端点坐标数组 (M,2) 与每个端点的局部外向切向 (M,2)。

    端点定义：8 邻域内恰好 1 个骨架邻居。向量化检出（3x3 卷积数邻域），
    切向 = 端点 - 沿分支向内 n_fit 步的位置，每端点 O(n_fit*8)。
    """
    from scipy.ndimage import convolve
    skel = skel_bin > 0
    if not skel.any():
        return np.empty((0, 2), dtype=int), np.empty((0, 2), dtype=float)
    H, W = skel.shape
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    nb = convolve(skel.astype(np.uint8), kernel, mode="constant", cval=0)
    eps = np.argwhere(skel & (nb == 1))
    if len(eps) == 0:
        return np.empty((0, 2), dtype=int), np.empty((0, 2), dtype=float)

    n_fit = 6
    tangents = np.zeros((len(eps), 2), dtype=float)
    for i, ep in enumerate(eps):
        prev = None
        cur = tuple(ep)
        for _ in range(n_fit):
            nxt = None
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = cur[0] + dr, cur[1] + dc
                    if 0 <= nr < H and 0 <= nc < W and skel[nr, nc] and (nr, nc) != prev:
                        nxt = (nr, nc)
                        break
                if nxt is not None:
                    break
            if nxt is None:
                break
            prev, cur = cur, nxt
        t = np.array([ep[0] - cur[0], ep[1] - cur[1]], dtype=float)
        n = np.hypot(t[0], t[1])
        tangents[i] = t / n if n > 1e-6 else np.array([0.0, 0.0])
    return eps.astype(int), tangents


def repair_boundary(cleaned_bin: np.ndarray, cfg: dict):
    rp = cfg["repair"]
    R = float(rp["R_connect"])
    ang_thr = float(rp["angle_thresh"])

    skel = morphology.skeletonize(cleaned_bin > 0, method="lee")
    skel_bin = (skel > 0).astype(np.uint8) * 255
    eps, tans = _skeleton_endpoints(skel_bin)
    if len(eps) < 2:
        return skel_bin, np.zeros_like(skel_bin), {"n_endpoints": int(len(eps)), "n_repaired": 0}

    try:
        from scipy.spatial import cKDTree
    except Exception:
        cKDTree = None

    added = np.zeros_like(skel_bin)
    used = set()
    n_rep = 0
    cos_thr = np.cos(np.deg2rad(ang_thr))

    if cKDTree is not None and len(eps) > 1:
        tree = cKDTree(eps)
        pairs = tree.query_pairs(r=R, output_type="ndarray")
        # 按距离从小到大排序，优先连最近的
        if len(pairs):
            dvec = eps[pairs[:, 0]] - eps[pairs[:, 1]]
            dists = np.hypot(dvec[:, 0], dvec[:, 1])
            order = np.argsort(dists)
            pairs = pairs[order]
        for i, j in pairs:
            if i in used or j in used:
                continue
            p, q = eps[i], eps[j]
            d = q - p
            dist = np.hypot(d[0], d[1])
            if dist < 1:
                continue
            d_uv = d / dist
            # p 处切向应指向 q；q 处切向应指向 p
            if tans[i].dot(d_uv) < cos_thr:
                continue
            if tans[j].dot(-d_uv) < cos_thr:
                continue
            cv2.line(added, (int(p[1]), int(p[0])), (int(q[1]), int(q[0])), 255, 1)
            used.add(i)
            used.add(j)
            n_rep += 1

    repaired = cv2.bitwise_or(skel_bin, added)
    repaired = (morphology.skeletonize(repaired > 0, method="lee") > 0).astype(np.uint8) * 255
    info = {"n_endpoints": int(len(eps)), "n_repaired": int(n_rep)}
    return repaired, added, info
