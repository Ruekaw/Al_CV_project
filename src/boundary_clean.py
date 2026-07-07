"""4.4 支晶/噪点边界剔除（核心）。

判定规则（阈值与 um_per_pixel 绑定；未标定用像素默认值）：
  1. 噪点段：L < L_min -> 删
  2. 孤立短线：iso_dist > D_iso 且 L < L_short -> 删
  3. 支晶臂：未落在任何"闭合区域边界带"上 + 较直(低曲率) + 中等长度 -> 删
     （闭合检测用形态学法：闭运算+填孔得填实区域，取其边界带判定段是否参与闭合）

被删段单独保留为 deleted_mask，供红色覆盖复核。
"""
import numpy as np
import cv2
from skimage import morphology
from typing import List, Tuple
from .segment_features import Segment


def _closure_band(boundary_bin: np.ndarray, cfg: dict) -> np.ndarray:
    """形态学法：闭运算+填孔 -> 取填实区域边界 -> 膨胀成容差带。"""
    bc = cfg["boundary_clean"]
    k = int(bc["closure_close_size"])
    closed = cv2.morphologyEx(
        boundary_bin, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)),
    )
    filled = morphology.remove_small_holes(
        closed > 0, area_threshold=int(bc["closure_fill_area"]), connectivity=2,
    )
    eroded = morphology.binary_erosion(filled, morphology.square(3))
    band = filled & ~eroded                       # 填实区域边界（单像素级）
    tol = int(bc["on_boundary_tol"])
    band = morphology.binary_dilation(band, morphology.square(2 * tol + 1))
    return band


def clean_boundary(boundary_bin: np.ndarray, segments: List[Segment], cfg: dict,
                   junction: np.ndarray = None):
    bc = cfg["boundary_clean"]
    L_min = int(bc["L_min"])
    L_short = int(bc["L_short"])
    D_iso = float(bc["D_iso"])
    curv_max = float(bc.get("dendrite_curv_max", 0.25))   # *待首跑调参*
    len_max = int(bc.get("dendrite_len_max", L_short * 2))  # *待首跑调参*

    band = _closure_band(boundary_bin, cfg)
    band = band.astype(bool)

    keep = []
    deleted = []
    reason_cnt = {"noise": 0, "isolated": 0, "dendrite": 0}

    for seg in segments:
        L = seg.length
        if L < L_min:
            deleted.append((seg, "noise"))
            reason_cnt["noise"] += 1
            continue
        if seg.iso_dist > D_iso and L < L_short:
            deleted.append((seg, "isolated"))
            reason_cnt["isolated"] += 1
            continue
        # 是否参与闭合：段像素落在容差带上的比例
        if seg.coords.size:
            on_frac = float(band[seg.coords[:, 0], seg.coords[:, 1]].mean())
        else:
            on_frac = 0.0
        not_on_closure = on_frac < 0.5
        # 支晶臂：未闭合 + 较直 + 中等长度
        if not_on_closure and seg.curvature < curv_max and L_min <= L < len_max:
            deleted.append((seg, "dendrite"))
            reason_cnt["dendrite"] += 1
            continue
        keep.append(seg)

    # 重建保留边界与删除掩膜；交叉点(节点)始终保留，不参与删除判定
    h, w = boundary_bin.shape
    kept_bin = np.zeros((h, w), np.uint8)
    deleted_mask = np.zeros((h, w), np.uint8)
    if junction is not None:
        kept_bin[junction > 0] = 255
    for seg in keep:
        kept_bin[seg.coords[:, 0], seg.coords[:, 1]] = 255
    for seg, _reason in deleted:
        deleted_mask[seg.coords[:, 0], seg.coords[:, 1]] = 255

    info = {
        "n_total": len(segments),
        "n_keep": len(keep),
        "n_deleted": len(deleted),
        "reason_cnt": reason_cnt,
    }
    return kept_bin, deleted_mask, info
