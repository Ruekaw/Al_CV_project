"""4.7 ASTM E112 多方向截线法。

对每个方向生成一组平行截线，沿每条线数与晶界的交点数 P_j，线长 L_j。
  L_bar = sum(L_j) / sum(P_j)            # 平均线截距长度（像素 or μm）
  d     = 1.128 * L_bar                  # 等效晶粒直径
  G     = -6.6439 * log10(L_bar_mm) - 3.288   # ASTM 晶粒度号（需物理尺度）
累计交点数达 N_min 即提前停止。
"""
import math
import numpy as np


def _line_pixels_horizontal(H, W, spacing):
    """0° 水平线：每 spacing 行一条，返回 (lines, lengths)。"""
    lines = []
    for y in range(0, H, max(1, spacing)):
        lines.append([(y, x) for x in range(W)])
    return lines


def _line_pixels_vertical(H, W, spacing):
    """90° 垂直线。"""
    lines = []
    for x in range(0, W, max(1, spacing)):
        lines.append([(y, x) for y in range(H)])
    return lines


def _line_pixels_diag45(H, W, spacing):
    """45° 对角线（row = col + b），按 b 步进。"""
    lines = []
    step = max(1, int(round(spacing * 1.4142)))
    for b in range(-(W - 1), H, step):
        pts = [(c + b, c) for c in range(W) if 0 <= c + b < H]
        if len(pts) > 1:
            lines.append(pts)
    return lines


def _line_pixels_diag135(H, W, spacing):
    """135° 对角线（row + col = b）。"""
    lines = []
    step = max(1, int(round(spacing * 1.4142)))
    for b in range(0, H + W - 1, step):
        pts = [(b - c, c) for c in range(W) if 0 <= b - c < H]
        if len(pts) > 1:
            lines.append(pts)
    return lines


_LINE_GEN = {
    0: _line_pixels_horizontal,
    90: _line_pixels_vertical,
    45: _line_pixels_diag45,
    135: _line_pixels_diag135,
}


def _count_intersections(boundary_bin: np.ndarray, pts) -> int:
    """沿有序点列采样边界，数 1-游程个数 = 交点数。"""
    if len(pts) < 2:
        return 0
    arr = np.array(pts, dtype=int)
    vals = boundary_bin[arr[:, 0], arr[:, 1]] > 0
    # 1-游程数 = 上升沿数 + (首元素是否为1)
    rising = int(np.sum(vals[1:] > vals[:-1]))
    if vals[0]:
        rising += 1
    return rising


def run_intercept(boundary_bin: np.ndarray, cfg: dict, um_per_pixel=None,
                  return_lines: bool = False):
    ic = cfg["intercept"]
    directions = list(ic["directions"])
    spacing = int(ic["line_spacing"])
    N_min = int(ic["N_min"])
    H, W = boundary_bin.shape

    total_L = 0
    total_P = 0
    per_dir = {}
    lines_for_viz = []  # [(angle, (r,c), P)] 仅可视化用

    # 多方向全部跑完，不提前停：大图上一向往往就远超 N_min，提前停会让
    # 45/90/135° 缺失，无法消除择优取向偏置。N_min 仅作"数据量是否充足"指示。
    for ang in directions:
        gen = _LINE_GEN.get(ang)
        if gen is None:
            continue
        lines = gen(H, W, spacing)
        d_L = d_P = 0
        for pts in lines:
            P = _count_intersections(boundary_bin, pts)
            L = len(pts)
            d_L += L
            d_P += P
            if return_lines and P > 0:
                mid = pts[len(pts) // 2]
                lines_for_viz.append((ang, mid, P))
        per_dir[ang] = {"L": d_L, "P": d_P, "n_lines": len(lines)}
        total_L += d_L
        total_P += d_P
    sufficient = (N_min <= 0) or (total_P >= N_min)

    if total_P <= 0:
        result = {
            "L_bar_px": None, "d_px": None, "L_bar_um": None, "d_um": None, "G": None,
            "total_intersections": 0, "total_line_length_px": total_L,
            "per_direction": per_dir, "sufficient": False,
            "note": "未检测到截线-晶界交点，请检查边界图或减小 line_spacing",
        }
        return (result, lines_for_viz) if return_lines else result

    L_bar_px = total_L / total_P
    d_px = 1.128 * L_bar_px
    result = {
        "L_bar_px": L_bar_px,
        "d_px": d_px,
        "total_intersections": int(total_P),
        "total_line_length_px": int(total_L),
        "per_direction": per_dir,
        "sufficient": bool(sufficient),
        "n_directions_used": len(per_dir),
    }
    if um_per_pixel:
        L_bar_um = L_bar_px * um_per_pixel
        d_um = 1.128 * L_bar_um
        L_bar_mm = L_bar_um / 1000.0
        G = -6.6439 * math.log10(L_bar_mm) - 3.288 if L_bar_mm > 0 else None
        result.update({"L_bar_um": L_bar_um, "d_um": d_um, "G": G,
                       "unit": "um", "um_per_pixel": um_per_pixel})
    else:
        result.update({"L_bar_um": None, "d_um": None, "G": None,
                       "unit": "px", "um_per_pixel": None,
                       "note": "um_per_pixel 未标定，结果为像素单位（未标定）"})
    if not sufficient:
        result["warning"] = f"累计交点 {total_P} < N_min={N_min}，统计意义偏弱"
    return (result, lines_for_viz) if return_lines else result
