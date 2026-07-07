"""可视化：单步图保存 + 删除/补连叠加 + 截线叠加 + 四联/多联复核图。"""
import numpy as np
import cv2
from .image_io import imwrite_unicode


def _bgr_to_rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr.ndim == 3 else bgr


def overlay_on_original(orig_bgr: np.ndarray, mask: np.ndarray, color) -> np.ndarray:
    """把二值掩膜按颜色叠加到原图（RGB）。"""
    out = orig_bgr.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    m = mask > 0
    for ch in range(3):
        ch_img = out[:, :, ch]
        ch_img[m] = color[ch]
    return out


def boundary_overlay(orig_bgr: np.ndarray, boundary_bin: np.ndarray,
                     deleted=None, added=None) -> np.ndarray:
    """白=保留边界，红=被删段，蓝=补连段。"""
    out = orig_bgr.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    if boundary_bin is not None:
        m = boundary_bin > 0
        out[m] = (235, 235, 235)
    if deleted is not None and deleted.any():
        m = deleted > 0
        out[m] = (60, 60, 230)   # 红
    if added is not None and added.any():
        m = added > 0
        out[m] = (230, 160, 60)  # 蓝
    return out


def intercept_overlay(orig_bgr: np.ndarray, boundary_bin: np.ndarray,
                      lines_for_viz) -> np.ndarray:
    """在图上画截线交点（按方向着色）+ 边界半透明灰。"""
    out = orig_bgr.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    out[boundary_bin > 0] = (200, 200, 200)
    dir_color = {0: (0, 200, 200), 45: (200, 0, 200), 90: (0, 0, 220), 135: (220, 160, 0)}
    for ang, (r, c), _P in lines_for_viz:
        col = dir_color.get(ang, (0, 255, 0))
        cv2.circle(out, (int(c), int(r)), 3, col, -1)
    return out


def make_report(images: list, titles: list, save_path: str) -> None:
    """多图横向拼接成复核大图。images 已是 BGR 或灰度。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # 中文字体：Windows 优先微软雅黑/黑体，避免标题显示成方块
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, img, t in zip(axes, images, titles):
        ax.imshow(_bgr_to_rgb(img), cmap="gray" if img.ndim == 2 else None)
        ax.set_title(t, fontsize=11)
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_step(path: str, img: np.ndarray) -> None:
    imwrite_unicode(path, img)
