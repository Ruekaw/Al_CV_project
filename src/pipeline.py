"""pipeline：串起 4.1 -> 4.5（一期）+ 4.7（二期）。

run_pipeline(image_path, cfg, runs_dir) -> result dict，并把各步产物写到 runs_dir/{图名}/。
"""
import os
import json
import time
import numpy as np
import cv2
from skimage import morphology

from .image_io import imread_unicode, imwrite_unicode
from .preprocess import preprocess
from .candidate_boundary import generate_candidate_boundary
from .segment_features import extract_segments
from .boundary_clean import clean_boundary
from .boundary_repair import repair_boundary
from .intercept import run_intercept
from .viz import boundary_overlay, intercept_overlay, make_report


def _closure_count(boundary_bin: np.ndarray, cfg: dict) -> int:
    """闭合自检：闭运算+填孔后数填实区域个数（≈晶粒数）。"""
    k = int(cfg["boundary_clean"]["closure_close_size"])
    closed = cv2.morphologyEx(
        boundary_bin, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)),
    )
    filled = morphology.remove_small_holes(
        closed > 0, area_threshold=int(cfg["boundary_clean"]["closure_fill_area"]), connectivity=2,
    )
    filled = morphology.remove_small_objects(filled, min_size=200, connectivity=2)
    num, _ = cv2.connectedComponents(np.uint8(filled > 0), connectivity=8)
    return num - 1  # 去掉背景


def run_pipeline(image_path: str, cfg: dict, runs_dir: str = "runs") -> dict:
    name = os.path.splitext(os.path.basename(image_path))[0]
    out_dir = os.path.join(runs_dir, name)
    os.makedirs(out_dir, exist_ok=True)
    save_steps = bool(cfg["output"]["save_steps"])

    t0 = time.time()
    image_bgr = imread_unicode(image_path)

    # 4.1 预处理
    enhanced, um_per_pixel = preprocess(image_bgr, cfg)
    if save_steps:
        imwrite_unicode(os.path.join(out_dir, "01_preprocess.png"), enhanced)

    # 4.2 候选边界
    candidate, method = generate_candidate_boundary(image_bgr, cfg)
    if save_steps:
        imwrite_unicode(os.path.join(out_dir, "02_candidate_boundary.png"), candidate)

    # 4.3 段切分（返回段集合 + 交叉点掩膜，交叉点在剔除时始终保留）
    segments, junction = extract_segments(candidate, cfg)

    # 4.4 剔除
    cleaned, deleted, clean_info = clean_boundary(candidate, segments, cfg, junction=junction)
    if save_steps:
        imwrite_unicode(
            os.path.join(out_dir, "03_cleaned_boundary.png"),
            boundary_overlay(image_bgr, cleaned, deleted=deleted, added=None),
        )

    # 4.5 断口修复
    final, added, repair_info = repair_boundary(cleaned, cfg)
    if save_steps:
        imwrite_unicode(
            os.path.join(out_dir, "04_final_boundary.png"),
            boundary_overlay(image_bgr, final, deleted=deleted, added=added),
        )

    # 4.7 截线法（基于最终闭合晶界图）
    intercept_res, lines_viz = run_intercept(
        final, cfg, um_per_pixel=um_per_pixel, return_lines=True,
    )
    if save_steps:
        imwrite_unicode(
            os.path.join(out_dir, "05_intercept.png"),
            intercept_overlay(image_bgr, final, lines_viz),
        )

    # 闭合自检
    n_grains = _closure_count(final, cfg)

    # 复核大图
    try:
        make_report(
            [
                image_bgr,
                enhanced,
                candidate,
                boundary_overlay(image_bgr, cleaned, deleted=deleted),
                boundary_overlay(image_bgr, final, added=added),
                intercept_overlay(image_bgr, final, lines_viz),
            ],
            ["原图", "预处理", "候选边界", f"剔除(删{clean_info['n_deleted']})",
             f"修复(补{repair_info['n_repaired']})", "截线"],
            os.path.join(out_dir, "report.png"),
        )
    except Exception as e:
        print(f"[viz] report 生成失败: {e}")

    result = {
        "image": name,
        "method": method,
        "shape": list(image_bgr.shape[:2]),
        "um_per_pixel": um_per_pixel,
        "segments": clean_info,
        "repair": repair_info,
        "closure_grain_count": n_grains,
        "intercept": intercept_res,
        "elapsed_sec": round(time.time() - t0, 2),
    }
    with open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result
