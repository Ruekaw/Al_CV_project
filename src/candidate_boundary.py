"""4.2 候选边界生成。

两条同构路径，输出均为单通道二值细线边界图（0/255）：
  - matsam：MatSAM 主路径（需 torch + SAM 权重，GPU 推荐）
  - baseline：传统法（CLAHE/自适应阈值/Canny + 形态学 + 骨架化），兜底 + 对照

MatSAM 路径对 torch/SAM 权重/GPU 做延迟导入与失败守护；任一环节不可用即返回 None，
由 generate() 自动降级到 baseline 并在返回值里标注 method。
"""
import sys
import numpy as np
import cv2
from skimage import morphology

from .image_io import imread_unicode  # noqa: F401  (re-export for convenience)


# ---------------------------------------------------------------- baseline
def _baseline_boundary(image_bgr: np.ndarray, cfg: dict) -> np.ndarray:
    """传统法候选边界：暗晶界提取。

    逻辑：CLAHE 增强 -> 全局暗阈值取晶界前景 -> 开运算去细碎划痕/纹理
          -> 闭运算连断 -> 骨架化 -> 去小连通域。

    刻意不用 Canny / 自适应阈值：
      - Canny 是高频边缘检测，对抛光划痕和晶粒内部纹理极敏感，会把高频噪声当边界；
      - 自适应阈值取"局部相对暗"，晶粒内部灰度起伏也会被当前景，画满细线。
    真实晶界是"全局较暗 + 较粗 + 连续成网"，用全局暗阈值 + 尺度形态学更贴合。
    """
    from .preprocess import preprocess
    gray, _ = preprocess(image_bgr, cfg)
    b = cfg["baseline"]

    # 1. 全局暗阈值：把晶界（暗）当前景
    mode = b.get("threshold_mode", "percentile")
    if mode == "otsu":
        _, dark = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    elif mode == "mean_std":
        thr = int(gray.mean() - float(b["dark_k_std"]) * gray.std())
        dark = np.uint8(gray < thr) * 255
    else:  # percentile（默认，对晶界面积占比不敏感，最可控）
        thr = int(np.percentile(gray, float(b["dark_percentile"])))
        dark = np.uint8(gray < thr) * 255

    # 2. 开运算：去细碎划痕/纹理（核需大于划痕宽度、小于晶界宽度）
    open_k = int(b["morph_open_size"])
    if open_k > 1:
        dark = cv2.morphologyEx(
            dark, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k)),
        )

    # 3. 闭运算：连断口
    close_k = int(b["morph_close_size"])
    if close_k > 1:
        dark = cv2.morphologyEx(
            dark, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k)),
        )

    # 4. 骨架化 -> 去小连通域 -> 单像素宽细线
    skel = morphology.skeletonize(dark > 0, method="lee")
    cleaned = morphology.remove_small_objects(
        skel, min_size=int(b["min_segment_size"]), connectivity=2,
    )
    return (cleaned.astype(np.uint8)) * 255


# ---------------------------------------------------------------- matsam
def _matsam_boundary(image_bgr: np.ndarray, cfg: dict):
    """MatSAM 主路径。失败/降级时返回 None（不抛异常到外层）。

    复用 MatSAM/utils 的 PromptGenerator + segment_anything_。MatSAM 非 Python 包，
    这里把 MatSAM 目录临时塞进 sys.path 后再 import。
    """
    m = cfg["matsam"]
    if not m["enabled"]:
        return None

    try:
        import torch  # 延迟导入：baseline 路径与框架骨架无需 torch
    except Exception as e:
        print(f"[matsam] torch 不可用，降级 baseline: {e}")
        return None

    # 把 MatSAM 加入 sys.path（兼容直接运行脚本）
    import os
    matsam_dir = os.path.abspath("MatSAM")
    if matsam_dir not in sys.path:
        sys.path.insert(0, matsam_dir)
    try:
        from utils.prompt_generator import PromptGenerator
        from utils.postprocess import PostPrecess
        from utils.segment_anything_ import (
            sam_model_registry, SamAutomaticMaskGenerator,
        )
    except Exception as e:
        print(f"[matsam] 无法导入 MatSAM 模块，降级 baseline: {e}")
        return None

    ckpt = m["checkpoint"]
    if not os.path.exists(ckpt):
        print(f"[matsam] 权重不存在: {ckpt}，降级 baseline")
        return None

    device = m["device"]
    if device == "cuda" and not torch.cuda.is_available():
        print("[matsam] CUDA 不可用，降级 baseline（如需 CPU 推理请把 device 改 cpu）")
        return None

    try:
        sam = sam_model_registry[m["model_type"]](checkpoint=ckpt)
        sam.to(device=device)
    except Exception as e:
        print(f"[matsam] 模型加载失败，降级 baseline: {e}")
        return None

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    prompter = PromptGenerator(
        image_rgb, m["layers"], m["scales"],
        m["n_per_side_base"], m["method_type"],
    )
    points_layers = prompter.generate_prompt_points()

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=None,
        point_grids=points_layers,
        pred_iou_thresh=float(m["pred_iou_thresh"]),
        crop_n_layers=int(m["layers"]),
        crop_n_points_downscale_factor=int(m["scales"]),
        box_nms_thresh=float(m["box_nms_thresh"]),
        crop_nms_thresh=float(m["crop_nms_thresh"]),
        stability_score_thresh=float(m["stability_score_thresh"]),
        points_per_batch=int(m.get("points_per_batch", 64)),
        min_mask_region_area=0,
    )

    try:
        masks = mask_generator.generate(image_rgb)
    except RuntimeError as e:
        # 典型：CUDA out of memory
        print(f"[matsam] 推理失败（可能显存不足），降级 baseline: {e}")
        if device == "cuda":
            torch.cuda.empty_cache()
        return None

    # 按 notebook 流程：对每个 mask Laplacian 取边界，骨架化+膨胀腐蚀连成细线
    area_threshold = int(m["area_threshold"])
    h, w = image_bgr.shape[:2]
    result = np.zeros((h, w), np.uint8)
    for mk in masks:
        tmp = np.uint8(mk["segmentation"].astype(np.uint8) > 0) * 255
        if tmp.sum() == 0:
            continue
        avg_pixel = float(np.average(image_rgb[tmp > 0])) if (tmp > 0).any() else 0.0
        area = int(np.sum(tmp == 255))
        if area <= area_threshold:
            tmp = cv2.Laplacian(tmp, cv2.CV_8U)
            tmp = PostPrecess.remove_small_objects(tmp, min_size=50)
            result = cv2.bitwise_or(result, (tmp > 0).astype(np.uint8) * 255)

    # 同 notebook 的后处理链：骨架->去小->膨胀->腐蚀->去小->骨架->膨胀
    result = PostPrecess.skeletonize(np.uint8(result > 0)) * 255
    result = PostPrecess.remove_small_objects(result, min_size=int(m["min_size"]))
    result = PostPrecess.dilation(np.uint8(result > 0) * 255, square=5)
    result = PostPrecess.erosion(result, square=3)
    result = PostPrecess.remove_small_objects(result, min_size=int(m["min_size"]))
    result = PostPrecess.skeletonize(np.uint8(result > 0)) * 255
    result = PostPrecess.dilation(np.uint8(result > 0) * 255, square=2)
    return (np.uint8(result > 0)) * 255


# ---------------------------------------------------------------- entry
def generate_candidate_boundary(image_bgr: np.ndarray, cfg: dict):
    """返回 (boundary_bin 0/255, method_used: 'matsam'|'baseline')。"""
    boundary = _matsam_boundary(image_bgr, cfg)
    if boundary is not None and boundary.any():
        return boundary, "matsam"
    print("[candidate] 采用 baseline 路径生成候选边界")
    return _baseline_boundary(image_bgr, cfg), "baseline"
