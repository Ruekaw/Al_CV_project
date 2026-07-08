import argparse
import csv
import itertools
import json
import os
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.ndimage import convolve
from skimage import morphology

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.boundary_clean import clean_boundary  # noqa: E402
from src.boundary_repair import repair_boundary  # noqa: E402
from src.image_io import imread_unicode, imwrite_unicode  # noqa: E402
from src.intercept import run_intercept  # noqa: E402
from src.preprocess import preprocess  # noqa: E402
from src.segment_features import extract_segments  # noqa: E402
from src.viz import boundary_overlay, intercept_overlay, make_report  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)


@dataclass
class MaskEdge:
    y: np.ndarray
    x: np.ndarray
    area: int
    bbox: tuple
    pred_iou: float
    stability: float
    aspect: float
    extent: float


def parse_values(raw, cast=float):
    return [cast(v.strip()) for v in raw.split(",") if v.strip()]


def parse_roi(raw):
    if not raw:
        return None
    vals = parse_values(raw, int)
    if len(vals) != 4:
        raise ValueError("--roi must be x,y,w,h")
    x, y, w, h = vals
    if w <= 0 or h <= 0:
        raise ValueError("--roi width and height must be positive")
    return x, y, w, h


def crop_roi(arr, roi):
    if roi is None:
        return arr
    x, y, w, h = roi
    return arr[y:y + h, x:x + w]


def list_images(input_dir, names=None, limit=None):
    files = sorted(
        p for p in Path(input_dir).iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    )
    if names:
        wanted = set(names)
        files = [p for p in files if p.stem in wanted or p.name in wanted]
    if limit:
        files = files[:limit]
    return files


def to_matsam_rgb(image):
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.ndim == 3 and image.shape[2] == 1:
        return cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def mask_shape_stats(mask_info, area):
    bbox = mask_info.get("bbox", [0, 0, 0, 0])
    if not bbox or len(bbox) < 4:
        return (0, 0, 0, 0), 1.0, 1.0
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return tuple(bbox), 1.0, 1.0
    aspect = max(w, h) / max(min(w, h), 1)
    extent = area / max(w * h, 1)
    return (x, y, w, h), float(aspect), float(extent)


def generate_mask_pool(image_rgb, cfg, pred_iou, stability, points_per_batch=None):
    import torch

    matsam_dir = ROOT / "MatSAM"
    if str(matsam_dir) not in sys.path:
        sys.path.insert(0, str(matsam_dir))

    from utils.postprocess import PostPrecess
    from utils.prompt_generator import PromptGenerator
    from utils.segment_anything_ import SamAutomaticMaskGenerator, sam_model_registry

    m = cfg["matsam"]
    sam = sam_model_registry[m["model_type"]](checkpoint=m["checkpoint"])
    sam.to(device=m["device"])

    prompter = PromptGenerator(
        image_rgb,
        m["layers"],
        m["scales"],
        m["n_per_side_base"],
        m["method_type"],
    )
    point_layers = prompter.generate_prompt_points()

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=None,
        point_grids=point_layers,
        pred_iou_thresh=float(pred_iou),
        crop_n_layers=int(m["layers"]),
        crop_n_points_downscale_factor=int(m["scales"]),
        box_nms_thresh=float(m["box_nms_thresh"]),
        crop_nms_thresh=float(m["crop_nms_thresh"]),
        stability_score_thresh=float(stability),
        points_per_batch=int(points_per_batch or m.get("points_per_batch", 64)),
        min_mask_region_area=0,
    )

    with torch.no_grad():
        masks = mask_generator.generate(image_rgb)

    pool = []
    for mk in masks:
        seg = np.uint8(mk["segmentation"].astype(np.uint8) > 0) * 255
        area = int(mk.get("area", np.count_nonzero(seg)))
        edge = cv2.Laplacian(seg, cv2.CV_8U)
        edge = PostPrecess.remove_small_objects(edge, min_size=50)
        yy, xx = np.nonzero(edge)
        if len(yy) == 0:
            continue
        bbox, aspect, extent = mask_shape_stats(mk, area)
        pool.append(
            MaskEdge(
                y=yy.astype(np.uint16),
                x=xx.astype(np.uint16),
                area=area,
                bbox=tuple(float(v) for v in bbox),
                pred_iou=float(mk.get("predicted_iou", 1.0)),
                stability=float(mk.get("stability_score", 1.0)),
                aspect=aspect,
                extent=extent,
            )
        )
    return pool


def build_candidate(mask_pool, shape, params, cfg):
    m = cfg["matsam"]
    max_area = int(params["area_threshold"])
    min_area = int(params["min_area_threshold"])
    max_aspect = float(params["max_mask_aspect_ratio"])
    min_extent = float(params["min_mask_extent"])
    result = np.zeros(shape, np.uint8)
    selected = 0

    for mk in mask_pool:
        if mk.pred_iou < params["pred_iou_thresh"]:
            continue
        if mk.stability < params["stability_score_thresh"]:
            continue
        if mk.area < min_area or mk.area > max_area:
            continue
        if max_aspect > 0 and min_extent > 0:
            if mk.aspect >= max_aspect and mk.extent <= min_extent:
                continue
        result[mk.y, mk.x] = 255
        selected += 1

    result = morphology.skeletonize(result > 0, method="lee")
    result = morphology.remove_small_objects(
        result, min_size=int(m.get("min_size", 200)), connectivity=2,
    )
    result = morphology.dilation(result, morphology.square(5))
    result = morphology.erosion(result, morphology.square(3))
    result = morphology.remove_small_objects(
        result > 0, min_size=int(m.get("min_size", 200)), connectivity=2,
    )
    result = morphology.skeletonize(result, method="lee")
    result = morphology.dilation(result, morphology.square(2))
    return np.uint8(result > 0) * 255, selected


def make_reference_skeleton(enhanced, percentile, min_size):
    gray = enhanced if enhanced.ndim == 2 else cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    threshold = np.percentile(gray, percentile)
    dark = gray <= threshold
    dark = morphology.remove_small_objects(dark, min_size=min_size, connectivity=2)
    dark = morphology.binary_closing(dark, morphology.square(3))
    return morphology.skeletonize(dark, method="lee")


def endpoint_count(binary):
    skel = morphology.skeletonize(binary > 0, method="lee")
    kernel = np.ones((3, 3), dtype=np.int16)
    neighbors = convolve(skel.astype(np.int16), kernel, mode="constant", cval=0) - skel
    return int(np.count_nonzero(skel & (neighbors == 1))), int(np.count_nonzero(skel))


def score_result(candidate, final, ref_skel, clean_info, repair_info, ref_tol):
    final_skel = morphology.skeletonize(final > 0, method="lee")
    ref_band = morphology.binary_dilation(ref_skel, morphology.square(2 * ref_tol + 1))
    final_band = morphology.binary_dilation(final_skel, morphology.square(2 * ref_tol + 1))

    ref_count = max(int(np.count_nonzero(ref_skel)), 1)
    final_count = max(int(np.count_nonzero(final_skel)), 1)
    recall = np.count_nonzero(ref_skel & final_band) / ref_count
    precision = np.count_nonzero(final_skel & ref_band) / final_count
    density_ratio = final_count / ref_count
    density_penalty = abs(np.log2(max(density_ratio, 1e-6)))

    deleted_ratio = clean_info["n_deleted"] / max(clean_info["n_total"], 1)
    endpoints, skel_count = endpoint_count(final)
    endpoint_per_1000 = endpoints / max(skel_count / 1000, 1)

    score = (
        1.40 * recall
        + 0.90 * precision
        - 0.30 * deleted_ratio
        - 0.04 * endpoint_per_1000
        - 0.12 * density_penalty
    )
    return {
        "score": round(float(score), 6),
        "recall": round(float(recall), 6),
        "precision": round(float(precision), 6),
        "density_ratio": round(float(density_ratio), 6),
        "deleted_ratio": round(float(deleted_ratio), 6),
        "endpoints": int(endpoints),
        "endpoint_per_1000": round(float(endpoint_per_1000), 6),
        "n_total": int(clean_info["n_total"]),
        "n_deleted": int(clean_info["n_deleted"]),
        "n_repaired": int(repair_info["n_repaired"]),
    }


def evaluate_params(params, mask_pool, image_bgr, enhanced, cfg, ref_skel, ref_tol, roi=None):
    candidate, selected_masks = build_candidate(mask_pool, enhanced.shape[:2], params, cfg)
    segments, junction = extract_segments(candidate, cfg)
    cleaned, deleted, clean_info = clean_boundary(candidate, segments, cfg, junction=junction)
    final, added, repair_info = repair_boundary(cleaned, cfg)
    metrics = score_result(
        crop_roi(candidate, roi),
        crop_roi(final, roi),
        crop_roi(ref_skel, roi),
        clean_info,
        repair_info,
        ref_tol,
    )
    metrics["selected_masks"] = selected_masks
    return metrics, candidate, cleaned, final, deleted, added


def save_visuals(out_dir, rank, image_bgr, enhanced, candidate, cleaned, final, deleted, added, cfg):
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"rank_{rank:02d}"
    output_cfg = cfg.get("output", {})
    overlay_line_width = int(output_cfg.get("overlay_line_width", 3))
    intercept_point_radius = int(output_cfg.get("intercept_point_radius", 4))
    imwrite_unicode(str(out_dir / f"{tag}_candidate.png"), candidate)
    imwrite_unicode(
        str(out_dir / f"{tag}_cleaned_overlay.png"),
        boundary_overlay(
            image_bgr, cleaned, deleted=deleted,
            line_width=overlay_line_width,
        ),
    )
    imwrite_unicode(
        str(out_dir / f"{tag}_final_overlay.png"),
        boundary_overlay(
            image_bgr, final, deleted=deleted, added=added,
            line_width=overlay_line_width,
        ),
    )
    intercept_res, lines_viz = run_intercept(final, cfg, return_lines=True)
    report_path = out_dir / f"{tag}_report.png"
    make_report(
        [
            image_bgr,
            enhanced,
            candidate,
            boundary_overlay(
                image_bgr, cleaned, deleted=deleted,
                line_width=overlay_line_width,
            ),
            boundary_overlay(
                image_bgr, final, deleted=deleted, added=added,
                line_width=overlay_line_width,
            ),
            intercept_overlay(
                image_bgr, final, lines_viz,
                line_width=overlay_line_width,
                point_radius=intercept_point_radius,
            ),
        ],
        ["original", "enhanced", "candidate", "cleaned", "final", "intercept"],
        str(report_path),
    )
    return intercept_res


def param_grid(args):
    keys = [
        ("pred_iou_thresh", parse_values(args.pred_iou)),
        ("stability_score_thresh", parse_values(args.stability)),
        ("area_threshold", parse_values(args.area_threshold, int)),
        ("min_area_threshold", parse_values(args.min_area_threshold, int)),
        ("max_mask_aspect_ratio", parse_values(args.max_mask_aspect_ratio)),
        ("min_mask_extent", parse_values(args.min_mask_extent)),
    ]
    combos = [dict(zip([k for k, _ in keys], values)) for values in itertools.product(*[v for _, v in keys])]
    if args.max_trials and len(combos) > args.max_trials:
        indices = np.linspace(0, len(combos) - 1, args.max_trials, dtype=int)
        combos = [combos[i] for i in indices]
    return combos


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--input", default="pretrain_sets")
    ap.add_argument("--runs", default="runs_tune_matsam")
    ap.add_argument("--names", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=1)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--max-trials", type=int, default=32)
    ap.add_argument("--points-per-batch", type=int, default=None)
    ap.add_argument("--pred-iou", default="0.84,0.86,0.88")
    ap.add_argument("--stability", default="0.89,0.91,0.93")
    ap.add_argument("--area-threshold", default="90000,120000,150000")
    ap.add_argument("--min-area-threshold", default="3000,5000,7000")
    ap.add_argument("--max-mask-aspect-ratio", default="6,8")
    ap.add_argument("--min-mask-extent", default="0.10,0.12")
    ap.add_argument("--reference-percentile", type=float, default=15.0)
    ap.add_argument("--reference-min-size", type=int, default=80)
    ap.add_argument("--reference-tol", type=int, default=3)
    ap.add_argument("--roi", default=None, help="Optional scoring ROI: x,y,w,h")
    args = ap.parse_args()

    os.chdir(ROOT)
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    images = list_images(args.input, args.names, args.limit)
    if not images:
        raise SystemExit(f"No images found in {args.input}")

    out_root = Path(args.runs)
    out_root.mkdir(parents=True, exist_ok=True)
    combos = param_grid(args)
    roi = parse_roi(args.roi)
    all_rows = []

    for image_path in images:
        t0 = time.time()
        image_bgr = imread_unicode(str(image_path))
        enhanced, _ = preprocess(image_bgr, cfg)
        image_rgb = to_matsam_rgb(enhanced)
        ref_skel = make_reference_skeleton(
            enhanced,
            percentile=args.reference_percentile,
            min_size=args.reference_min_size,
        )

        pool_pred = min(c["pred_iou_thresh"] for c in combos)
        pool_stability = min(c["stability_score_thresh"] for c in combos)
        print(
            f"[{image_path.name}] generating mask pool "
            f"pred>={pool_pred} stability>={pool_stability}",
            flush=True,
        )
        try:
            mask_pool = generate_mask_pool(
                image_rgb,
                cfg,
                pool_pred,
                pool_stability,
                points_per_batch=args.points_per_batch,
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                raise SystemExit(
                    "CUDA OOM while generating the mask pool. Try higher "
                    "--pred-iou/--stability lower bounds, or set "
                    "--points-per-batch 32/16."
                ) from e
            raise
        print(f"[{image_path.name}] mask pool: {len(mask_pool)} masks", flush=True)

        image_rows = []
        for idx, params in enumerate(combos, 1):
            metrics, *_ = evaluate_params(
                params, mask_pool, image_bgr, enhanced, cfg, ref_skel, args.reference_tol, roi=roi,
            )
            row = {
                "image": image_path.stem,
                "trial": idx,
                "roi": args.roi or "",
                **params,
                **metrics,
            }
            image_rows.append(row)
            all_rows.append(row)
            print(
                f"  trial {idx:03d}/{len(combos)} "
                f"score={metrics['score']} recall={metrics['recall']} "
                f"precision={metrics['precision']} selected={metrics['selected_masks']}",
                flush=True,
            )

        image_rows.sort(key=lambda r: r["score"], reverse=True)
        image_out = out_root / image_path.stem
        image_out.mkdir(parents=True, exist_ok=True)
        with open(image_out / "trials.json", "w", encoding="utf-8") as f:
            json.dump(image_rows, f, ensure_ascii=False, indent=2)
        write_csv(image_out / "trials.csv", image_rows)

        top_rows = image_rows[: args.top_k]
        for rank, row in enumerate(top_rows, 1):
            params = {k: row[k] for k in (
                "pred_iou_thresh",
                "stability_score_thresh",
                "area_threshold",
                "min_area_threshold",
                "max_mask_aspect_ratio",
                "min_mask_extent",
            )}
            _, candidate, cleaned, final, deleted, added = evaluate_params(
                params, mask_pool, image_bgr, enhanced, cfg, ref_skel, args.reference_tol, roi=roi,
            )
            intercept = save_visuals(
                image_out, rank, image_bgr, enhanced, candidate, cleaned, final, deleted, added, cfg,
            )
            row["top_report"] = (image_out / f"rank_{rank:02d}_report.png").as_posix()
            row["top_candidate"] = (image_out / f"rank_{rank:02d}_candidate.png").as_posix()
            row["intercept_d_px"] = intercept.get("d_px")

        with open(image_out / "top.json", "w", encoding="utf-8") as f:
            json.dump(top_rows, f, ensure_ascii=False, indent=2)

        print(
            f"[{image_path.name}] done in {round(time.time() - t0, 1)}s. "
            f"best score={top_rows[0]['score'] if top_rows else None}",
            flush=True,
        )

    all_rows.sort(key=lambda r: r["score"], reverse=True)
    with open(out_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(all_rows, f, ensure_ascii=False, indent=2)
    write_csv(out_root / "summary.csv", all_rows)


if __name__ == "__main__":
    main()
