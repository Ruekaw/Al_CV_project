"""4.1 预处理与尺度标定。

输入：原图 BGR
操作：灰度 -> CLAHE -> 中值滤波 -> 双边滤波
输出：增强灰度图 + um_per_pixel（一期 None=像素单位）
"""
import cv2
import numpy as np


def preprocess(image_bgr: np.ndarray, cfg: dict):
    p = cfg["preprocess"]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # CLAHE 增强晶界对比度
    clahe = cv2.createCLAHE(
        clipLimit=p["clahe_clip_limit"],
        tileGridSize=tuple(p["clahe_grid"]),
    )
    clahe_img = clahe.apply(gray)

    # 中值滤波去椒盐
    k = int(p["median_kernel"])
    if k > 1:
        med = cv2.medianBlur(clahe_img, k if k % 2 == 1 else k + 1)
    else:
        med = clahe_img

    # 双边滤波保边平滑
    d = int(p["bilateral_d"])
    enhanced = cv2.bilateralFilter(med, d=d, sigmaColor=50, sigmaSpace=50)

    um_per_pixel = cfg["scale"]["um_per_pixel"]
    return enhanced, um_per_pixel
