"""支持中文路径的图像读写。cv2.imread/imwrite 在 Windows 上遇到非 ASCII 路径会返回 None，
统一用 np.fromfile + cv2.imdecode / cv2.imencode + tofile 规避。"""
import numpy as np
import cv2


def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    """读取（支持中文路径）任意 flags 同 cv2.imread。"""
    data = np.fromfile(path, np.uint8)
    if data.size == 0:
        raise FileNotFoundError(f"无法读取图像或文件为空: {path}")
    img = cv2.imdecode(data, flags)
    if img is None:
        raise ValueError(f"解码失败，可能不是支持的图像格式: {path}")
    return img


def imwrite_unicode(path, img):
    """写入（支持中文路径）。根据扩展名推断编码格式。"""
    ext = "." + path.rsplit(".", 1)[-1] if "." in path else ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise ValueError(f"编码失败: {path}")
    buf.tofile(path)
