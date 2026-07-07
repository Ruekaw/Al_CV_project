import scipy.sparse as sparse
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from skimage import morphology, measure
from skimage.measure import label
from PIL import Image
import time
from natsort import ns, natsorted
import csv
import math
from typing import Tuple


# PostPrecess 已迁移至 postprocess.py；此处保留 re-export 以兼容 `from utils.metrics import PostPrecess`
from .postprocess import PostPrecess


def contingency_table(seg, gt, *, ignore_seg=(), ignore_gt=(), norm=True):
    segr = seg.ravel()
    gtr = gt.ravel()
    ignored = np.zeros(segr.shape, np.bool)
    data = np.ones(gtr.shape)
    for i in ignore_seg:
        ignored[segr == i] = True
    for j in ignore_gt:
        ignored[gtr == j] = True
    data[ignored] = 0
    cont = sparse.coo_matrix((data, (segr, gtr))).tocsr()
    if norm:
        cont /= cont.sum()
    return cont


def rand_values(cont_table):
    n = cont_table.sum()
    sum1 = (cont_table.multiply(cont_table)).sum()
    sum2 = (np.asarray(cont_table.sum(axis=1)) ** 2).sum()
    sum3 = (np.asarray(cont_table.sum(axis=0)) ** 2).sum()
    a = (sum1 - n) / 2.0;
    b = (sum2 - sum1) / 2
    c = (sum3 - sum1) / 2
    d = (sum1 + n ** 2 - sum2 - sum3) / 2
    return a, b, c, d


def adj_rand_index(x, y=None):
    cont = x if y is None else contingency_table(x, y, norm=False)
    a, b, c, d = rand_values(cont)
    nk = a + b + c + d
    return (nk * (a + d) - ((a + b) * (a + c) + (c + d) * (b + d))) / (
            nk ** 2 - ((a + b) * (a + c) + (c + d) * (b + d)))


# calculate grain information
def cal_atom_region_info(img_: np.ndarray):
    res = img_
    np.uint8(img_ > 0)
    label = measure.label(img_, background=1, connectivity=1)
    regions = measure.regionprops(label)

    return regions


class Metric(object):
    def __init__(self):
        pass

    @staticmethod
    def get_iou(pred: np.ndarray, mask: np.ndarray) -> float:
        """
        Referenced by:
        Long J , Shelhamer E , Darrell T . Fully Convolutional Networks for Semantic Segmentation[J].
        IEEE Transactions on Pattern Analysis & Machine Intelligence, 2014, 39(4):640-651.
        """
        class_num = np.amax(mask) + 1

        temp = 0.0
        for i_cl in range(class_num):
            n_ii = np.count_nonzero(mask[pred == i_cl] == i_cl)
            t_i = np.count_nonzero(mask == i_cl)
            temp += n_ii / (t_i + np.count_nonzero(pred == i_cl) - n_ii)
        value = temp / class_num
        return value

    @staticmethod
    def get_dice(pred: np.ndarray, mask: np.ndarray) -> float:
        """
        Dice score
        From now, it is suited to binary segmentation, where 0 is background and 1 is foreground
        """
        intersection = np.count_nonzero(mask[pred == 1] == 1)
        area_sum = np.count_nonzero(mask == 1) + np.count_nonzero(pred == 1)
        value = 2 * intersection / area_sum
        return value

    @staticmethod
    def get_ari(pred: np.ndarray, mask: np.ndarray, bg_value: int = 0) -> float:
        """
        Adjusted rand index
        Implemented by gala (https://github.com/janelia-flyem/gala.)
        """
        label_pred, num_pred = label(pred, connectivity=1, background=bg_value, return_num=True)
        label_mask, num_mask = label(mask, connectivity=1, background=bg_value, return_num=True)
        value = adj_rand_index(label_pred, label_mask)
        return value

    def get_vi(pred: np.ndarray, mask: np.ndarray, bg_value: int = 0, method: int = 1) -> Tuple:
        """
        Referenced by:
        Marina Meilă (2007), Comparing clusterings—an information based distance,
        Journal of Multivariate Analysis, Volume 98, Issue 5, Pages 873-895, ISSN 0047-259X, DOI:10.1016/j.jmva.2006.11.013.
        :param method: 0: skimage implementation and 1: gala implementation (https://github.com/janelia-flyem/gala.)
        :return Tuple = (VI, merger_error, split_error)
        """
        vi, merger_error, split_error = 0.0, 0.0, 0.0

        label_pred, num_pred = label(pred, connectivity=1, background=bg_value, return_num=True)
        label_mask, num_mask = label(mask, connectivity=1, background=bg_value, return_num=True)
        if method == 1:
            # gala (延迟导入：gala 在 Windows 上常装不上；仅 VI 指标需要，核心流程不依赖)
            try:
                import gala.evaluate as ev
            except ImportError:
                # gala 不可用 -> VI 指标降级，返回 NaN，调用方应跳过该指标
                return float('nan'), float('nan'), float('nan')
            merger_error, split_error = ev.split_vi(label_pred, label_mask)
        vi = merger_error + split_error
        if math.isnan(vi):
            return 10, 5, 5
        return merger_error, split_error, vi

    def get_F1(pred: np.ndarray, mask: np.ndarray):  # mask, pred
        # bool type for calculatel
        mask = mask.astype(bool)
        pred = pred.astype(bool)

        # calculate the number of true cases, false positive cases, and false negative cases
        true_positive = np.logical_and(mask, pred).sum()
        false_positive = np.logical_and(~mask, pred).sum()
        false_negative = np.logical_and(mask, ~pred).sum()

        # calculate accuracy and recall
        precision = true_positive / (true_positive + false_positive + 1e-10)
        recall = true_positive / (true_positive + false_negative + 1e-10)

        # calculate F1 metrics
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-10)

        return f1_score

    def get_recall(pred: np.ndarray, mask: np.ndarray):
        pred = pred.astype(bool)
        mask = mask.astype(bool)

        true_positive = np.logical_and(mask, pred).sum()
        false_negative = np.logical_and(mask, ~pred).sum()

        recall = true_positive / (true_positive + false_negative + 1e-10)

        return recall
