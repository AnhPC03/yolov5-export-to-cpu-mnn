import os
import time

import cv2
import numpy as np

class_names = ["product"]


def preprocess(image_src, input_height, input_width, nchw_shape=True):
    in_image_src, _, _ = resize(image_src, (input_width, input_height))
    # BRG -> RGB
    in_image_src = in_image_src[..., ::-1]
    if nchw_shape:
        in_image_src = in_image_src.transpose(
            (2, 0, 1)
        )  # Change data layout from HWC to CHW

    in_image_src = np.expand_dims(in_image_src, axis=0)

    img_in = in_image_src / 255.0
    img_in = img_in.astype(np.float32)

    return img_in


def resize(image, target_shape=(320, 320), interpolation=cv2.INTER_CUBIC):
    shape = image.shape[0:2][::-1]
    iw, ih = shape
    # n, c, h, w = self.input_shape
    w, h = target_shape
    scale = min(w / iw, h / ih)
    ratio = scale, scale
    nw = int(round(iw * scale))
    nh = int(round(ih * scale))
    new_unpad = nw, nh

    if shape[0:2] != new_unpad:
        image = cv2.resize(image, (nw, nh), interpolation=interpolation)

    new_image = np.full((h, w, 3), 128)
    dx = (w - nw) // 2
    dy = (h - nh) // 2
    new_image[dy : dy + nh, dx : dx + nw, :] = image

    return new_image, ratio, (dx, dy)


def nms(predictions, conf_thres=0.1, iou_thres=0.6):
    """Performs Non-Maximum Suppression (NMS) on inference results

    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """
    output = None
    max_det = 10  # maximum number of detections per image

    # if batch_size = 1. predictions.shape[0] = 1, predictions.shape = (1, 6300, 6)
    prediction = predictions[0]

    xc = prediction[..., 4] > conf_thres  # idx of candidates

    # get only bboxes having obj_conf higher than "conf_thres"
    x = prediction[xc]
    if not x.shape[0]:
        return output

    # Compute conf
    x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
    # Box (center x, center y, width, height) to (x1, y1, x2, y2)
    box = xywh2xyxy(x[:, :4])

    # box.shape = (N, 4), x[:, 5:].shape = (N, 1)
    x = np.concatenate((box, x[:, 5:]), 1)
    # x.shape = (N, 5): N x [x1, y1, x2, y2, conf]
    x = x[x[:, 4] > conf_thres]
    if not x.shape[0]:
        return output

    i = py_cpu_nms(x[:, :4], x[:, 4], iou_thres)

    if i.shape[0] > max_det:  # limit detections
        i = i[:max_det]

    output = x[i]
    return output


def py_cpu_nms(dets, scores, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    keep = np.array(keep)
    return keep


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def scale_coord(img1_shape, box, img0_shape):
    # Rescale coord (xyxy) from img1_shape to img0_shape
    gain = min(
        img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1]
    )  # gain  = old / new
    pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (
        img1_shape[0] - img0_shape[0] * gain
    ) / 2  # wh padding

    box[0] -= pad[0]  # x padding
    box[2] -= pad[0]  # x padding
    box[1] -= pad[1]  # y padding
    box[3] -= pad[1]  # y padding
    box[:4] = [b / gain for b in box[:4]]

    box = clip_box(box, img0_shape)

    return box


def clip_box(box, img_shape):
    for i, x in enumerate(box):
        if x < 0:
            box[i] = 0

    box[0] = min(box[0], img_shape[1])
    box[1] = min(box[1], img_shape[0])
    box[2] = min(box[2], img_shape[1])
    box[3] = min(box[3], img_shape[0])
    box[:4] = list(map(int, box[:4]))

    return box


# https://docs.opencv.org/3.4/d8/dc8/tutorial_histogram_comparison.html
def calc_hist(img_bgr, h_bins=50, s_bins=60):
    hist_size = [h_bins, s_bins]
    h_ranges = [0, 180]
    s_ranges = [0, 256]
    ranges = h_ranges + s_ranges
    channels = [0, 1]

    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    img_hist = cv2.calcHist(
        [img_hsv], channels, None, hist_size, ranges, accumulate=False
    )
    cv2.normalize(img_hist, img_hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    return img_hist


# https://docs.opencv.org/3.4/d8/dc8/tutorial_histogram_comparison.html
def isBbackground(product_hist, bg_hists, bg_thresh=0.35, compare_method=0):
    isBg = False
    for bg_hist in bg_hists:
        product_bg = cv2.compareHist(product_hist, bg_hist, compare_method)
        if product_bg > bg_thresh:
            isBg = True
            break

    return isBg


def isCenter(box_info, infer_shape):
    H, W = infer_shape
    min_dim = min(infer_shape)
    cx = W // 2
    cy = H // 2

    c = np.array([cx, cy])

    boxes = xyxy2xywh(box_info[:, :4])
    vec = boxes[:, :2] - c
    dists = np.linalg.norm(vec, axis=1)
    out_box_info = np.concatenate((box_info, dists.reshape(-1, 1)), 1)
    # sort according to score: https://stackoverflow.com/questions/2828059/sorting-arrays-in-numpy-by-column
    out_box_info[out_box_info[:, 4].argsort()[::-1]]
    # get the highest score box
    out = out_box_info[0]
    dist = out[-1]
    if dist > min_dim / 5:
        return None
    return out[:-1].tolist()