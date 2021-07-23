"""
https://deeplizard.com/learn/video/kF2AlpykJGY
"""

import os
import time

import cv2
import numpy as np


# class names
class_names = ["product"]
num_classes = len(class_names)


def preprocess(image_src, input_height, input_width, nchw_shape=True):
    in_image_src, _, _ = resize(image_src, (input_width, input_height))
    in_image_src = in_image_src[..., ::-1]
    if nchw_shape:
        in_image_src = in_image_src.transpose(
            (2, 0, 1)
        )  # Change data layout from HWC to CHW
    # in_image_src = np.ascontiguousarray(in_image_src)
    in_image_src = np.expand_dims(in_image_src, axis=0)

    img_in = in_image_src / 255.0
    img_in = img_in.astype(np.float32)
    return img_in


def resize(image, target_shape=(480, 480), interpolation=cv2.INTER_CUBIC):
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


def nms(prediction, conf_thres=0.3, iou_thres=0.5):
    """Performs Non-Maximum Suppression (NMS) on inference results
    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    max_det = 20  # maximum number of detections per image
    time_limit = 3.0  # seconds to quit after

    t = time.time()
    output = [None] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        x = x[xc[xi]]  # confidence
        # If none remain process next image
        if not x.shape[0]:
            continue
        # Compute conf

        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        # conf = np.amax(x[:, 5:], axis=1, keepdims=True)
        conf = x[:, 5:]
        j = np.zeros_like(conf)
        x = np.concatenate((box, conf, j), 1)
        x = x[x[:, 4] > conf_thres]

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        i = py_cpu_nms(dets=x[:, :4], scores=x[:, 4], thresh=iou_thres)
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded
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


def argmax(*args, keepdims=False, **kwargs):
    res = np.argmax(*args, **kwargs)
    axis = kwargs.get("axis", None)
    if axis is not None:
        res = np.expand_dims(res, axis=axis)
    return res


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def draw_boxes(
    image_src,
    class_names,
    detections,
    input_size=640,
    line_thickness=None,
    text_bg_alpha=0.6,
):
    labels = detections[..., -1]
    boxs = detections[..., :4]
    confs = detections[..., 4]

    h, w, _ = image_src.shape

    boxs[:, :] = scale_coords((input_size, input_size), boxs[:, :], (h, w)).round()

    tl = line_thickness or round(0.002 * (w + h) / 2) + 1
    for i, box in enumerate(boxs):
        box = map(int, box)
        x1, y1, x2, y2 = box
        np.random.seed(int(labels[i]) + 2020)
        color = [np.random.randint(0, 255), 0, np.random.randint(0, 255)]
        cv2.rectangle(
            image_src,
            (x1, y1),
            (x2, y2),
            color,
            thickness=max(int((w + h) / 600), 1),
            lineType=cv2.LINE_AA,
        )
        label = "%s %.2f" % (class_names[int(labels[i])], confs[i])
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=1)[0]
        c2 = x1 + t_size[0] + 3, y1 - t_size[1] - 5
        if text_bg_alpha == 0.0:
            cv2.rectangle(image_src, (x1 - 1, y1), c2, color, cv2.FILLED, cv2.LINE_AA)
        else:
            alphaReserve = text_bg_alpha
            BChannel, GChannel, RChannel = color
            xMin, yMin = int(x1 - 1), int(y1 - t_size[1] - 3)
            xMax, yMax = int(x1 + t_size[0]), int(y1)
            image_src[yMin:yMax, xMin:xMax, 0] = image_src[
                yMin:yMax, xMin:xMax, 0
            ] * alphaReserve + BChannel * (1 - alphaReserve)
            image_src[yMin:yMax, xMin:xMax, 1] = image_src[
                yMin:yMax, xMin:xMax, 1
            ] * alphaReserve + GChannel * (1 - alphaReserve)
            image_src[yMin:yMax, xMin:xMax, 2] = image_src[
                yMin:yMax, xMin:xMax, 2
            ] * alphaReserve + RChannel * (1 - alphaReserve)

        cv2.putText(
            image_src,
            label,
            (x1 + 3, y1 - 4),
            0,
            tl / 3,
            [255, 255, 255],
            thickness=1,
            lineType=cv2.LINE_AA,
        )


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(
            img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1]
        )  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (
            img1_shape[0] - img0_shape[0] * gain
        ) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clip(0, img_shape[1])  # x1
    boxes[:, 1].clip(0, img_shape[0])  # y1
    boxes[:, 2].clip(0, img_shape[1])  # x2
    boxes[:, 3].clip(0, img_shape[0])  # y2


def get_vid_properties(vid):
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = vid.get(cv2.CAP_PROP_FPS)
    num_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    return width, height, fps, num_frames


class VideoWriter:
    def __init__(self, width, height, fps, save_path, basename):
        output_fname = os.path.join(save_path, basename)
        output_fname = os.path.splitext(output_fname)[0] + "_inferenced.mp4"
        print(f"file name is {output_fname}")
        self.output_file = cv2.VideoWriter(
            filename=output_fname,
            fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
            fps=float(fps),
            frameSize=(width, height),
            isColor=True,
        )

    def write(self, frame):
        self.output_file.write(frame)

    def release(self):
        self.output_file.release()