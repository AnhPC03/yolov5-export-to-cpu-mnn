import sys

sys.path.append("./")

from argparse import ArgumentParser
import time

import cv2
import MNN
import numpy as np
import imutils

from utils.utilities_prj14 import (
    class_names,
    scale_coord,
    nms,
    preprocess,
    calc_hist,
    isBbackground,
    isCenter,
)
from utils.video_utils import VideoWriter, get_vid_properties


def build_argparser():
    parser = ArgumentParser(prog="demo_mmn_video.py")
    parser.add_argument(
        "--model-path",
        type=str,
        default="../weights/mnn/yolov5xs_192_GenProduct_v3.mnn",
        help="model path",
    )
    parser.add_argument(
        "--video-path",
        type=str,
        default="inference/videos/VID_20201119_173806.mp4",
        help="video path",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="../inference/output_mnn_video",
        help="Output dir",
    )
    parser.add_argument(
        "--img-size", type=int, default=640, help="inference size (pixels)"
    )
    parser.add_argument(
        "--conf-thres", type=float, default=0.25, help="object confidence threshold"
    )
    parser.add_argument(
        "--iou-thres", type=float, default=0.45, help="IOU threshold for NMS"
    )
    parser.add_argument("--display", action="store_true", default=True, help="display image")
    parser.add_argument("--save", action="store_true", default=True, help="display image")

    return parser


def main(args):
    print(f"model: {args.model_path}")
    print(f"video: {args.video_path}")

    interpreter = MNN.Interpreter(args.model_path)
    session = interpreter.createSession()
    input_tensor = interpreter.getSessionInput(session)
    
    if args.display:
        winname = "visualizer"
        cv2.namedWindow(winname, cv2.WINDOW_NORMAL)

    vid = cv2.VideoCapture(0)
    # vid = cv2.VideoCapture('rtsp://admin:12345@192.168.3.22/live')
    # vid.set(cv2.CAP_PROP_FRAME_WIDTH, 900)
    # vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 900)

    isGetVidProperties = False
    num_consec = 0
    while vid.isOpened():
        grabbed, frame = vid.read()
        if not grabbed:
            break
        if not isGetVidProperties:
            isGetVidProperties = True
            if args.save:
                h, w, _ = frame.shape
                vid_writer = VideoWriter(w, h, 30, args.out_dir, "vid_01.mp4")

        org_frame = frame.copy()
        frame_in = preprocess(frame, args.img_size, args.img_size)

        tmp_input = MNN.Tensor(
            (1, 3, args.img_size, args.img_size),
            MNN.Halide_Type_Float,
            frame_in,
            MNN.Tensor_DimensionType_Caffe,
        )

        input_tensor.copyFrom(tmp_input)
        interpreter.runSession(session)
        output = interpreter.getSessionOutput(session, "output").getData()

        dets = nms(output, conf_thres=args.conf_thres, iou_thres=args.iou_thres)

        if dets is not None:
            num_consec += 1
        else:
            num_consec = 0

        if (
            num_consec >= 1
            and isCenter(dets, (args.img_size, args.img_size)) is not None
        ):
            det = isCenter(dets, (args.img_size, args.img_size))
            box = scale_coord((args.img_size, args.img_size), det, frame.shape)

            x1, y1, x2, y2, conf = box[:5]
            product = org_frame[y1:y2, x1:x2]

            cv2.rectangle(org_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(
                org_frame,
                "conf: {}".format(round(conf, 2)),
                (x1, y1),
                cv2.FONT_HERSHEY_DUPLEX,
                3,
                (0, 0, 255),
            )

        if args.display:
            frame_show = imutils.resize(org_frame, width=1280)
            cv2.imshow(winname, frame_show)
            if cv2.waitKey(30) & 0xFF == ord("q"):
                break

        if args.save:
            vid_writer.write(org_frame)

    vid.release()
    if args.save:
        vid_writer.release()
    

if __name__ == "__main__":
    args = build_argparser().parse_args()

    sys.exit(main(args) or 0)
