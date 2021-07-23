import sys

sys.path.append("./")

from argparse import ArgumentParser

import cv2
import MNN
import numpy as np

from utils.utilities_np import (
    class_names,
    draw_boxes,
    nms,
    num_classes,
    preprocess,
)


def build_argparser():
    parser = ArgumentParser(prog="demo_mmn_img.py")
    parser.add_argument(
        "--model-path",
        type=str,
        default="/home/anhpc03/Documents/yolov5_product/yolov5/weights/mnn/yolov5xs_160_GenProduct_v4.mnn",
        help="model path",
    )
    parser.add_argument(
        "--image-path",
        default="",
        nargs="*",
        help="image path",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="../inference/output_mnn",
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
    parser.add_argument("--display", action="store_true", help="display image")
    parser.add_argument("--save", action="store_true", default=True, help="save image")

    return parser


def main(args):
    interpreter = MNN.Interpreter(args.model_path)
    session = interpreter.createSession()

    for image in args.image_path:
        print(image)
        img_brg = cv2.imread(image)
        input_tensor = interpreter.getSessionInput(session)

        img = preprocess(img_brg, args.img_size, args.img_size)

        tmp_input = MNN.Tensor(
            (1, 3, args.img_size, args.img_size),
            MNN.Halide_Type_Float,
            img,
            MNN.Tensor_DimensionType_Caffe,
        )

        input_tensor.copyFrom(tmp_input)
        interpreter.runSession(session)
        output = interpreter.getSessionOutput(session, "output").getData()

        detections = nms(output, conf_thres=args.conf_thres, iou_thres=args.iou_thres)

        if detections[0] is not None:
            draw_boxes(
                img_brg,
                class_names,
                detections[0],
                input_size=args.img_size,
                text_bg_alpha=0.6,
            )
        
        if args.save:
            path_img_out = f"{args.out_dir}/{image.split('/')[-1]}"
            cv2.imwrite(path_img_out, img_brg)
            print("Done")

if __name__ == "__main__":
    args = build_argparser().parse_args()

    sys.exit(main(args) or 0)
