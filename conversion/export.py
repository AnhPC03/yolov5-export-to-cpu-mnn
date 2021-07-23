"""Exports a YOLOv5 *.pt model to ONNX formats

Usage:
    $ export PYTHONPATH="$PWD" && python models/export.py --weights ./weights/yolov5s.pt --img-size 640 --batch-size 1 weights/onnx
"""

import os
import argparse
import torch

from utils.google_utils import attempt_download

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        type=str,
        default="weights/pytorch/yolov5xxs_320_face.pt",
        help="weights path",
    )
    parser.add_argument(
        "--img-size", nargs="+", type=int, default=640, help="image size"
    )
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument(
        "--out-dir", type=str, default="weights/onnx", help="output directory"
    )

    opt = parser.parse_args()
    opt.img_size *= 2 if len(opt.img_size) == 1 else 1  # expand
    print(opt)

    # Input
    img = torch.zeros(
        (opt.batch_size, 3, *opt.img_size)
    )

    # Load PyTorch model
    attempt_download(opt.weights)
    model = torch.load(opt.weights, map_location=torch.device("cpu"))["model"].float()
    # prune(model, 0.2)
    model.eval()
    model.model[-1].export = True  # set Detect() layer export=True

    y = model(img)  # dry run

    # ONNX export
    try:
        import onnx

        print("\nStarting ONNX export with onnx %s..." % onnx.__version__)
        # f = opt.weights.replace(".pt", ".onnx")  # filename
        bname = os.path.basename(opt.weights).replace(".pt", ".onnx")
        f = os.path.join(opt.out_dir, bname)

        model.fuse()  # only for ONNX

        torch.onnx.export(
            model,
            img,
            f,
            verbose=True,
            opset_version=10,
            input_names=["images"],
            output_names=["classes", "boxes"] if y is None else ["output"],
        )

        # Checks
        onnx_model = onnx.load(f)  # load onnx model
        onnx.checker.check_model(onnx_model)  # check onnx model
        print(
            onnx.helper.printable_graph(onnx_model.graph)
        )  # print a human readable model
        print("ONNX export success, saved as %s" % f)
    except Exception as e:
        print("ONNX export failure: %s" % e)

    # Finish
    print("\nExport complete. Visualize with https://github.com/lutzroeder/netron.")
