import sys
sys.path.append("./")

import os
import cv2
import MNN
import imutils
import numpy as np

from argparse import ArgumentParser

from utils.utilities import (
    class_names,
    draw_boxes,
    nms,
    preprocess,
    get_vid_properties,
    VideoWriter
)

def build_argparser():
    parser = ArgumentParser()
    parser.add_argument('--weights', type=str, default='weights/yolov5s.mnn', help='model.mnn path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='file/dir/URL, 0 for webcam')
    parser.add_argument('--img-size', '--img', '--imgsz', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--out-dir', type=str, default='results', help='Output dir')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--nodisplay', action='store_true', default=False, help='do not display image, webcam while inference')
    parser.add_argument('--nosave', action='store_true', default=False, help='do not save image')
    opt = parser.parse_args()
    return opt

def detect_image(image, interpreter, session, input_tensor, img_size, conf_thres, iou_thres):
    img = preprocess(image, img_size, img_size)
    tmp_input = MNN.Tensor((1, 3, img_size, img_size), MNN.Halide_Type_Float, img, MNN.Tensor_DimensionType_Caffe,)
    
    input_tensor.copyFrom(tmp_input)
    interpreter.runSession(session)
    # output_ori = interpreter.getSessionOutput(session, "output").getNumpyData()
    # detections = nms(output_ori, conf_thres=conf_thres, iou_thres=iou_thres)
    output_ori = interpreter.getSessionOutput(session, "output")
    output_ori_shape = output_ori.getShape()
    output = MNN.Tensor(output_ori_shape, MNN.Halide_Type_Float, np.zeros_like(output_ori.getData()), MNN.Tensor_DimensionType_Caffe,)
    output_ori.copyToHostTensor(output)
    detections = nms(output.getData(), conf_thres=conf_thres, iou_thres=iou_thres)
    if detections[0] is not None:
        draw_boxes(
            image,
            class_names,
            detections[0],
            input_size=img_size,
            text_bg_alpha=0.6,
        )
    return image

def detect_video(vid,
                interpreter, 
                session, 
                input_tensor, 
                out_dir, 
                img_size, 
                conf_thres,
                iou_thres, 
                video_name, 
                nosave=False, 
                nodisplay=True
                ):
    if not nodisplay:
        winname = "visualizer"
        cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
    
    if not nosave:
        width, height, fps, _ = get_vid_properties(vid)
        vid_writer = VideoWriter(width, height, fps, out_dir, video_name)

    mean_time = 0
    while vid.isOpened():
        current_time = cv2.getTickCount()
        grabbed, frame = vid.read()
        if not grabbed:
            break
        frame = detect_image(frame, interpreter, session, input_tensor, img_size, conf_thres, iou_thres)

        current_time = (cv2.getTickCount() - current_time) / cv2.getTickFrequency()
        if mean_time == 0:
            mean_time = current_time
        else:
            mean_time = mean_time * 0.95 + current_time * 0.05

        cv2.putText(
            frame,
            "FPS: {}".format(int(1 / mean_time * 10) / 10.0),
            (20, 40),
            cv2.FONT_HERSHEY_DUPLEX,
            1,
            (0, 0, 255),
        )

        if not nosave:
            vid_writer.write(frame)

        if not nodisplay:
            frame_show = imutils.resize(frame, width=1280)
            cv2.imshow(winname, frame_show)
            if cv2.waitKey(30) & 0xFF == ord("q"):
                break

    vid.release()
    if not nosave:
        vid_writer.release()
        print(f"Video was saved in {out_dir}")

def main(weights='yolov5s.mnn',  # model.mnn path(s)
        source='inference/images',  # file/dir/URL/glob, 0 for webcam
        img_size=640,  # inference size (pixels)
        out_dir="results", # output result
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        nodisplay=False, # display webcam while inference
        nosave=False, # save image
        ):
    interpreter = MNN.Interpreter(weights)
    session = interpreter.createSession()

    if source.endswith('.jpg') or source.endswith('.png'):
        image = cv2.imread(source)
        input_tensor = interpreter.getSessionInput(session)
        image = detect_image(image, interpreter, session, input_tensor, img_size, conf_thres, iou_thres)
        
        if not nosave: # if save
            path_img_out = f"{out_dir}/{source.split('/')[-1]}"
            cv2.imwrite(path_img_out, image)
            print(f"Image was saved in {out_dir}")
        
        if not nodisplay: # if display
            winname = "visualizer"
            cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
            frame_show = imutils.resize(image, width=1280)
            cv2.imshow(winname, frame_show)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    elif os.path.isdir(source):
        images = os.listdir(source)
        if not nosave:
            output_dir_name = source.split('/')[-1]
            output_dir_name_ = f'{out_dir}/{output_dir_name}'
            if not os.path.exists(output_dir_name_):
                os.mkdir(output_dir_name_)
        for image_file in images:
            if not image_file.endswith('.jpg') and not image_file.endswith('.png') and not image_file.endswith('.jpeg'):
                continue
            image = cv2.imread(os.path.join(source, image_file))
            input_tensor = interpreter.getSessionInput(session)
            image = detect_image(image, interpreter, session, input_tensor, img_size, conf_thres, iou_thres)
            
            if not nosave:
                path_img_out = f"{output_dir_name_}/{image_file}"
                cv2.imwrite(path_img_out, image)
        if not nosave:    
            print(f"All images were saved in {output_dir_name_}")
    
    elif source.endswith('.mp4'):
        video_name = source.split('/')[-1].split('.')[0]
        vid = cv2.VideoCapture(source)
        input_tensor = interpreter.getSessionInput(session)
        detect_video(vid, interpreter, session, input_tensor, out_dir, img_size, conf_thres, iou_thres, video_name, nosave, True)
    
    elif source.isnumeric() or source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://')):
        if source.isnumeric():
            vid = cv2.VideoCapture(0)
        else:
            vid = cv2.VideoCapture(source)
        input_tensor = interpreter.getSessionInput(session)
        detect_video(vid, interpreter, session, input_tensor, out_dir, img_size, conf_thres, iou_thres, "webcam_captured", nosave, False)

if __name__ == "__main__":
    opt = build_argparser()
    sys.exit(main(**vars(opt)) or 0)