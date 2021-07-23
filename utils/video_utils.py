import os
import cv2


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