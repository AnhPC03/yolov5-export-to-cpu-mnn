# Export YOLOv5 to run on Intel/AMD CPU
Export processing consists of two steps:<br />
1️⃣ Convert Pytorch model weights to MNN model weights.<br />
2️⃣ Run the inference on Intel/AMD CPU.<br />

## Documentation
ℹ️ MNN is a lightweight deep neural network inference engine.<br />
🔎 You can find more information about MNN in [here](https://www.yuque.com/mnn/en/about).<br />
🔎 MNN github repository in [here](https://github.com/alibaba/MNN).<br />

## Requirements
👍 python>=3.6 is required.

## Installation
### Step 1: Convert Pytorch model weights to MNN model weights

**If you don't want to install anything on your system then use this [Google Colab](https://colab.research.google.com/drive/1CpV_RTNJamhMpFLT4tW2gBHB41bWaACp?usp=sharing) (*Recommended*).**  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](hhttps://colab.research.google.com/drive/1CpV_RTNJamhMpFLT4tW2gBHB41bWaACp?usp=sharing) (👍👍👍***Open and run on Google Chrome recommended***).

**And if you want to perform the conversion on your system then follow bellow instructions:**

📣 I recommend create a new conda environment (python version 3.6 recommended): 

```bash
$ conda create -n yolov5_conversion python=3.6
$ conda activate yolov5_conversion
```

➡️ Then run below commands and replace **yolov5s.pt** with your own model weights in path **weights/pt/** and also change **yolov5s.yaml** in path **models/** accordingly. 

```bash
$ git clone https://github.com/AnhPC03/yolov5.git
$ cd yolov5
$ pip install -r requirements.txt
$ bash export_mnn.sh yolov5s 640
```
✅ With ***yolov5s*** is model name and ***640*** is input size of your Pytorch model.<br />
✅ After you run above commands, you will see **successfully message**. And you can find MNN converted model in path **weights/mnn/**.<br />
✅ The size of MNN model weight is much smaller than origin Pytorch model weight.<br />

### Step 2: Run the inference on Intel/AMD CPU
⚙️ **Setup**

If you have created conda environment in conversion step then activated it (`$ conda activate yolov5_conversion`) and follow below steps.<br />
📣 If you **don't want** to install conda environment into your system, feel free to **skip** these below commands.<br />
Otherwise I recommend you creat a conda environment (python version 3.6 recommended): 

```bash
$ conda create -n yolov5_conversion python=3.6
$ conda activate yolov5_conversion
```

➡️ Follow below steps to install minimum required environment for converting Pytorch to MNN model weight file:

```bash
$ git clone https://github.com/AnhPC03/yolov5-export-to-cpu-mnn.git
$ cd yolov5-export-to-cpu-mnn
$ pip install -r requirements.txt
```
✅ Replace content of **classes.txt** file with your classes when you trained your Pytorch model.<br />

🎉 **Run inference**<br />
🍻 *Result will be saved to **results/** folder*
```bash
$ python inference/run_mnn_detector.py \
            --weights path/to/your/mnn/weight \
            --source 0  # webcam
                     file.jpg  # image 
                     file.mp4  # video
                     path/  # directory
                     'https://youtu.be/NUsoVlDFqZg'  # YouTube video
                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
```
✅ With *path/to/your/mnn/weight* is path to MNN model weight which you just converted in the above step.<br />
✅ You can use --nodisplay to do not display image or webcam while inference. Or --nosave to do not save inference results.<br />
✅ **For example**: my MNN model weight in ***yolov5-export-to-cpu-mnn/weights/yolov5s.mnn***. Then I run inference on images in ***inference/images/*** folder as below:
```bash
$ python inference/run_mnn_detector.py --weights weights/yolov5s.mnn --source inference/images
```