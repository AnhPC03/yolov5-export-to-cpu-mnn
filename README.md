# Export YOLOv5 to run on Intel/AMD CPU
Export processing consists of two steps:
1️⃣ Convert Pytorch model weights to MNN model weights.
2️⃣ Run the inference on Intel/AMD CPU.

## Documentation
ℹ️ MNN is a lightweight deep neural network inference engine.
🔎 You can find more information about MNN in [here](https://www.yuque.com/mnn/en/about).
🔎 MNN github repository in [here](https://github.com/alibaba/MNN).

## Requirements
👍 Python>=3.6.0 is required.

## Installation
### Step 1: Convert Pytorch model weights to MNN model weights

**If you don't want to install anything on your system then use this [Google Colab](https://colab.research.google.com/drive/1oZN9azdFyrlbzeVcGaqdddJ_YVatVTJJ?usp=sharing) (*Recommended*).**  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1oZN9azdFyrlbzeVcGaqdddJ_YVatVTJJ?usp=sharing)

##### And if you want to perform the conversion on your system then follow bellow instructions:

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
$ bash export_mnn.sh 
```
✅ You can change the ***INPUT_SIZE*** and ***MODEL_NAME*** of your model in file ***export_mnn.sh***.
✅ After you run above commands, you will see **successfully message**. And you can find MNN converted model in path **weights/mnn/**.
✅ The size of MNN model weight is much smaller than origin Pytorch model weight.

### Step 2: Run the inference on Intel/AMD CPU
##### ⚙️ Setup

If you have created conda environment in conversion step then activated it (`$ conda activate yolov5_conversion`) and follow below steps. Otherwise I recommend you creat a conda environment (python version 3.6 recommended): 

```bash
$ conda create -n yolov5_conversion python=3.6
$ conda activate yolov5_conversion
```

➡️ Then follow below steps:

```bash
$ git clone https://github.com/AnhPC03/yolov5-export-to-cpu-mnn.git
$ cd yolov5-export-to-cpu-mnn
$ pip install -r requirements.txt
```

##### 🎉 Run inference
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
✅ With *path/to/your/mnn/weight* is path to MNN model weight which you just converted in the above step.
✅ **For example**: my MNN model weight in ***yolov5-export-to-cpu-mnn/weights/yolov5s.mnn***. Then I run inference on images in ***inference/images/*** folder as below:
```bash
$ python inference/run_mnn_detector.py --weights weights/yolov5s.mnn --source inference/images
```