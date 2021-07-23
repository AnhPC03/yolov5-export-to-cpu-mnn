# Convert yolov5 models (verson 2) in `*.pt` format to `OpenVINO`

## 1. Convert form PyTorch to ONNX

### 1.1. Requirements

Install following packages:

```bash
$ pip install torch==1.5.1 torchvision==0.6.1
$ pip install onnx==1.7.0
$ pip install onnxruntime==1.4.0
$ pip3 install onnx-simplifier
```

### 1.2. Conversion

```bash
$ export PYTHONPATH=$PYTHONPATH:$PWD
$ INPUT_SIZE=320
$ MODEL_NAME=yolov5xxs_${INPUT_SIZE}_face
$ python conversion/export.py \
    --weights weights/pytorch/${MODEL_NAME}.pt \
    --img-size ${INPUT_SIZE} \
    --batch-size 1 \
    --out-dir weights/onnx
```

### 1.3. Verification

```bash
$ python conversion/demo_onnx.py --display
```

## 2. Convert from ONNX to OpenVINO

### 2.1. OpenVINO installation

Install `OpenVINO` 2020.4 in a `USER` account.

### 2.2. Configuration

Please use `python 3.6`

```bash
$ cd <INSTALL_DIR>/deployment_tools/model_optimizer/
$ virtualenv -p python3 openvino --system-site-packages
$ source openvino/bin/activate
$ pip3 install -r requirements_onnx.txt
```

### 2.3. Conversion

You may need to use [onnx simplifier](https://github.com/daquexian/onnx-simplifier) before converting to `OpenVINO`:

```bash
$ INPUT_SIZE=320
$ MODEL_NAME=yolov5xxs_${INPUT_SIZE}_face
$ python -m onnxsim weights/onnx/${MODEL_NAME}.onnx weights/onnx/${MODEL_NAME}.onnx --input-shape 1,3,${INPUT_SIZE},${INPUT_SIZE}
```

Activate the conversion env.:

```bash
$ cd <YOLOv5_INSTALL_DIR>
$ source ~/intel/openvino/deployment_tools/model_optimizer/openvino/bin/activate
$ source ~/intel/openvino/bin/setupvars.sh
```

Now, convert to `OpenVINO`:

```bash
$ INPUT_SIZE=320
$ MODEL_NAME=yolov5xxs_${INPUT_SIZE}_face
$ python ~/intel_2020R4/openvino/deployment_tools/model_optimizer/mo_onnx.py \
    --input_model /Users/thanhnguyen/Documents/Sourcecodes/yolov5/weights/onnx/${MODEL_NAME}.onnx \
    --output_dir /Users/thanhnguyen/Documents/Sourcecodes/yolov5/weights/openvino \
    --input_shape="[1,3,${INPUT_SIZE},${INPUT_SIZE}]" \
    --data_type FP16 \
    --scale 255 \
    --reverse_input_channels
    # --log_level DEBUG
```

### 2.4. Verification

```bash
$ python conversion/demo_openvino_video.py --display
```

## 3. Evaluate OpenVINO models

```bash
# Remove cached labels
$ rm /Users/thanhnguyen/Documents/Datasets/MergedMaskFace_YOLO/labels/*.cache
$ INPUT_SIZE=320
$ python conversion/eval.py \
    --model-type openvino \
    --model-xml weights/openvino/yolov5xxs_${INPUT_SIZE}_face.xml \
    --data ./data/mergedmaskface.yaml \
    --batch 1 \
    --img ${INPUT_SIZE} \
    --task 'test' \
    --device HDDL \
    --conf-thres 0.3 \
    --verbose
```



