

MODEL_NAME=yolov5s

mnnconvert -f ONNX \
        --modelFile weights/onnx/${MODEL_NAME}.onnx \
        --MNNModel weights/mnn/${MODEL_NAME}.mnn \
        --weightQuantBits 8

