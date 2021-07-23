
# export PYTHONPATH=$PYTHONPATH:$PWD
export PYTHONPATH="$PWD"

INPUT_SIZE=640
MODEL_NAME=yolov5s

python conversion/export.py \
       --weights weights/pt/${MODEL_NAME}.pt \
       --img-size ${INPUT_SIZE} \
       --batch-size 1 \
       --out-dir weights/onnx

