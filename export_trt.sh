#!/usr/bin/env bash

ONNX_MODEL=weights/LoFTR_teacher.onnx
TRT_MODEL=weights/LoFTR_teacher.trt 

# /usr/src/tensorrt/bin/trtexec --onnx=$ONNX_MODEL --saveEngine=$TRT_MODEL --verbose

/usr/src/tensorrt/bin/trtexec --onnx=$ONNX_MODEL --saveEngine=$TRT_MODEL --best --verbose --useCudaGraph --workspace=8

