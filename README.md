# LoFTR Coarse TRT

This project provides a deep learning model for the `Local Feature Matching` for two images that can be used on the embedded devices like NVidia Jetson Nano 2GB with a reasonable accuracy and performance - `5 FPS`. The algorithm is based on the `coarse part` of "LoFTR: Detector-Free Local Feature Matching with Transformers". But the model has a reduced number of ResNet and coarse transformer layers so there is the much lower memory consumption and the better performance. The required level of accuracy was achieved by applying the `Knowledge distillation` technique and training on the [BlendedMVS](https://github.com/YoYo000/BlendedMVS) dataset.

The code is based on the original [LoFTR](https://github.com/zju3dv/LoFTR) repository, but was adapted for compatibility with [TensorRT](https://developer.nvidia.com/tensorrt) technology, especially dependencies to `einsum` and `einops` were removed.

### Model weights
Weights for the PyTorch model, ONNX model and TensorRT engine files are located in the `weights` folder.

Weights for original LoFTR coarse module can be downloaded using the original [url](https://drive.google.com/drive/folders/1DOcOPZb3-5cWxLqn256AhwUVjBPifhuf?usp=sharing) that was provider by paper authors, now only the `outdoor-ds` file is supported.

### Demo

There is a Demo application, that can be ran with the `webcam.py` script. There are following parameters:
* `--weights` - The path to PyTorch model weights, for example 'weights/LoFTR_teacher.pt' or 'weights/outdoor_ds.ckpt'                       
* `--trt` - The path to the TensorRT engine, for example 'weights/LoFTR_teacher.trt'
* `--onnx` - The path to the ONNX model, for example 'weights/LoFTR_teacher.onnx'
* `--original` - If specified the original LoFTR model will be used, can be used only with `--weights` parameter
* `--camid` - OpenCV webcam video capture ID, usually 0 or 1, default 0
* `--device` - Selects the runtime back-end CPU or CUDA, default is CUDA

Sample command line:
```
python3 webcam.py --trt=weights/LoFTR_teacher.trt --camid=0
```

Demo application shows a window with pair of images captured with a camera. Initially there will be the two same images. Then you can choose a view of interest and press the `s` button, the view will be remembered and will be visible as the left image. Then you can change the view and press the `p` button to make a snapshot of the feature matching result, the corresponding features will be marked with the same numbers at the two images. If you press the `p` button again then application will allow you to change the view and repeat the feature matching process. Also this application shows the real-time FPS counter so you can estimate the model performance.

### Training

[LoFTR Paper:](https://arxiv.org/pdf/2104.00680.pdf)

```bibtex
@article{sun2021loftr,
  title={{LoFTR}: Detector-Free Local Feature Matching with Transformers},
  author={Sun, Jiaming and Shen, Zehong and Wang, Yuang and Bao, Hujun and Zhou, Xiaowei},
  journal={{CVPR}},
  year={2021}
}
```
