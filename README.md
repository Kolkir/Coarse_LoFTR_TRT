# LoFTR Coarse TRT

This project provides a deep learning model for the `Local Feature Matching` for two images that can be used on the emmeded devices like NVidia Jetson Nano 2GB with a reasonable accuracy and performance - `5 FPS`. The algorithm is based on the `coarse part` of "LoFTR: Detector-Free Local Feature Matching with Transformers". But the model has a reduced number of ResNet and coarse transformer layers so there is the much lower memory consumption and the better performnce. The required level of accuracy was achived by applying the `Knowledge distilation` technique and training on the [BlendedMVS](https://github.com/YoYo000/BlendedMVS) dataset.

The code is based on the original [LoFTR](https://github.com/zju3dv/LoFTR) repository, but was adapted for compatibility with [TensorRT](https://developer.nvidia.com/tensorrt) technology, especially dependencies to `einsum` and `einops` were removed.

### Model weights
Weights for the PyTorch model, ONNX model and TensorRT engine files are located in the `weights` folder.

Weights for original LoFTR coarse module can be downloaded using the original [url](https://drive.google.com/drive/folders/1DOcOPZb3-5cWxLqn256AhwUVjBPifhuf?usp=sharing) that was provider by paper authors, now only the `outdoor-ds` file is supported.

### Demo

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
