# LoFTR_TRT
The TensorRT adaptation of LoFTR: Detector-Free Local Feature Matching with Transformers.

This a clone of the original [LoFTR](https://github.com/zju3dv/LoFTR) repository.
The code was adapted for compatibility with [TRTorch](https://github.com/NVIDIA/TRTorch) compile, especially dependencies to `einsum` and `einops` were removed.

The goal is to create TensorRT optimized inference script that can be used on the NVidia Jetson Nano device with a reasonable performance.

For the weights download please use the original [url](https://drive.google.com/drive/folders/1DOcOPZb3-5cWxLqn256AhwUVjBPifhuf?usp=sharing) that was provider by paper authors, the `outdoor-ds` file is only supported.

TODO:
- [x] Create a student model with reduced number of parameters
- [ ] Develop a Knowledge distillation training pipe-line
- [ ] Share the complete optimized PyTorch script

[Paper:](https://arxiv.org/pdf/2104.00680.pdf)

```bibtex
@article{sun2021loftr,
  title={{LoFTR}: Detector-Free Local Feature Matching with Transformers},
  author={Sun, Jiaming and Shen, Zehong and Wang, Yuang and Bao, Hujun and Zhou, Xiaowei},
  journal={{CVPR}},
  year={2021}
}
```