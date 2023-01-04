# Coarse LoFTR TRT

[Google Colab demo notebook](https://colab.research.google.com/drive/1RFMAqfJeDaBoBQ7p5zXtJNXZE7DHGqlt?usp=sharing)

This project provides a deep learning model for the `Local Feature Matching` for two images that can be used on the embedded devices like NVidia Jetson Nano 2GB with a reasonable accuracy and performance - `5 FPS`. The algorithm is based on the `coarse part` of "LoFTR: Detector-Free Local Feature Matching with Transformers". But the model has a reduced number of ResNet and coarse transformer layers so there is the much lower memory consumption and the better performance. The required level of accuracy was achieved by applying the `Knowledge distillation` technique and training on the [BlendedMVS](https://github.com/YoYo000/BlendedMVS) dataset.

The code is based on the original [LoFTR](https://github.com/zju3dv/LoFTR) repository, but was adapted for compatibility with [TensorRT](https://developer.nvidia.com/tensorrt) technology, especially dependencies to `einsum` and `einops` were removed.

## Model weights

Weights for the PyTorch model, ONNX model and TensorRT engine files are located in the `weights` folder.

Weights for original LoFTR coarse module can be downloaded using the original [url](https://drive.google.com/drive/folders/1DOcOPZb3-5cWxLqn256AhwUVjBPifhuf?usp=sharing) that was provider by paper authors, now only the `outdoor-ds` file is supported.

## Demo

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

## Training

To repeat the training procedure you should use the low-res set of the [BlendedMVS](https://github.com/YoYo000/BlendedMVS) dataset. After download you can use the `train.py` script to run training process. There are following parameters for this script:

* `--path` - Path to the dataset
* `--checkpoint_path` - Where to store a log information and checkpoints, default value is 'weights'
* `--weights` - Path to the LoFTR teacher model weights, default value is 'weights/outdoor_ds.ckpt'

Sample command line:

```
python3 train.py --path=/home/user/datasets/BlendedMVS --checkpoint_path=weights/experiment1/
```

Please use the `train/settings.py` script to configure the training process. Please notice that by default the following parameters are enabled:

```
self.batch_size = 32
self.batch_size_divider = 8  # Used for gradient accumulation
self.use_amp = True
self.epochs = 35
self.epoch_size = 5000
```

This set of parameters was chosen for training with the Nvidia GTX1060 GPU, which is the low level consumer level card. The `use_amp` parameter means the [automatic mixed precision](https://pytorch.org/docs/stable/amp.html) will be used to reduce the memory consumption and the training time. Also, the gradient accumulation technique is enabled with the `batch_size_divider` parameter, it means the actual batch size will be `32/8` but for larger batch size simulation the 8 batches will be averaged. Moreover, the actual size of the epoch is reduced with the `epoch_size` parameter, it means that on every epoch only 5000 dataset elements will be randomly picked from the whole dataset.

[Paper](https://arxiv.org/abs/2202.00770)

```bibtex
@misc{kolodiazhnyi2022local,
      title={Local Feature Matching with Transformers for low-end devices}, 
      author={Kyrylo Kolodiazhnyi},
      year={2022},
      eprint={2202.00770},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

[LoFTR Paper:](https://arxiv.org/pdf/2104.00680.pdf)

```bibtex
@article{sun2021loftr,
  title={{LoFTR}: Detector-Free Local Feature Matching with Transformers},
  author={Sun, Jiaming and Shen, Zehong and Wang, Yuang and Bao, Hujun and Zhou, Xiaowei},
  journal={{CVPR}},
  year={2021}
}
```
