import argparse
import os
import time

import cv2
import numpy as np

from camera import Camera
from loftr.utils.cvpr_ds_config import default_cfg
from utils import get_coarse_match, make_query_image, make_student_config


def main():
    parser = argparse.ArgumentParser(description="LoFTR demo.")
    parser.add_argument(
        "--weights",
        type=str,
        default="weights/LoFTR_teacher.pt",  # 'weights/outdoor_ds.ckpt',
        help="Path to network weights.",
    )
    parser.add_argument(
        "--original",
        action="store_true",
        help="If specified the original LoFTR model will be used.",
    )
    parser.add_argument(
        "--camid",
        type=int,
        default=0,
        help="OpenCV webcam video capture ID, usually 0 or 1.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="cpu or cuda")
    parser.add_argument("--trt", type=str, help="TensorRT model engine path")
    parser.add_argument("--onnx", type=str, help="ONNX model path")

    opt = parser.parse_args()
    print(opt)

    if opt.original:
        model_cfg = default_cfg
    else:
        model_cfg = make_student_config(default_cfg)

    print("Loading pre-trained network...")
    use_trt = False
    use_onnx = False
    if opt.trt and os.path.exists(opt.trt):
        from trtmodel import TRTModel

        matcher = TRTModel(opt.trt)
        print("Successfully loaded TensorRT model.")
        use_trt = True
    elif opt.onnx and os.path.exists(opt.onnx):
        import onnxruntime

        if opt.device == "cuda":
            onnx_providers = ["CUDAExecutionProvider"]
        else:
            onnx_providers = ["CPUExecutionProvider"]
        matcher = onnxruntime.InferenceSession(opt.onnx, providers=onnx_providers)
        print(f"ONNX runtime device {onnxruntime.get_device()}")
        use_onnx = True
    else:
        # import torch only if it's required because it occupies too much memory
        import torch
        import torch.nn.functional

        from loftr import LoFTR

        matcher = LoFTR(config=model_cfg)
        checkpoint = torch.load(opt.weights)
        if checkpoint is not None:
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint["model_state_dict"]
            missed_keys, unexpected_keys = matcher.load_state_dict(
                state_dict, strict=False
            )
            if len(missed_keys) > 0:
                print("Checkpoint is broken")
                return 1
            if not use_trt:
                device = torch.device(opt.device)
                matcher = matcher.eval().to(device=device)
            print("Successfully loaded pre-trained weights.")
        else:
            print("Failed to load checkpoint")
            return 1

    print("Opening camera...")
    camera = Camera(opt.camid)

    win_name = "LoFTR features"
    cv2.namedWindow(win_name)
    prev_frame_time = 0

    stop_img = None
    stop_frame = None
    do_blur = False
    do_pause = False

    img_size = (model_cfg["input_width"], model_cfg["input_height"])
    loftr_coarse_resolution = model_cfg["resolution"][0]

    while True:
        frame, ret = camera.get_frame()
        if ret and frame is not None:
            if not do_pause:
                if do_blur:
                    frame = cv2.blur(frame, (3, 3))

                new_img = frame.copy()

                # make batch
                if stop_frame is None:
                    frame0 = make_query_image(frame, img_size)
                    stop_img = new_img.copy()
                else:
                    frame0 = stop_frame

                frame1 = make_query_image(frame, img_size)
                if use_trt or use_onnx:
                    img0 = frame0[None][None] / 255.0
                    img1 = frame1[None][None] / 255.0
                else:
                    img0 = (
                        torch.from_numpy(frame0)[None][None].to(device=device) / 255.0
                    )
                    img1 = (
                        torch.from_numpy(frame1)[None][None].to(device=device) / 255.0
                    )

                # Inference with LoFTR and get prediction
                start = time.perf_counter()
                if use_trt:
                    conf_matrix = matcher(img0, img1)
                    if opt.original:
                        conf_matrix = conf_matrix.reshape((1, 4800, 4800))
                    else:
                        conf_matrix = conf_matrix.reshape((1, 1200, 1200))
                elif use_onnx:
                    onnx_inputs = {
                        matcher.get_inputs()[0].name: img0.astype(np.float32),
                        matcher.get_inputs()[1].name: img1.astype(np.float32),
                    }
                    onnx_outputs = matcher.run(None, onnx_inputs)
                    conf_matrix = onnx_outputs[0]
                else:
                    with torch.no_grad():
                        conf_matrix, _ = matcher(img0, img1)
                        conf_matrix = conf_matrix.cpu().numpy()

                mkpts0, mkpts1, mconf = get_coarse_match(
                    conf_matrix, img_size[1], img_size[0], loftr_coarse_resolution
                )

                infer_time = time.perf_counter() - start

                # filter only the most confident features
                n_top = 20
                indices = np.argsort(mconf)[::-1]
                indices = indices[:n_top]
                mkpts0 = mkpts0[indices, :]
                mkpts1 = mkpts1[indices, :]

            left_image = stop_img.copy()
            draw_features(left_image, mkpts0, img_size, color=(0, 255, 0))
            draw_features(new_img, mkpts1, img_size, color=(0, 255, 0))

            # combine images
            res_img = np.hstack((left_image, new_img))

            # draw FPS
            new_frame_time = time.perf_counter()
            time_diff = new_frame_time - prev_frame_time
            prev_frame_time = new_frame_time

            draw_fps(time_diff, res_img)
            draw_inference(infer_time, res_img)

            cv2.imshow(win_name, res_img)
            key = cv2.waitKey(delay=1)
            if key == ord("q"):
                print("Quitting, 'q' pressed.")
                break
            if key == ord("s"):
                stop_img = frame.copy()
                stop_frame = make_query_image(stop_img, img_size)
            if key == ord("b"):
                do_blur = not do_blur
            if key == ord("p"):
                do_pause = not do_pause

    camera.close()
    cv2.destroyAllWindows()


def draw_fps(time_diff, image):
    fps = 1.0 / time_diff
    fps_str = "FPS: " + str(int(fps))
    cv2.putText(
        image,
        fps_str,
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (200, 200, 200),
        2,
        cv2.LINE_AA,
    )


def draw_inference(time_diff, image):
    fps_str = f"Inference: {time_diff:.2} s"
    cv2.putText(
        image,
        fps_str,
        (10, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (200, 200, 200),
        2,
        cv2.LINE_AA,
    )


def draw_features(image, features, img_size, color, draw_text=True):
    indices = range(len(features))
    sx = image.shape[1] / img_size[0]
    sy = image.shape[0] / img_size[1]

    for i, point in zip(indices, features):
        point_int = (int(round(point[0] * sx)), int(round(point[1] * sy)))
        cv2.circle(image, point_int, 2, color, -1, lineType=16)
        if draw_text:
            cv2.putText(
                image,
                str(i),
                point_int,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )


if __name__ == "__main__":
    main()
