import argparse
import time
import os
import cv2
import numpy as np
from camera import Camera
from utils import make_query_image, get_coarse_match


def main():
    parser = argparse.ArgumentParser(description='LoFTR demo.')
    parser.add_argument('--weights', type=str, default='weights/outdoor_ds.ckpt',
                        help='Path to network weights.')
    parser.add_argument('--camid', type=int, default=0,
                        help='OpenCV webcam video capture ID, usually 0 or 1.')
    parser.add_argument('--device', type=str, default='cuda',
                        help='cpu or cuda')
    parser.add_argument('--trt', type=str, help='TensorRT model engine path')

    opt = parser.parse_args()
    print(opt)

    print('Loading pre-trained network...')
    use_trt = False
    if opt.trt and os.path.exists(opt.trt):
        from trtmodel import TRTModel
        matcher = TRTModel(opt.trt)
        print('Successfully loaded TensorRT model.')
        use_trt = True
    else:
        # import torch only it's required because it occupies too much memory
        from loftr import LoFTR, default_cfg
        from utils import make_student_config
        import torch

        make_student_config(default_cfg)
        matcher = LoFTR(config=default_cfg)
        # checkpoint = torch.load(opt.weights)
        # if checkpoint is not None:
        #     missed_keys, unexpected_keys = matcher.load_state_dict(checkpoint['state_dict'], strict=False)
        #     if len(missed_keys) > 0:
        #         print('Checkpoint is broken')
        #         return 1
        #     print('Successfully loaded pre-trained weights.')
        # else:
        #     print('Failed to load checkpoint')
        #     return 1

    if not use_trt:
        device = torch.device(opt.device)
        matcher = matcher.eval().to(device=device)

    print('Opening camera...')
    camera = Camera(opt.camid)

    win_name = 'LoFTR features'
    cv2.namedWindow(win_name)
    prev_frame_time = 0

    stop_img = None
    stop_frame = None
    do_blur = False
    do_pause = False

    img_size = (640, 480)
    loftr_coarse_resolution = 16  # 8

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
                if use_trt:
                    img0 = frame0[None][None] / 255.0
                    img1 = frame1[None][None] / 255.0
                else:
                    img0 = torch.from_numpy(frame0)[None][None].to(device=device) / 255.0
                    img1 = torch.from_numpy(frame1)[None][None].to(device=device) / 255.0

                # Inference with LoFTR and get prediction
                start = time.perf_counter()
                if use_trt:
                    conf_matrix = matcher(img0, img1)
                    # conf_matrix = conf_matrix.reshape((1, 4800, 4800))
                    conf_matrix = conf_matrix.reshape((1, 1200, 1200))
                else:
                    with torch.no_grad():
                        conf_matrix = matcher(img0, img1)
                        conf_matrix = conf_matrix.cpu().numpy()

                mkpts0, mkpts1, mconf = get_coarse_match(conf_matrix, img_size[1], img_size[0], loftr_coarse_resolution)

                infer_time = time.perf_counter() - start

                # filter only the most confident features
                n_top = 20
                indices = np.argsort(mconf)[::-1]
                indices = indices[:n_top]
                mkpts0 = mkpts0[indices, :]
                mkpts1 = mkpts1[indices, :]

            left_image = stop_img.copy()
            draw_features(left_image, mkpts0, img_size)
            draw_features(new_img, mkpts1, img_size)

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
            if key == ord('q'):
                print('Quitting, \'q\' pressed.')
                break
            if key == ord('s'):
                stop_img = frame.copy()
                stop_frame = make_query_image(stop_img, img_size)
            if key == ord('b'):
                do_blur = not do_blur
            if key == ord('p'):
                do_pause = not do_pause

    camera.close()
    cv2.destroyAllWindows()


def draw_fps(time_diff, image):
    fps = 1. / time_diff
    fps_str = 'FPS: ' + str(int(fps))
    cv2.putText(image, fps_str, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 200), 2, cv2.LINE_AA)


def draw_inference(time_diff, image):
    fps_str = f'Inference: {time_diff:.2} s'
    cv2.putText(image, fps_str, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 200), 2, cv2.LINE_AA)


def draw_features(image, features, img_size):
    indices = range(len(features))
    sx = image.shape[1] / img_size[0]
    sy = image.shape[0] / img_size[1]

    for i, point in zip(indices, features):
        point_int = (int(round(point[0] * sx)), int(round(point[1] * sy)))
        cv2.circle(image, point_int, 2, (0, 255, 0), -1, lineType=16)
        cv2.putText(image, str(i), point_int, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1, cv2.LINE_AA)


if __name__ == "__main__":
    main()
