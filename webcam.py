import argparse
import time
import os

from loftr import LoFTR, default_cfg
from trtmodel import TRTModel
import torch
import cv2
from camera import Camera
import numpy as np


def make_query_image(frame, img_size):
    # ratio preserving resize
    img_h, img_w, _ = frame.shape
    scale_h = img_size[1] / img_h
    scale_w = img_size[0] / img_w
    scale_max = max(scale_h, scale_w)
    new_size = [int(img_w * scale_max), int(img_h * scale_max)]
    query_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    query_img = cv2.resize(query_img, new_size, interpolation=cv2.INTER_LINEAR)
    # center crop
    x = new_size[0] // 2 - img_size[0] // 2
    y = new_size[1] // 2 - img_size[1] // 2
    query_img = query_img[y:y + img_size[1], x:x + img_size[0]]
    return query_img


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

    device = torch.device(opt.device)
    print('Loading pre-trained network...')
    use_trt = False
    if os.path.exists(opt.trt):
        matcher = TRTModel(opt.trt)
        print('Successfully loaded TensorRT model.')
        use_trt = True
    else:
        matcher = LoFTR(config=default_cfg)
        checkpoint = torch.load(opt.weights)
        if checkpoint is not None:
            missed_keys, unexpected_keys = matcher.load_state_dict(checkpoint['state_dict'])
            if len(missed_keys) > 0:
                print('Checkpoint is broken')
                return 1
            print('Successfully loaded pre-trained weights.')
        else:
            print('Failed to load checkpoint')
            return 1
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
                    mconf, mkpts1, mkpts0,  = matcher(img0, img1)
                    mkpts0 = mkpts0.reshape((-1, 2))
                    mkpts1 = mkpts1.reshape((-1, 2))
                else:
                    with torch.no_grad():
                        mkpts0, mkpts1, mconf = matcher(img0, img1)
                        mkpts0 = mkpts0.cpu().numpy()
                        mkpts1 = mkpts1.cpu().numpy()
                        mconf = mconf.cpu().numpy()
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
