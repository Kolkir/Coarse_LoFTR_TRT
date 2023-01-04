from threading import Thread

import cv2


class Camera(object):
    def __init__(self, index):
        self.cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            print("Failed to open camera {0}".format(index))
            exit(-1)

        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        self.status = False
        self.frame = None

    def update(self):
        while True:
            try:
                if self.cap.isOpened():
                    (self.status, self.frame) = self.cap.read()
                else:
                    break
            except cv2.error as e:
                print(e)
                break

    def get_frame(self):
        return self.frame, self.status

    def close(self):
        self.cap.release()
        self.thread.join()
