import cv2
import time
import numpy as np
import imutils
import matplotlib.pyplot as plt
from imutils.video import VideoStream

class IO:
    """
    General class to Input feed manipulation.
    Take the input feed from 1. webcam 2. recorded video file 3. other possible sources.
    Read the input stream and return the pre-processed frames to get specified frame format to further process
    """

    def __init__(self, feed_option=0, webcam_id=0, file_path=None, frame_size=450, fps=10, show_frames=True, plot_act=True):

        SRC_OPTIONS = {'webcam': 0, 'rec_video': 1, 'image': 2}

        self.fps = fps
        self.frame_size = frame_size
        self.src = SRC_OPTIONS[feed_option]
        self.show_frames = show_frames
        self.plot_act = plot_act
        self.horizon = np.inf
        self.score_list = []

        if self.src == 0:
            self.video = VideoStream(src=webcam_id).start()
            time.sleep(1.0)
        elif self.src == 1 and file_path:
            self.video = cv2.VideoCapture(file_path)
        elif self.src == 2 and file_path:
            self.frame = cv2.imread(file_path)
            self.horizon = 1
        else:
            self.video = None
            self.frame = None

        self.time = 0

    def next_frame(self, face_box=None, eye_boxes=None, structure_pts=None, score=-1, flag=0, redflag=0):
        if self.time > 0:
            if self.show_frames:
                self.add_structures(face_box, eye_boxes, structure_pts, flag=flag)
                self.show_frame(redflag)
            if self.plot_act and score != -1:
                self.show_plot(score)

        if self.time >= self.horizon:
            print('Run complete, total {} frames processed.'.format(self.time))
            return None

        if self.src == 0:
            self.image = imutils.resize(self.video.read(), width=self.frame_size)
            self.frame = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        elif self.src == 1:
            ret, frame = self.video.read()
            if not ret:
                return None
            self.image = imutils.resize(frame, width=self.frame_size)
            self.frame = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        elif self.src == 2:
            self.image = imutils.resize(self.frame, width=self.frame_size)
            self.frame = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            return None
        self.time += 1
        return self.frame

    def add_structures(self, face_box, eye_boxes, structure_pts, flag=0):
        color = {0: (0, 255, 0), 1: (0, 0, 255)}
        if face_box is not None:
            tl, br = face_box.tl_corner(), face_box.br_corner()
            self.image = cv2.rectangle(self.image, (tl.x, tl.y), (br.x, br.y), color[flag], 2)
        if eye_boxes is not None:
            for eye_box in eye_boxes:
                for pt in range(eye_box.shape[0]):
                    self.image = cv2.circle(self.image, tuple(eye_box[pt,:]), 0, color[flag], 3)
        if structure_pts is not None:
            for pt in range(structure_pts.shape[0]):
                self.image = cv2.circle(self.image, tuple(structure_pts[pt,:]), 0, (255, 0, 0), 2)

    def show_frame(self, redflag=0):
        if self.frame is not None:
            if redflag==1:
                self.image=cv2.applyColorMap(self.image, cv2.COLORMAP_AUTUMN)
            cv2.imshow('frame---press q to stop', self.image)
            if self.src == 2:
                cv2.waitKey()
                self.horizon = self.time
            else:
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    self.horizon = self.time

    def show_plot(self, score):
        history = 40
        self.score_list = self.score_list[-(history-1):]+[score]
        img_size = (200, 500, 3)
        img = 255*np.ones(img_size, dtype='uint8')
        cv2.polylines(img, [np.array([[10,160], [10,10]])], 0, (0,0,0), thickness=1)
        cv2.polylines(img, [np.array([[10,160], [480,160]])], 0, (0,0,0), thickness=1)
        cv2.polylines(img, [np.array([[10*(idx+1)+10, 10+int(200*(0.5-scr))] for idx, scr in enumerate(self.score_list)])], 0, (0,128,0), thickness=1)
        cv2.imshow('plot---press q to stop', img)
        if self.src == 2:
            cv2.waitKey()
            self.horizon = self.time
        else:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                self.horizon = self.time
