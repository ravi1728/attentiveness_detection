from scipy.spatial import distance as dist
from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2


class Face:
    def __init__(self, model):
        self.face_detector = dlib.get_frontal_face_detector()
        self.get_points = dlib.shape_predictor(model)

        self.left_i = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        self.right_i = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


    def extract(self, frame):
        face_box = self.face_detector(frame, 0)
        if len(face_box) > 0:
            areas = np.array([box.area() for box in face_box])
            face_box = face_box[np.argmax(areas)]
            structure_pts = self.get_points(frame, face_box)
            structure_pts = face_utils.shape_to_np(structure_pts)
            eye_boxes = [structure_pts[self.left_i[0]:self.left_i[1]], structure_pts[self.right_i[0]:self.right_i[1]]]
            timeflag=1
        else:
            face_box, eye_boxes, structure_pts = None, None, None
            timeflag=0
        return face_box, eye_boxes, structure_pts, timeflag
