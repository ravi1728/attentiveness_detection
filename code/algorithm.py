import numpy as np
from scipy.spatial import distance

class EARtracker:
    def __init__(self):
        pass

    def ear(self, eye_box):
    	A = distance.euclidean(eye_box[1], eye_box[5])
    	B = distance.euclidean(eye_box[2], eye_box[4])
    	C = distance.euclidean(eye_box[0], eye_box[3])
    	ear = (A + B) / (2.0 * C)
    	return ear

    def process(self, face_box, eye_boxes, structure_pts):
        if eye_boxes is None:
            return 0, -1
        EAR = [self.ear(eye_box) for eye_box in eye_boxes]
        if len(EAR) == 0:
            return 0, -1
        score = sum(EAR)/len(EAR)
        flag = (score < 0.22)
        print(score, flag)
        return score, flag
