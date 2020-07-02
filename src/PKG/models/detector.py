import cv2
import numpy as np 
import concurrent.futures


class CascadeDetector:
    def __init__(self, scaleFactor, minNeighbors):
        self.scaleFactor = scaleFactor
        self.minNeighbors = minNeighbors
        self.params_path = "../configs/haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(self.params_path)
    

    def __call__(self, frame, visualize=False):
        detection = frame.copy()
        detection = cv2.cvtColor(detection, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(
            detection, scaleFactor=self.scaleFactor, 
            minNeighbors = self.minNeighbors
        )
        if visualize:
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (235, 135, 21), 2)
                cv2.rectangle(frame, (x, y-20), (x+50, y), (235, 135, 21), -1)
                cv2.putText(
                    frame, "Person", (x+5, y-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, 
                    (255, 255, 255), 1, cv2.LINE_AA
                )
            return frame
        else:
            faces_batch = []
            for face in faces:
                if len(face) == 4:
                    x, y, w, h = face
                    faces_batch.append(frame[y:y+h, x:x+w])
            return faces_batch
