import cv2
import numpy as np
from functools import reduce
import dlib
import glob
from sklearn.externals import joblib
from imutils import face_utils
from scipy.spatial import distance as dist
from os import path

LANDMARKS = ['mouth', 'right_eyebrow', 'left_eyebrow', 'right_eye', 'left_eye']

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def getLandmarks(facial_landmarks):
    mouth = facial_landmarks[LANDMARKS[0]]
    right_eyebrow = facial_landmarks[LANDMARKS[1]]
    left_eyebrow = facial_landmarks[LANDMARKS[2]]
    right_eye = facial_landmarks[LANDMARKS[3]]
    left_eye = facial_landmarks[LANDMARKS[4]]

    center_rigth_eye = (
        right_eye[1][0] + ((right_eye[2][0] - right_eye[1][0]) / 2))

    center_left_eye = (left_eye[1][0] +
                       ((left_eye[2][0] - left_eye[1][0])/2))
    unit = (center_left_eye - center_rigth_eye) / 64

    lips_width = dist.euclidean(mouth[6], mouth[0])

    lips_height = (dist.euclidean(mouth[2], mouth[10]) +
                   dist.euclidean(mouth[4], mouth[8]))/2

    inner_mouth = (dist.euclidean(mouth[19], mouth[13])
                   + dist.euclidean(mouth[17], mouth[15]))/2

    rigth_eyebrown_eye_distance = (dist.euclidean(
        right_eye[1], right_eyebrow[2]) + dist.euclidean(right_eye[2], right_eyebrow[3]))/2

    left_eyebrown_eye_distance = (dist.euclidean(
        left_eye[1], left_eyebrow[1]) + dist.euclidean(left_eye[2], left_eyebrow[2]))/2

    return [(rigth_eyebrown_eye_distance * unit),
            (left_eyebrown_eye_distance * unit),
            (lips_width * unit),
            (lips_height * unit),
            (inner_mouth * unit)]


def analyzeFrames(fileName):
    model_clone = joblib.load('castro_model.pkl')
    stream = cv2.VideoCapture(str(fileName))
    FPS = stream.get(cv2.CAP_PROP_FPS)
    index = 0
    predictions = []
    while True:
        (grabbed, frame) = stream.read()
        if not grabbed:
            return predictions
        if((index % 3 == 0)):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 0)
            for (i, rect) in enumerate(rects):
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)
                facial_landmarks = {}
                for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
                    if name in LANDMARKS:
                        facial_landmarks[name] = shape[i:j]
                predictions.append(model_clone.predict(
                    [getLandmarks(facial_landmarks)])[0])

        index += 1
    stream.release()
    return predictions


def expression():
    files = glob.glob('./*.avi')
    lastIndex = 0
    if (len(files) > 0):
        lastIndex = int(
            max(files, key=path.getctime).split("-")[1].split(".")[0])
    if (path.exists('./interview-%s.avi' % (str(lastIndex)))):
        result = analyzeFrames('./interview-%s.avi' % (str(lastIndex)))
        countTrue = 0
        countLie = 0
        for i in result:
            if(i == 0):
                countLie += 1
            else:
                countTrue += 1
        if (countLie > countTrue):
            response = "false"
            trust = ((countLie / (countLie + countTrue)) * 0.63) * 100
        else:
            response = "true"
            trust = ((countTrue / (countLie + countTrue))) * 0.63 * 100
        return response, trust
    return "false", 63
