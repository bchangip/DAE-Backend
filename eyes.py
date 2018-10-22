# shape_predictor_68_face_landmarks.dat
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import imutils
import time
import dlib
import cv2
from keras.models import load_model
import glob
from os import path


data_set = "shape_predictor_68_face_landmarks.dat"
EYE_AR_THRESH = 0.27
EYE_AR_CONSEC_FRAMES = 3
window_size = 800
in_loop= True

def eyePos(array):
    c = array[0]
    l = array[1]
    l2 = array[2]
    r = array[3]
    r2 = array[4]
    d = array[5]
    dl = array[6]
    dr = array[7]

    if c == 0 and d == 0 and l == 0:
        return 5
    elif r == 1 and r2 == 1 and l2 == 0 and l == 1:
        return 6
    elif r == 0 and r2 == 0 and l2 == 1 and l == 1:
        return 4
    elif dl == 1 and dr == 1 and l2 == 0 and d == 1:
        return 8
    elif dl == 1 and dr == 1 and l==0 and l2==0:
        return 9
    elif dl == 1 and dr == 1 and r == 0 and r == 0:
        return 7
    else:
        return 5

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


def square(frame, array):
    cv2.line(frame,(int(array[1][0]),int(array[1][1])),(int(array[2][0]),int(array[2][1])),(255,255,255),1 )
    cv2.line(frame,(int(array[4][0]),int(array[4][1])),(int(array[5][0]),int(array[5][1])),(255,0,255),1 )

def check_range(intensity):
    if intensity > 160:
        return [(0,0,0),0]
    else:
        return [(255,255,255),1]

def check_range2(center, intensity):
    if intensity < center:
        return [(0,0,0),0]
    else:
        return [(255,255,255),1]

def points(frame, array):
    upperleft_x = int(array[1][0])
    upperleft_y = int(array[1][1])

    upperright_x = int(array[2][0])
    upperright_y = int(array[2][1])

    downleft_x = int(array[4][0])
    downleft_y = int(array[4][1]) - 2

    downright_x = int(array[5][0]) 
    downright_y = int(array[5][1]) - 2

    uppermid_x = int((array[1][0] + array[2][0])/2)
    uppermid_y = int((array[1][1] + array[2][1])/2)

    downmid_x = int((array[4][0] + array[5][0])/2)
    downmid_y = int((array[4][1] + array[5][1])/2)

    center_x = int((uppermid_x+downmid_x)/2)
    center_y = int((uppermid_y+downmid_y)/2)

    left_x = upperleft_x-2
    left_y = int((upperleft_y+downleft_y)/2)
    right_x = upperright_x +1
    right_y = int((array[2][1]+array[5][1])/2)

    further_l_x = left_x-3
    further_l_y =left_y

    further_r_x = right_x + 2
    further_r_y = right_y

    left = frame[left_x, left_y]
    right = frame[right_x, right_y]
    center = frame[center_x, center_y]
    upleft = frame[upperleft_x,upperleft_y]
    up = frame[uppermid_x,uppermid_y]
    upright = frame[upperright_x,upperright_y]
    downleft = frame[downleft_x,downleft_y]
    down = frame[downmid_x,downmid_y]
    downright = frame[downright_x,downright_y]
    left2 = frame[further_l_x, further_l_y]
    right2 = frame[further_r_x, further_r_y]

    rad = 0
    thic = 2

    # colors
    l = check_range2(left, center)
    r = check_range2(right, center)
    c = check_range(center)
    ul = check_range(upleft)
    u = check_range(up)
    ur = check_range(upright)
    dl = check_range2(downleft, down)
    d = check_range(down)
    dr =check_range2(downright, down)
    l2 = check_range2(left2, center)
    r2 = check_range2(right2, center)

    data = [c[1], l[1], l2[1], r[1], r2[1], d[1], dl[1], dr[1]]

    # print(further_l_x, further_l_y)
    # center points
    cv2.circle(frame, (left_x, left_y), rad, l[0], thic)
    cv2.circle(frame, (right_x, right_y), rad,r[0], thic)
    cv2.circle(frame, (center_x, center_y), rad, (255,255,255), thic)
    #cv2.circle(frame, (further_l_x, further_l_y), l2[0], thic)
    cv2.circle(frame, (further_r_x, further_r_y), rad,r2[0], thic)


    # upper points
    cv2.circle(frame, (uppermid_x, uppermid_y), rad,(255,255,255), thic)
    cv2.circle(frame, (upperleft_x, upperleft_y), rad, ul[0],thic)
    cv2.circle(frame, (upperright_x, upperright_y), rad,ur[0], thic)

    # downer points
    cv2.circle(frame, (downmid_x, downmid_y), rad,(255,255,255), thic)
    cv2.circle(frame, (downleft_x, downleft_y), rad,dl[0], thic)
    cv2.circle(frame, (downright_x, downright_y), rad,dr[0], thic) 

    pos = eyePos(data)

    return pos

def blinkCounter(video):
    eyepost = 77
    directions = []
    TOTAL = 0
    COUNTER = 0
    Total_Frames = 0
    time_for_blink = []
    consecutive_frames = []
    print("starting eyedetection: ", video)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(data_set)
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    vs = FileVideoStream(video).start()
    fileStream = True
    time.sleep(1.0)
    while in_loop:
        if fileStream and not vs.more():
            return TOTAL, Total_Frames, time_for_blink, directions

        frame = vs.read()
        Total_Frames += 1
        frame = imutils.resize(frame, width=window_size)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rects = detector(gray, 0)
        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            square(frame, leftEye)
            ear = (leftEAR + rightEAR) / 2.0
            new_eyepos = points(frame, rightEye)
            # print(new_eyepos)
            if eyepost != new_eyepos:
                directions.append(new_eyepos)
                eyepost = new_eyepos

            if ear < EYE_AR_THRESH:
                COUNTER += 1
                consecutive_frames.append(Total_Frames)
            else:
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    TOTAL += 1
                    if len(consecutive_frames) != 0:
                        timeStamp = consecutive_frames[0] / 30
                        time_for_blink.append(int(timeStamp))
                consecutive_frames = []
                COUNTER = 0
            cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            return TOTAL, Total_Frames, time_for_blink, directions

    cv2.destroyAllWindows()
    vs.stop()

def createFormatedData(array):
    movements = [0,0,0,0,0,0,0,0,0,0]
    for i in range(len(array)):
        if array[i] == 0:
            movements[0] += 1
        elif array[i] == 1:
            movements[1] += 1
        elif array[i] == 2:
            movements[2] += 1
        elif array[i] == 3:
            movements[3] += 1
        elif array[i] == 4:
            movements[4] += 1
        elif array[i] == 5:
            movements[5] += 1
        elif array[i] == 6:
            movements[6] += 1
        elif array[i] == 7:
            movements[7] += 1
        elif array[i] == 8:
            movements[8] += 1
        elif array[i] == 9:
            movements[9] += 1
        else:
            print('error in  creating format')
    return np.array([movements])  

def save_blinks(video):

    blinks, Total_Frames, seconds, dirs = blinkCounter(video)
    #print(dirs)
    #print(seconds)
    for i in range(len(seconds)):
        dirs.append(0)

    #print(dirs)
    dirs = createFormatedData(dirs)
    #print(dirs)
    return dirs

def getVideo():
    files = glob.glob('./*.avi')
    lastIndex = 0
    if (len(files) > 0):
        lastIndex = int(
            max(files, key=path.getctime).split("-")[1].split(".")[0])
    if (path.exists('./interview-%s.avi' % (str(lastIndex)))):
        return './interview-%s.avi' % (str(lastIndex))


def final_prediction():
    video = getVideo()
    data = save_blinks(video)
    model = load_model('eyes.h5')
    model.compile(loss='binary_crossentropy',
    optimizer='Adadelta',
    metrics=['accuracy'])
    a = model.predict_classes(data)
    p = model.predict(data)
    if a == 1:
        response = 'true'
    else:
        response = 'false'
    per = p[0][0]
    trust = round(per * 100, 2)
    return response, trust




#dirs = save_blinks('E016M.mp4')
#print(dirs)
#response, trust = final_prediction()
#print(response, trust)
