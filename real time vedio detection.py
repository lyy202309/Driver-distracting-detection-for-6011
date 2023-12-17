import dlib
import numpy as np
import cv2
from collections import OrderedDict


shape_predictor_68_face_landmark=OrderedDict([
    ('mouth',(48,68)),
    ('right_eyebrow',(17,22)),
    ('left_eye_brow',(22,27)),
    ('right_eye',(36,42)),
    ('left_eye',(42,48)),
    ('nose',(27,36)),
    ('jaw',(0,17))
])

def get_eye_ratio(eye_points):

    eye_width = np.linalg.norm(eye_points[0] - eye_points[3])
    eye_height1 = np.linalg.norm(eye_points[1] - eye_points[5])
    eye_height2 = np.linalg.norm(eye_points[2] - eye_points[4])
    eye_height = (eye_height1 + eye_height2) / 2
    eye_ratio = eye_height / eye_width
    return eye_ratio
def drawRectangle(detected,frame):
    margin = 0.2
    img_h,img_w,_=np.shape(frame)
    if len(detected) > 0:
        for i, locate in enumerate(detected):
            x1, y1, x2, y2, w, h = locate.left(), locate.top(), locate.right() + 1, locate.bottom() + 1, locate.width(), locate.height()

            xw1 = max(int(x1 - margin * w), 0)
            yw1 = max(int(y1 - margin * h), 0)
            xw2 = min(int(x2 + margin * w), img_w - 1)
            yw2 = min(int(y2 + margin * h), img_h - 1)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            face = frame[yw1:yw2 + 1, xw1:xw2 + 1, :]
            cv2.putText(frame, '', (locate.left(), locate.top() - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
    return frame

def predict2Np(predict):
    # 创建68*2关键点的二维空数组[(x1,y1),(x2,y2)……]
    dims=np.zeros(shape=(predict.num_parts,2),dtype=int)
    #遍历人脸的每个关键点获取二维坐标
    length=predict.num_parts
    for i in range(0,length):
        dims[i]=(predict.part(i).x,predict.part(i).y)
    return dims



detector = dlib.get_frontal_face_detector()
criticPoints = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")



def drawCriticPoints(detected,frame):

    for (step,locate) in enumerate(detected):
        dims=criticPoints(frame,locate)
        dims=predict2Np(dims)

        for (name,(i,j)) in shape_predictor_68_face_landmark.items():
            #对每个部位进行绘点
            for (x,y) in dims[i:j]:
                cv2.circle(img=frame,center=(x,y),
                           radius=2,color=(0,255,0),thickness=-1)

        #detect yawning
        mouth_width = np.linalg.norm(dims[54] - dims[48])
        mouth_height = np.linalg.norm(dims[66] - dims[62])
        yawning_ratio = mouth_height / mouth_width
        if yawning_ratio > 0.5:
            cv2.putText(frame, 'Yawning', (locate.left(), locate.top() - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        #detect Eye close
        left_eye_ratio = get_eye_ratio(dims[42:48])
        right_eye_ratio = get_eye_ratio(dims[36:42])
        eye_ratio = (left_eye_ratio + right_eye_ratio) / 2

        if eye_ratio<0.2:
            cv2.putText(frame, 'Close Eyes', (locate.left()-20, locate.bottom() + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    return frame




def detect_time():

    cap=cv2.VideoCapture(0)
    while cap.isOpened():
        ret,frame=cap.read()
        detected = detector(frame)
        frame = drawRectangle(detected, frame)
        frame=drawCriticPoints(detected,frame)


        # Show the frame
        cv2.imshow('Distracted Driver Detection', frame)

        #cv2.imshow('frame', frame)
        key=cv2.waitKey(1)
        if key==27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    detect_time()
