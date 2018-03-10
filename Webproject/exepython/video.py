import cv2
from keras.models import load_model
from h5py import *


import numpy as np
def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x
def draw_bounding_box(face_coordinates, image_array, color):
    x, y, w, h = face_coordinates
    cv2.rectangle(image_array, (x, y), (x + w, y + h), color, 2)
def draw_text(coordinates, image_array, text, color, x_offset=0, y_offset=0,
                                                font_scale=2, thickness=2):
    x, y = coordinates[:2]
    cv2.putText(image_array, text, (x + x_offset, y + y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness, cv2.LINE_AA)
#导入file视频
def emotionrecognition(file):
    emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}




    emotion_offsets = (20, 40)
    gender_offsets = (30, 60)

    detection_model = cv2.CascadeClassifier('./detection/haarcascade_frontalface_default.xml')
    emotion_classifier = load_model('./data/fer2013_mini_XCEPTION.102-0.66.hdf5', compile=False)

    emotion_target_size = emotion_classifier.input_shape[1:3]
    #导入本地视频
    video_capture = cv2.VideoCapture(file)
    l=[]
    count=video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
    # width = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    # height = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # print(width,height)
    # if width >= 512 and height >= 512:
    #     width=512
    #     height=512
    # else:
    #     pass

    for zhenshu in range(int(count)):

        flag,frame=video_capture.read()
        frame_change=cv2.resize(frame,(512,512))





        if flag:
            gray_image = cv2.cvtColor(frame_change, cv2.COLOR_BGR2GRAY)
            rgb_image = cv2.cvtColor(frame_change, cv2.COLOR_BGR2RGB)

            faces = detection_model.detectMultiScale(gray_image, 1.3, 5)
            for face_coordinates in faces:
                x, y, w, h = face_coordinates
                x0, y0 = emotion_offsets
                x3, y3 = gender_offsets
                x1 = x - x0
                x2 = x + w + x0
                y1 = y - y0
                y2 = y + h + y0
                gray_face = gray_image[y1:y2, x1:x2]
                x1 = x - x3
                x2 = x + w + x3
                y1 = y - y3
                y2 = y + h + y3

                try:
                    gray_face = cv2.resize(gray_face, (emotion_target_size))

                except:
                    continue
                gray_face = preprocess_input(gray_face, False)
                gray_face = np.expand_dims(gray_face, 0)
                gray_face = np.expand_dims(gray_face, -1)
                emotion_prediction = emotion_classifier.predict(gray_face)

                d = dict()
                for i in range(7):
                    d[emotion_labels[i]] = emotion_prediction[0][i]
                l.append(d)
                print(d)
    return l






if  __name__=='__main__':
    emotionrecognition("tmp.mp4")




