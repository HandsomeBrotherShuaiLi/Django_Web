from django.shortcuts import render
from django.shortcuts import HttpResponse
import os
from statistics import mode
import cv2
from keras.models import load_model
from h5py import *
import numpy as np
# Create your views here.
def index(request):
    return render(request,'index.html')


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


def handle_upload_file(file,filename):
    path='media/uploads/'     #上传文件的保存路径，可以自己指定任意的路径
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path+filename,'wb+')as destination:
        for chunk in file.chunks():
            destination.write(chunk)

def upload(request):
    if request.method == "POST":
        handle_upload_file(request.FILES['file'], str(request.FILES['file']))
        return HttpResponse('成功')
    return render(request,'upload.html')


