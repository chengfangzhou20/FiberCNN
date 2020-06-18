from django.shortcuts import render
from django.http import HttpResponseRedirect
from django.core.urlresolvers import reverse

from mlapp.models import PicUpload
from mlapp.forms import ImageForm
from PIL import Image
# Create your views here.
def index(request):
    return render(request,'index.html')

def method(request):
    return render(request,'method.html')

def list(request):
    image_path = ''
    image_path1 = ''
    if request.method == "POST":
        form = ImageForm(request.POST, request.FILES)

        if form.is_valid():
            newdoc = PicUpload(imagefile=request.FILES['imagefile'])
            newdoc.save()

            return HttpResponseRedirect(reverse('list'))

    else:
        form = ImageForm()

    documents = PicUpload.objects.all()
    for document in documents:
        image_path = document.imagefile.name
        image_path1 = '/'+image_path

        im = Image.open(image_path)
        im.thumbnail(im.size)

        if im.mode != 'RGB':
            im = im.convert('RGB')

        im.save('static/tempory.jpg', "JPEG", quality=100)
        image_path1 = 'static/tempory.jpg'
        document.delete()

    request.session['image_path'] = image_path


    return render(request,'list.html',
    {'documents':documents,'image_path':image_path1, 'form':form}
    )



#***************Fiber recruitment dectection*******************

import os
import json

import h5py
from PIL import Image

# keras imports
import numpy as np
import pandas as pd
import tensorflow as tf
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout, GlobalMaxPooling2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import VGG16,MobileNet,InceptionV3,ResNet50,Xception,InceptionResNetV2
from sklearn.model_selection import KFold

from tensorflow.keras import backend as K
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import cv2
#***************Prepare Image for processing*******************
# Define the function convert RGB to grayscale
def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

def prepare_img_224(img_path):
    # convert image to np array
    im = Image.open(img_path)
    imrgb = np.array(im)
    # convert image to gray scale
    imgray = rgb2gray(imrgb)
    # if the image size is larger than 224 X 224, then resize the image
    imgray = cv2.resize(imgray,(224,224),interpolation = cv2.INTER_AREA)
    # normalize the gray scale image by dividing the maximum pixel value
    imgray = imgray/float(np.max(imgray))
    # Convert gray scale image to 'rgb' by adding two dummy layers
    x_data = np.zeros((1,224,224,3))

    for i in range(3):
        x_data[0,:,:,i] = imgray
    return x_data

global graph
graph = tf.compat.v1.get_default_graph()

def get_predict(img_224):
    model = load_model('static/model.h5')
    #output = model.predict(img_224).argmax(axis =1)
    output = model.predict(img_224)
    return output


#*************** ENGINE *******************

def engine(request):
    MPM = request.session['image_path']
    img_path = MPM
    request.session.pop('image_path',None)
    request.session.modified = True
    with graph.as_default():

        img_224 = prepare_img_224(img_path)
        output = get_predict(img_224)

        p0 = output[0][0]
        p1 = output[0][1]
        p2 = output[0][2]

        class_num = output.argmax(axis =1)
        if class_num == 0:
            prediction = 'The fiber recruitment percentage is about 0 ~ 20 %'
        elif class_num == 1:
            prediction = 'The fiber recruitment percentage is about 20 ~ 50 %'
        elif class_num ==2:
            prediction = 'The fiber recruitment percentage is about 50 ~ 100 %'




    src = 'pic_upload/'
    import os
    for image_file_name in os.listdir(src):
        if image_file_name.endswith(".tif"):
            os.remove(src +image_file_name)

    K.clear_session()


    return render(request,'results.html',context={'prediction':prediction,'p0':p0,'p1':p1,'p2':p2})
