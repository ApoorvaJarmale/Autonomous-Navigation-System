import glob
import numpy as np
from collections import deque
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from skimage.exposure import rescale_intensity

import argparse
import base64
import json
import cv2
import csv
import random
import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
import tensorflow as tf

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation,Lambda
from keras.layers import Input, ELU
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras.layers import Convolution2D, MaxPooling2D, Flatten
from keras import initializers
from keras.utils import multi_gpu_model



## Defining variables
pr_threshold = 1
new_size_col = 92
new_size_row = 69

def preprocess(image):
    # get shape and chop off 1/3 from the top
    shape = image.shape
    #print("shape: " + str(shape))
    # note: numpy arrays are (row, col)!
    #image = image[shape[0]//4:shape[0]-25, 0:shape[1]]
    #image = cv2.resize(image, (200,66), interpolation=cv2.INTER_AREA)
    image = cv2.resize(image, (new_size_col,new_size_row), interpolation=cv2.INTER_AREA)
    return image


def get_model():
    # model start here
    input_shape = (new_size_row, new_size_col, 3)
    filter_size = 3
    pool_size = (2,2)
    model = Sequential()
    model.add(Lambda(lambda x: x/255.-0.5,input_shape=input_shape))

    #model.add(BatchNormalization(epsilon=0.001,mode=2, axis=1)

    model.add(Convolution2D(24,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
    model.add(Convolution2D(36,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
    model.add(Convolution2D(48,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
    model.add(Convolution2D(64,3,3,border_mode='valid', activation='relu', subsample=(1,1)))
    model.add(Convolution2D(64,3,3,border_mode='valid', activation='relu', subsample=(1,1)))
    model.add(Flatten())
    model.add(Dense(1164, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    return model



if __name__ == "__main__":        
    #filenames = glob.glob("imgs/*.jpg")

    model = get_model()
    model.compile("adam", "mse")
    weights_file = 'model_best.h5'
    model.load_weights(weights_file)
    with open('train.csv') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            file_name = str(row[0])
            image_pre = cv2.imread(file_name)
            image_array = preprocess(image_pre)
            transformed_image_array = image_array[None, :, :, :]
            steering_angle = 1.0 * float(model.predict(transformed_image_array, batch_size=10))
            #steering_angle = 0.7*float(row[1]) + 0.6*float(steering_angle) +0.15*(0.001)*random.randint(0,100)
            print(steering_angle)
            with open('test_steering_for_visualization.csv', 'a') as csvfile:
                fieldnames = ['center','CNN','predicted']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                # writer.writeheader()
                writer.writerow({'center': str(row[0]), 'CNN': str(row[2]),'predicted': str(steering_angle)})
