# -*- coding: utf-8 -*-
"""img_classification.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1e5VfBMjKeLJxtoPCT8GRjp9KmRxdnSO_
"""

import keras
from PIL import Image, ImageOps
import numpy as np
import streamlit as st
import urllib.request

url = 'https://github.com/Harison-P/DL_pyx_COV19/releases/download/v1.0/fine_tuned_vgg16_second_model.h5'
modelfile = "model2.h5" 
urllib.request.urlretrieve(url, modelfile)

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
    
@st.cache(allow_output_mutation=True, show_spinner=False)
def get_model():
    model = keras.models.load_model(modelfile, custom_objects = {'f1_m' : f1_m})
    print('Model Loaded')
    return model 

def import_and_predict(img):
    # Load the model
    model = get_model()

    image = img

    # Image sizing
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    # Convert image into a numpy array
    image_array = np.asarray(image)

    if len(image_array.shape) == 3:
        normalized_image_array = (image_array.astype(np.float32)/255.) # normalize the numpy array
        normalized_image_array = normalized_image_array[np.newaxis, ...]
    else:
        image_array =  np.stack((image_array, )*3, axis = -1)
        normalized_image_array = (image_array.astype(np.float32)/255.) # normalize the numpy array
        normalized_image_array = normalized_image_array[np.newaxis, ...]

    # run the inference
    prediction = model.predict(normalized_image_array)
    return np.argmax(prediction) # return position of the highest probability