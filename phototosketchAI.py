#Import Libraries

import numpy as np
import tensorflow as tf
import keras
from keras.layers import Dense, Conv2D, MaxPool2D, UpSampling2D, Dropout, Input
from keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import os
import re
import streamlit as st
from PIL import Image
import io

#Showing Image
def show_image(real,predict):
    col1, col2 = st.columns(2)
    with col1:
        st.image(real, caption="Uploaded Image.")
    with col2:
        st.image(predict, caption="Sketch Image")


def choosing_size(x,upload_image):

    if upload_image is not None:

        image_data = np.asarray(bytearray(upload_image.read()), dtype=np.uint8)
        upload_image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

        upload_image = cv2.cvtColor(upload_image, cv2.COLOR_BGR2RGB)

        upload_image = cv2.resize(upload_image, (SIZE, SIZE))

        upload_image = upload_image.astype('float32') / 255.0

        predict =np.clip(x.predict(upload_image.reshape(1,SIZE,SIZE,3)),0.0,1.0).reshape(SIZE,SIZE,3)

        show_image(upload_image,predict)

        image_bytes = Image.fromarray((predict * 255).astype(np.uint8)).convert("RGB")
        image_byte_array = io.BytesIO()
        image_bytes.save(image_byte_array, format="PNG")
        image_byte_array.seek(0)

        st.download_button(
            label="Download Sketch Image",
            key="download_generated_image_link",
            data=image_byte_array,
            file_name="generated_image.png",
            mime="image/png",
    )
        
small_model = tf.keras.models.load_model('AI Final Project/256size.h5')
large_model = tf.keras.models.load_model('AI Final Project/512size.h5')

st.write("Our application has some limitations: it works better on portrait face images with clear background.")

upload_image =  st.file_uploader("Choose an image...",type=["jpg", "jpeg", "png"])

choice = st.radio(
        "Please select one of the sizes for your photo ",
        ["small size (lower quality)", "big size (better quality)"],
        index = None
        )

if choice == "small size (lower quality)" :
    SIZE = 256
    x = small_model
    choosing_size(x,upload_image)
elif choice == "big size (better quality)":
    SIZE = 512
    x = large_model
    choosing_size(x,upload_image)











