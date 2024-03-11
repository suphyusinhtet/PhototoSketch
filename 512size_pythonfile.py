# -*- coding: utf-8 -*-
"""512size_pythonfile.py

This is AI project made by project members
Su Phyu Sin Htet (Leader)
Hein Thu Aung
Aye Chan Phyo

"""

#Import Libraries

# import numpy as np
# import tensorflow as tf
# import keras
# from keras.layers import Dense, Conv2D, MaxPool2D, UpSampling2D, Dropout, Input
# from keras.preprocessing.image import img_to_array
# import matplotlib.pyplot as plt
# import cv2
# from tqdm import tqdm
# import os
# import re

# #Load Data

# # to get the files in proper order
# def sorted_alphanumeric(data):
#     convert = lambda text: int(text) if text.isdigit() else text.lower()
#     alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)',key)]
#     return sorted(data,key = alphanum_key)


# # defining the size of image
# SIZE = 512

# image_path = '/content/drive/MyDrive/Dataset for Project/photos'
# img_array = []

# sketch_path = '/content/drive/MyDrive/Dataset for Project/sketches'
# sketch_array = []

# image_file = sorted_alphanumeric(os.listdir(image_path))
# sketch_file = sorted_alphanumeric(os.listdir(sketch_path))


# for i in tqdm(image_file):
#     image = cv2.imread(image_path + '/' + i,1)

#     # as opencv load image in bgr format converting it to rgb
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     # resizing images
#     image = cv2.resize(image, (SIZE, SIZE))

#     # normalizing image
#     image = image.astype('float32') / 255.0

#     #appending normal normal image
#     img_array.append(img_to_array(image))
#     # Image Augmentation

#     # horizontal flip
#     img1 = cv2.flip(image,1)
#     img_array.append(img_to_array(img1))
#      #vertical flip
#     img2 = cv2.flip(image,-1)
#     img_array.append(img_to_array(img2))
#      #vertical flip
#     img3 = cv2.flip(image,-1)
#     # horizontal flip
#     img3 = cv2.flip(img3,1)
#     img_array.append(img_to_array(img3))
#     #rotate clockwise
#     img4 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
#     img_array.append(img_to_array(img4))
#     # flip rotated image
#     img5 = cv2.flip(img4,1)
#     img_array.append(img_to_array(img5))
#      # rotate anti clockwise
#     img6 = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
#     img_array.append(img_to_array(img6))
#     # flip rotated image
#     img7 = cv2.flip(img6,1)
#     img_array.append(img_to_array(img7))


# for i in tqdm(sketch_file):
#     image = cv2.imread(sketch_path + '/' + i,1)

#     # as opencv load image in bgr format converting it to rgb
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     # resizing images
#     image = cv2.resize(image, (SIZE, SIZE))

#     # normalizing image
#     image = image.astype('float32') / 255.0
#     # appending normal sketch image
#     sketch_array.append(img_to_array(image))

#     #Image Augmentation
#     # horizontal flip
#     img1 = cv2.flip(image,1)
#     sketch_array.append(img_to_array(img1))
#      #vertical flip
#     img2 = cv2.flip(image,-1)
#     sketch_array.append(img_to_array(img2))
#      #vertical flip
#     img3 = cv2.flip(image,-1)
#     # horizontal flip
#     img3 = cv2.flip(img3,1)
#     sketch_array.append(img_to_array(img3))
#     #rotate clockwise
#     img4 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
#     sketch_array.append(img_to_array(img4))
#     # flip rotated image
#     img5 = cv2.flip(img4,1)
#     sketch_array.append(img_to_array(img5))
#      # rotate anti clockwise
#     img6 = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
#     sketch_array.append(img_to_array(img6))
#     # flip rotated image
#     img7 = cv2.flip(img6,1)
#     sketch_array.append(img_to_array(img7))

# print("Total number of sketch images:",len(sketch_array))
# print("Total number of images:",len(img_array))

# #Visualizing Images

# # defining function to plot images pair
# def plot_images(image, sketches):
#     plt.figure(figsize=(7,7))
#     plt.subplot(1,2,1)
#     plt.title('Image', color = 'green', fontsize = 20)
#     plt.imshow(image)
#     plt.subplot(1,2,2)
#     plt.title('Sketches ', color = 'black', fontsize = 20)
#     plt.imshow(sketches)

#     plt.show()

# ls = [i for i in range(0,65,8)]
# for i in ls:
#     plot_images(img_array[i],sketch_array[i])

# #Split Training & Testing & Reshape

# train_sketch_image = sketch_array[:320]
# train_image = img_array[:320]
# test_sketch_image = sketch_array[320:]
# test_image = img_array[320:]
# # reshaping
# train_sketch_image = np.reshape(train_sketch_image,(len(train_sketch_image),SIZE,SIZE,3))
# train_image = np.reshape(train_image, (len(train_image),SIZE,SIZE,3))
# print('Train color image shape:',train_image.shape)
# test_sketch_image = np.reshape(test_sketch_image,(len(test_sketch_image),SIZE,SIZE,3))
# test_image = np.reshape(test_image, (len(test_image),SIZE,SIZE,3))
# print('Test color image shape',test_image.shape)

# #Downsample Layer

# def downsample(filters, size, apply_batch_normalization = True):
#     downsample = tf.keras.models.Sequential()
#     downsample.add(keras.layers.Conv2D(filters = filters, kernel_size = size, strides = 2, use_bias = False, kernel_initializer = 'he_normal'))
#     if apply_batch_normalization:
#         downsample.add(keras.layers.BatchNormalization())
#     downsample.add(keras.layers.LeakyReLU())
#     return downsample

# #Upsample Layer
# def upsample(filters, size, apply_dropout = False):
#     upsample = tf.keras.models.Sequential()
#     upsample.add(keras.layers.Conv2DTranspose(filters = filters, kernel_size = size, strides = 2, use_bias = False, kernel_initializer = 'he_normal'))
#     if apply_dropout:
#         upsample.add(tf.keras.layers.Dropout(0.1, trainable=True))
#     upsample.add(tf.keras.layers.LeakyReLU())
#     return upsample

# #Model

# def model():
#     encoder_input = keras.Input(shape = (SIZE, SIZE, 3))
#     x = downsample(16, 4, False)(encoder_input)
#     x = downsample(32,4)(x)
#     encoder_output = downsample(64,4)(x)

#     decoder_input = upsample(64,4,False)(encoder_output)
#     x = upsample(32,4)(decoder_input)
#     x = upsample(16,4)(x)

#     x = tf.keras.layers.Conv2DTranspose(8,(2,2),strides = (1,1), padding = 'valid', activation="relu")(x)
#     decoder_output = tf.keras.layers.Conv2DTranspose(3,(2,2),strides = (1,1), padding = 'valid', activation="sigmoid")(x)


#     return tf.keras.Model(encoder_input, decoder_output)

# # to get summary of model
# model = model()
# model.summary()

# #Compiling and Fitting our model (Adam)

# model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001), loss = 'mean_absolute_error',
#               metrics = ['acc'])

# # Set up early stopping callback
# early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)

# history = model.fit(train_image, train_sketch_image, epochs = 100, verbose = 0,callbacks=[early_stopping])

# #Evaluating our model

# prediction_on_test_data = model.evaluate(test_image, test_sketch_image)
# print("Loss: ", prediction_on_test_data[0])
# print("Accuracy: ", np.round(prediction_on_test_data[1] * 100,1))

# #Plotting our predicted sketch along with real sketch

# def show_images(real,sketch, predicted):
#     plt.figure(figsize = (12,12))
#     plt.subplot(1,3,1)
#     plt.title("Image",fontsize = 15, color = 'Lime')
#     plt.imshow(real)
#     plt.subplot(1,3,2)
#     plt.title("sketch",fontsize = 15, color = 'Blue')
#     plt.imshow(sketch)
#     plt.subplot(1,3,3)
#     plt.title("Predicted",fontsize = 15, color = 'gold')
#     plt.imshow(predicted)
# ls = [i for i in range(0,80,8)]
# for i in ls:
#     predicted =np.clip(model.predict(test_image[i].reshape(1,SIZE,SIZE,3)),0.0,1.0).reshape(SIZE,SIZE,3)
#     show_images(test_image[i],test_sketch_image[i],predicted)

# model.save('/new50data_test_3_with_512size.h5')

# saved_model = tf.keras.models.load_model('/content/new50data_test_3_with_512size.h5')

# #Load Data

# # to get the files in proper order
# def sorted_alphanumeric(data):
#     convert = lambda text: int(text) if text.isdigit() else text.lower()
#     alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)',key)]
#     return sorted(data,key = alphanum_key)


# # defining the size of image
# SIZE = 512

# image_path = '/content/drive/MyDrive/NewTestDataset/photos'
# img_array = []


# image_file = sorted_alphanumeric(os.listdir(image_path))
# # sketch_file = sorted_alphanumeric(os.listdir(sketch_path))

# for i in tqdm(image_file):
#     image = cv2.imread(image_path + '/' + i,1)

#     # as opencv load image in bgr format converting it to rgb
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     # resizing images
#     image = cv2.resize(image, (SIZE, SIZE))

#     # normalizing image
#     image = image.astype('float32') / 255.0
#     img_array.append(img_to_array(image))

# # defining function to plot images pair
# def plot_images(image):
#     plt.figure(figsize=(7,7))
#     plt.subplot(1,2,1)
#     plt.title('Image', color = 'green', fontsize = 20)
#     plt.imshow(image)
#     # plt.subplot(1,2,2)
#     # plt.title('Sketches ', color = 'black', fontsize = 20)
#     # plt.imshow(sketches)

#     plt.show()

# ls = [i for i in range(0,29,3)]
# for i in ls:
#     plot_images(img_array[i])

# new_test_image = np.reshape(img_array, (len(img_array),SIZE,SIZE,3))
# print("Total number of images:",len(new_test_image))

# def show_images(real, predicted):
#     plt.figure(figsize = (12,12))
#     plt.subplot(1,3,1)
#     plt.title("Image",fontsize = 15, color = 'Lime')
#     plt.imshow(real)
#     # plt.subplot(1,3,2)
#     # plt.title("sketch",fontsize = 15, color = 'Blue')
#     # plt.imshow(sketch)
#     plt.subplot(1,3,2)
#     plt.title("Predicted",fontsize = 15, color = 'gold')
#     plt.imshow(predicted)

# ls = [i for i in range(0,29)]
# for i in ls:
#     predicted =np.clip(saved_model.predict(new_test_image[i].reshape(1,SIZE,SIZE,3)),0.0,1.0).reshape(SIZE,SIZE,3)
#     show_images(new_test_image[i],predicted)
