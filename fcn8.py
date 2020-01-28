import numpy as np
import pandas as pd
import requests
import json
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
import sklearn.model_selection
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose
from tensorflow.python.keras.models import Model
from keras.backend.tensorflow_backend import set_session

from PIL import Image

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
# np.set_printoptions(threshold=sys.maxsize)
# np.set_printoptions(threshold=1000)

image=[]
in_train=[]
in_test=[]
in_valid=[]
out_train=[]
out_test=[]
out_valid=[]
img_width = 600
img_height = 400
train_size = 500
test_size = 100
valid_size = 100

for i in os.listdir('C:/Users/Leonardo/Documents/Leonardo-Poco importante/mapillarydb/training/images/')[0:train_size]:
    # load the image
    in_train.append(Image.open('C:/Users/Leonardo/Documents/Leonardo-Poco importante/mapillarydb/training/images/'+i))

    # resize image (nearest neighbors) and divide dataset into input and desired output images
    in_train[-1] = np.array(in_train[-1].resize((img_width,img_height),Image.NEAREST))
    in_train[-1] = np.transpose(in_train[-1], (1, 0, 2))

for i in os.listdir('C:/Users/Leonardo/Documents/Leonardo-Poco importante/mapillarydb/training/labels/')[0:train_size]:
    # load the image
    out_train.append(Image.open('C:/Users/Leonardo/Documents/Leonardo-Poco importante/mapillarydb/training/labels/'+i))

    # resize image (nearest neighbors) and divide dataset into input and desired output images
    out_train[-1] = np.array(out_train[-1].resize((img_width,img_height), Image.NEAREST))
    out_train[-1] = np.transpose(out_train[-1], (1, 0))

for i in os.listdir('C:/Users/Leonardo/Documents/Leonardo-Poco importante/mapillarydb/testing/images/')[0:test_size]:
    # load the image
    in_test.append(Image.open('C:/Users/Leonardo/Documents/Leonardo-Poco importante/mapillarydb/testing/images/'+i))

    # resize image (nearest neighbors) and divide dataset into input and desired output images
    in_test[-1] = np.array(in_test[-1].resize((img_width,img_height),Image.NEAREST))
    in_test[-1] = np.transpose(in_test[-1], (1, 0, 2))

for i in os.listdir('C:/Users/Leonardo/Documents/Leonardo-Poco importante/mapillarydb/testing/labels/')[0:test_size]:
    # load the image
    out_test.append(Image.open('C:/Users/Leonardo/Documents/Leonardo-Poco importante/mapillarydb/testing/labels/'+i))

    # resize image (nearest neighbors) and divide dataset into input and desired output images
    out_test[-1] = np.array(out_test[-1].resize((img_width,img_height), Image.NEAREST))
    out_test[-1] = np.transpose(out_test[-1], (1, 0))

for i in os.listdir('C:/Users/Leonardo/Documents/Leonardo-Poco importante/mapillarydb/validation/images/')[0:valid_size]:
    # load the image
    in_valid.append(Image.open('C:/Users/Leonardo/Documents/Leonardo-Poco importante/mapillarydb/validation/images/'+i))

    # resize image (nearest neighbors) and divide dataset into input and desired output images
    in_valid[-1] = np.array(in_valid[-1].resize((img_width,img_height),Image.NEAREST))
    in_valid[-1] = np.transpose(in_valid[-1], (1, 0, 2))


for i in os.listdir('C:/Users/Leonardo/Documents/Leonardo-Poco importante/mapillarydb/validation/labels/')[0:valid_size]:
    # load the image
    out_valid.append(Image.open('C:/Users/Leonardo/Documents/Leonardo-Poco importante/mapillarydb/validation/labels/'+i))

    # resize image (nearest neighbors) and divide dataset into input and desired output images
    out_valid[-1] = np.array(out_valid[-1].resize((img_width,img_height), Image.NEAREST))
    out_valid[-1] = np.transpose(out_valid[-1], (1, 0))


images =  Input(shape=(600, 400, 3))
output_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(images)
output_2 = Conv2D(64, (3, 3), activation='relu', padding='same')(output_1)
output_3 = MaxPooling2D((3, 2))(output_2)
output_4 = Conv2D(128, (3, 3), activation='relu', padding='same')(output_3)
output_5 = Conv2D(128, (3, 3), activation='relu', padding='same')(output_4)
output_6 = MaxPooling2D((2, 2))(output_5)
output_7 = Conv2D(256, (3, 3), activation='relu', padding='same')(output_6)
output_8 = Conv2D(256, (3, 3), activation='relu', padding='same')(output_7)
output_9 = Conv2D(256, (3, 3), activation='relu', padding='same')(output_8)
output_10 = MaxPooling2D((2, 2))(output_9)
output_11 = Conv2D(512, (3, 3), activation='relu', padding='same')(output_10)
jokerlap_a1 = MaxPooling2D((2, 2))(output_9)
jokerlap_a2 = Conv2D(66, (3, 3), activation='relu', padding='same')(jokerlap_a1)
output_12 = Conv2D(512, (3, 3), activation='relu', padding='same')(output_11)
output_13 = Conv2D(512, (3, 3), activation='relu', padding='same')(output_12)
output_14 = MaxPooling2D((2, 2))(output_13)
output_15 = Conv2D(512, (3, 3), activation='relu', padding='same')(output_14)
jokerlap_b1 = MaxPooling2D((2, 2))(output_13)
jokerlap_b2 = Conv2D(66, (3, 3), activation='relu', padding='same')(jokerlap_b1)
output_16 = Conv2D(512, (3, 3), activation='relu', padding='same')(output_15)
output_17 = Conv2D(512, (3, 3), activation='relu', padding='same')(output_16)
output_18 = MaxPooling2D((2, 2))(output_17)
output_19 = Conv2D(4096, (3, 3), activation='relu', padding='same')(output_18)
output_20 = Conv2D(4096, (3, 3), activation='relu', padding='same')(output_19)
output_21 = Conv2D(66, (3, 3), activation='relu', padding='same')(output_20)
output_22 = Conv2DTranspose(66, (3, 3), strides=(2,2), activation='relu')(output_21)
jokerlap_b3 = keras.layers.Add()([output_22, jokerlap_b2])
output_23 = Conv2DTranspose(66, (3, 3), strides=(2,2), activation='relu', padding='same')(jokerlap_b3)
jokerlap_a3 = keras.layers.Add()([output_23, jokerlap_a2])
predictions = Conv2DTranspose(66, (3, 3), strides=(12,8), activation='softmax', padding='same')(jokerlap_a3)

model = Model(inputs=images, outputs=predictions)

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

#in_train, in_test, out_train, out_test = sklearn.model_selection.train_test_split(in_image,out_image,test_size=0.2)

in_train = np.array([x for x in in_train])
out_train = np.array([x for x in out_train])
in_test = np.array([x for x in in_test])
out_test = np.array([x for x in out_test])
in_valid = np.array([x for x in in_valid])
out_valid = np.array([x for x in out_valid])

modelzero = model.fit(in_train, out_train, epochs=5, batch_size=2, validation_data=(in_valid, out_valid))

plt.plot(modelzero.history['accuracy'], label='accuracy')
plt.plot(modelzero.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.0, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(in_test,  out_test, verbose=2)