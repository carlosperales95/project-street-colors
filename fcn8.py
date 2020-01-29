import numpy as np
import pandas as pd
import requests
import json
import os
import tensorflow as tf
from tensorflow import keras
import sklearn.model_selection
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose
from tensorflow.python.keras.models import Model
from keras.backend.tensorflow_backend import set_session

from PIL import Image

#gpus = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(gpus[0], True)
# np.set_printoptions(threshold=sys.maxsize)
# np.set_printoptions(threshold=1000)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4500)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

dataset_dir = 'C:/Users/Leonardo/Documents/Leonardo-Poco importante/mapillarydb/'
img_width = 600
img_height = 400
n_epochs = 1
runs = 20 #dataset subdivisions
train_batch = 900
test_batch = 1 #zero makes the for loop crash
valid_batch = 100
accuracy_hist = []
valaccuracy_hist = []

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

#sparse categorical is ok for output where 1 class is true and the others are false
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

for epochs in range(n_epochs):
    for r in range(runs):
        in_train=[]
        in_test=[]
        in_valid=[]
        out_train=[]
        out_test=[]
        out_valid=[]
        train_size = train_batch*(r+1)
        test_size = test_batch*(r+1)
        valid_size = valid_batch*(r+1)
        print('checkpoint 1')

        for i in os.listdir(dataset_dir+'training/images/')[train_size-train_batch:train_size]:
            # load the image
            in_train.append(Image.open(dataset_dir+'training/images/'+i))

            # resize image (nearest neighbors) and divide dataset into input and desired output images
            in_train[-1] = np.array(in_train[-1].resize((img_width,img_height),Image.NEAREST))
            in_train[-1] = np.transpose(in_train[-1], (1, 0, 2))

        print('checkpoint 2')

        for i in os.listdir(dataset_dir+'training/labels/')[train_size-train_batch:train_size]:
            # load the image
            out_train.append(Image.open(dataset_dir+'training/labels/'+i))

            # resize image (nearest neighbors) and divide dataset into input and desired output images
            out_train[-1] = np.array(out_train[-1].resize((img_width,img_height), Image.NEAREST))
            out_train[-1] = np.transpose(out_train[-1], (1, 0))

        for i in os.listdir(dataset_dir+'testing/images/')[test_size-test_batch:test_size]:
            # load the image
            in_test.append(Image.open(dataset_dir+'testing/images/'+i))

            # resize image (nearest neighbors) and divide dataset into input and desired output images
            in_test[-1] = np.array(in_test[-1].resize((img_width,img_height),Image.NEAREST))
            in_test[-1] = np.transpose(in_test[-1], (1, 0, 2))

        for i in os.listdir(dataset_dir+'testing/labels/')[test_size-test_batch:test_size]:
            # load the image
            out_test.append(Image.open(dataset_dir+'testing/labels/'+i))

            # resize image (nearest neighbors) and divide dataset into input and desired output images
            out_test[-1] = np.array(out_test[-1].resize((img_width,img_height), Image.NEAREST))
            out_test[-1] = np.transpose(out_test[-1], (1, 0))

        for i in os.listdir(dataset_dir+'validation/images/')[valid_size-valid_batch:valid_size]:
            # load the image
            in_valid.append(Image.open(dataset_dir+'validation/images/'+i))

            # resize image (nearest neighbors) and divide dataset into input and desired output images
            in_valid[-1] = np.array(in_valid[-1].resize((img_width,img_height),Image.NEAREST))
            in_valid[-1] = np.transpose(in_valid[-1], (1, 0, 2))


        for i in os.listdir(dataset_dir+'validation/labels/')[valid_size-valid_batch:valid_size]:
            # load the image
            out_valid.append(Image.open(dataset_dir+'validation/labels/'+i))

            # resize image (nearest neighbors) and divide dataset into input and desired output images
            out_valid[-1] = np.array(out_valid[-1].resize((img_width,img_height), Image.NEAREST))
            out_valid[-1] = np.transpose(out_valid[-1], (1, 0))

        print('checkpoint 3')
        #in_train, in_test, out_train, out_test = sklearn.model_selection.train_test_split(in_image,out_image,test_size=0.2)

        in_train = np.array([x for x in in_train])
        out_train = np.array([x for x in out_train])
        in_test = np.array([x for x in in_test])
        out_test = np.array([x for x in out_test])
        in_valid = np.array([x for x in in_valid])
        out_valid = np.array([x for x in out_valid])

        modelzero = model.fit(in_train, out_train, epochs=1, batch_size=2, validation_data=(in_valid, out_valid))

        print('checkpoint 4')

    accuracy_hist += modelzero.history['accuracy']
    valaccuracy_hist += modelzero.history['val_accuracy']

plt.plot(accuracy_hist, label='accuracy')
plt.plot(valaccuracy_hist, label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
#plt.ylim([0.0, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(in_test,  out_test, verbose=2)