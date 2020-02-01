import numpy as np
import pandas as pd
import requests
import json
import os
import tensorflow as tf
from tensorflow import keras
import sklearn.model_selection
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Dense
from tensorflow.python.keras.models import Model
#from sklearn.metrics import roc_auc_score

from PIL import Image

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

#def auroc(y_true, y_pred):
#    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)

dataset_dir = 'C:/Users/Leonardo/Documents/Leonardo-Poco importante/mapillarydb/'
img_width = 600
img_height = 400
n_epochs = 100
runs = 20 #dataset subdivisions #20
train_batch = 800 #900
accuracy_hist = []
val_accuracy_hist = []
loss_hist = []
val_loss_hist = []


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
output_19 = Dense(4096, activation='relu')(output_18)
output_20 = Dense(4096, activation='relu')(output_19)
output_21 = Conv2D(66, (3, 3), activation='relu', padding='same')(output_20)
output_22 = Conv2DTranspose(66, (3, 3), strides=(2,2), activation='relu')(output_21)
jokerlap_b3 = keras.layers.Add()([output_22, jokerlap_b2])
output_23 = Conv2DTranspose(66, (3, 3), strides=(2,2), activation='relu', padding='same')(jokerlap_b3)
jokerlap_a3 = keras.layers.Add()([output_23, jokerlap_a2])
predictions = Conv2DTranspose(66, (3, 3), strides=(12,8), activation='softmax', padding='same')(jokerlap_a3)

model = Model(inputs=images, outputs=predictions)

#sparse categorical is ok for output where 1 class is true and the others are false
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.summary()

for epochs in range(n_epochs):
    in_valid = []
    out_valid = []
    for r in range(runs):
        in_train=[]
        out_train=[]
        train_size = train_batch*(r+1)
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

        print('checkpoint 3')
        #in_train, in_test, out_train, out_test = sklearn.model_selection.train_test_split(in_image,out_image,test_size=0.2)

        in_train = np.array([x for x in in_train])
        out_train = np.array([x for x in out_train])

        modelzero = model.fit(in_train, out_train, epochs=1, batch_size=2)

        print('checkpoint 4')

    for i in os.listdir(dataset_dir + 'validation/images/'):
        # load the image
        in_valid.append(Image.open(dataset_dir + 'validation/images/' + i))

        # resize image (nearest neighbors) and divide dataset into input and desired output images
        in_valid[-1] = np.array(in_valid[-1].resize((img_width, img_height), Image.NEAREST))
        in_valid[-1] = np.transpose(in_valid[-1], (1, 0, 2))

    print('checkpoint 5')

    for i in os.listdir(dataset_dir + 'validation/labels/'):
        # load the image
        out_valid.append(Image.open(dataset_dir + 'validation/labels/' + i))

        # resize image (nearest neighbors) and divide dataset into input and desired output images
        out_valid[-1] = np.array(out_valid[-1].resize((img_width, img_height), Image.NEAREST))
        out_valid[-1] = np.transpose(out_valid[-1], (1, 0))

    print('checkpoint 6')

    in_valid = np.array([x for x in in_valid])
    out_valid = np.array([x for x in out_valid])

    val_loss, val_accuracy = model.evaluate(in_valid, out_valid, batch_size=1, verbose=1)

    accuracy_hist += modelzero.history['accuracy']
    loss_hist += modelzero.history['loss']
    #valaccuracy_hist += modelzero.history['val_accuracy']
    val_accuracy_hist += [val_accuracy]
    val_loss_hist += [val_loss]

    if val_loss==min(val_loss_hist):
        model.save_weights('fcn8.h5')


plt.plot(accuracy_hist, label='accuracy')
plt.plot(valaccuracy_hist, label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
#plt.ylim([0.0, 1])
plt.legend(loc='lower right')

in_train = []
out_train = []
in_valid = []
out_valid = []
in_test = []
out_test = []

for i in os.listdir(dataset_dir + 'training/images/')[16000:]:
    # load the image
    in_test.append(Image.open(dataset_dir + 'testing/images/' + i))

    # resize image (nearest neighbors) and divide dataset into input and desired output images
    in_test[-1] = np.array(in_test[-1].resize((img_width, img_height), Image.NEAREST))
    in_test[-1] = np.transpose(in_test[-1], (1, 0, 2))

for i in os.listdir(dataset_dir + 'training/labels/')[16000:]:
    # load the image
    out_test.append(Image.open(dataset_dir + 'testing/labels/' + i))

    # resize image (nearest neighbors) and divide dataset into input and desired output images
    out_test[-1] = np.array(out_test[-1].resize((img_width, img_height), Image.NEAREST))
    out_test[-1] = np.transpose(out_test[-1], (1, 0))

in_test = np.array([x for x in in_test])
out_test = np.array([x for x in out_test])

test_loss, test_acc = model.evaluate(in_test,  out_test, verbose=2)