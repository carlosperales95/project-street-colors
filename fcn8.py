import numpy as np
import pandas as pd
import requests
import json
import os
import tensorflow as tf
from tensorflow import keras
import sklearn.model_selection
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Dense, BatchNormalization, Activation
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
img_width = 576
img_height = 384
n_epochs = 100
runs = 40 #dataset subdivisions #20
train_batch = 400 #900
accuracy_hist = []
val_accuracy_hist = []
loss_hist = []
val_loss_hist = []


images =  Input(shape=(img_width, img_height, 3))
output_1a = Conv2D(64, (3, 3),  padding='same')(images)
output_1b = InstanceNormalization()(output_1a)
output_1 = Activation('relu')(output_1b)
output_2a = Conv2D(64, (3, 3),  padding='same')(output_1)
output_2b = InstanceNormalization()(output_2a)
output_2 = Activation('relu')(output_2b)
output_3 = MaxPooling2D((2, 2))(output_2)
output_4a = Conv2D(128, (3, 3),  padding='same')(output_3)
output_4b = InstanceNormalization()(output_4a)
output_4 = Activation('relu')(output_4b)
output_5a = Conv2D(128, (3, 3),  padding='same')(output_4)
output_5b = InstanceNormalization()(output_5a)
output_5 = Activation('relu')(output_5b)
output_6 = MaxPooling2D((2, 2))(output_5)
output_7a = Conv2D(256, (3, 3),  padding='same')(output_6)
output_7b = InstanceNormalization()(output_7a)
output_7 = Activation('relu')(output_7b)
output_8a = Conv2D(256, (3, 3),  padding='same')(output_7)
output_8b = InstanceNormalization()(output_8a)
output_8 = Activation('relu')(output_8b)
output_9a = Conv2D(256, (3, 3),  padding='same')(output_8)
output_9b = InstanceNormalization()(output_9a)
output_9 = Activation('relu')(output_9b)
output_10 = MaxPooling2D((2, 2))(output_9)
output_11a = Conv2D(512, (3, 3),  padding='same')(output_10)
output_11b = InstanceNormalization()(output_11a)
output_11 = Activation('relu')(output_11b)
jokerlap_a1 = MaxPooling2D((2, 2))(output_9)
jokerlap_a2a = Conv2D(66, (3, 3),  padding='same')(jokerlap_a1)
jokerlap_a2b = InstanceNormalization()(jokerlap_a2a)
jokerlap_a2 = Activation('relu')(jokerlap_a2b)
output_12a = Conv2D(512, (3, 3),  padding='same')(output_11)
output_12b = InstanceNormalization()(output_12a)
output_12 = Activation('relu')(output_12b)
output_13a = Conv2D(512, (3, 3),  padding='same')(output_12)
output_13b = InstanceNormalization()(output_13a)
output_13 = Activation('relu')(output_13b)
output_14 = MaxPooling2D((2, 2))(output_13)
output_15a = Conv2D(512, (3, 3),  padding='same')(output_14)
output_15b = InstanceNormalization()(output_15a)
output_15 = Activation('relu')(output_15b)
jokerlap_b1 = MaxPooling2D((2, 2))(output_13)
jokerlap_b2a = Conv2D(66, (3, 3),  padding='same')(jokerlap_b1)
jokerlap_b2b = InstanceNormalization()(jokerlap_b2a)
jokerlap_b2 = Activation('relu')(jokerlap_b2b)
output_16a = Conv2D(512, (3, 3),  padding='same')(output_15)
output_16b = InstanceNormalization()(output_16a)
output_16 = Activation('relu')(output_16b)
output_17a = Conv2D(512, (3, 3),  padding='same')(output_16)
output_17b = InstanceNormalization()(output_17a)
output_17 = Activation('relu')(output_17b)
output_18 = MaxPooling2D((2, 2))(output_17)
output_19a = Dense(4096, activation='relu')(output_18)
output_19b = InstanceNormalization()(output_19a)
output_19 = Activation('relu')(output_19b)
output_20a = Dense(4096, activation='relu')(output_19)
output_20b = InstanceNormalization()(output_20a)
output_20 = Activation('relu')(output_20b)
output_21a = Dense(66, activation='relu')(output_20)
output_21b = InstanceNormalization()(output_21a)
output_21 = Activation('relu')(output_21b)
output_22a = Conv2DTranspose(66, (3, 3), strides=(2,2), padding='same')(output_21)
output_22b = InstanceNormalization()(output_22a)
output_22 = Activation('relu')(output_22b)
jokerlap_b3 = keras.layers.Add()([output_22, jokerlap_b2])
output_23a = Conv2DTranspose(66, (3, 3), strides=(2,2),  padding='same')(jokerlap_b3)
output_23b = InstanceNormalization()(output_23a)
output_23 = Activation('relu')(output_23b)
jokerlap_a3 = keras.layers.Add()([output_23, jokerlap_a2])
predictions = Conv2DTranspose(66, (3, 3), strides=(8,8), activation='softmax', padding='same')(jokerlap_a3)

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
            in_train[-1] = np.array(in_train[-1].resize((img_width,img_height)))
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

        modelzero = model.fit(in_train, out_train, epochs=1, batch_size=1)

        print('checkpoint 4')

    for i in os.listdir(dataset_dir + 'validation/images/'):
        # load the image
        in_valid.append(Image.open(dataset_dir + 'validation/images/' + i))

        # resize image (nearest neighbors) and divide dataset into input and desired output images
        in_valid[-1] = np.array(in_valid[-1].resize((img_width, img_height)))
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
        model.save_weights('fcn8bn2.h5')


plt.plot(accuracy_hist, label='accuracy')
plt.plot(val_accuracy_hist, label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
#plt.ylim([0.0, 1])
plt.legend(loc='lower right')

plt.plot(loss_hist, label='loss')
plt.plot(val_loss_hist, label = 'val_loss')
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
    in_test.append(Image.open(dataset_dir + 'training/images/' + i))

    # resize image (nearest neighbors) and divide dataset into input and desired output images
    in_test[-1] = np.array(in_test[-1].resize((img_width, img_height)))
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

############################

for i in os.listdir(dataset_dir + 'training/images/'):
    # load the image
    in_test=Image.open(dataset_dir + 'training/images/' + i)
    # resize image (nearest neighbors) and divide dataset into input and desired output images
    in_test = np.array(in_test.resize((img_width, img_height)))
    in_test = Image.fromarray(in_test, 'RGB')
    in_test.save(dataset_dir + 'training/images_lin/'+i)

    for i in os.listdir(dataset_dir + 'training/labels/'):
        # load the image
        in_test = Image.open(dataset_dir + 'training/labels/' + i)
        # resize image (nearest neighbors) and divide dataset into input and desired output images
        in_test = np.array(in_test.resize((img_width, img_height)))
        in_test = Image.fromarray(in_test, 'RGB')
        in_test.save(dataset_dir + 'training/labels_lin/' + i)