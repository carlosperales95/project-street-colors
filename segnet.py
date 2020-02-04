import numpy as np
import pandas as pd
import requests
import json
import os
import tensorflow as tf
from tensorflow import keras
import sklearn.model_selection
import matplotlib.pyplot as plt
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Dense, BatchNormalization, Activation, Reshape
from keras.models import Model
from layers import MaxPoolingWithArgmax2D, MaxUnpooling2D
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
runs = 20 #dataset subdivisions #20
train_batch = 80 #900
accuracy_hist = []
val_accuracy_hist = []
loss_hist = []
val_loss_hist = []



images =  Input(shape=(576, 384, 3))
output_1a = Conv2D(64, (3, 3),  padding='same')(images)
output_1b = BatchNormalization()(output_1a)
output_1 = Activation('relu')(output_1b)
output_2a = Conv2D(64, (3, 3),  padding='same')(output_1)
output_2b = BatchNormalization()(output_2a)
output_2 = Activation('relu')(output_2b)
output_3, argmax1 = MaxPoolingWithArgmax2D((2, 2))(output_2)
output_4a = Conv2D(128, (3, 3),  padding='same')(output_3)
output_4b = BatchNormalization()(output_4a)
output_4 = Activation('relu')(output_4b)
output_5a = Conv2D(128, (3, 3),  padding='same')(output_4)
output_5b = BatchNormalization()(output_5a)
output_5 = Activation('relu')(output_5b)
output_6, argmax2 = MaxPoolingWithArgmax2D((2, 2))(output_5)
output_7a = Conv2D(256, (3, 3),  padding='same')(output_6)
output_7b = BatchNormalization()(output_7a)
output_7 = Activation('relu')(output_7b)
output_8a = Conv2D(256, (3, 3),  padding='same')(output_7)
output_8b = BatchNormalization()(output_8a)
output_8 = Activation('relu')(output_8b)
output_9a = Conv2D(256, (3, 3),  padding='same')(output_8)
output_9b = BatchNormalization()(output_9a)
output_9 = Activation('relu')(output_9b)
output_10, argmax3 = MaxPoolingWithArgmax2D((2, 2))(output_9)
output_11a = Conv2D(512, (3, 3),  padding='same')(output_10)
output_11b = BatchNormalization()(output_11a)
output_11 = Activation('relu')(output_11b)
output_12a = Conv2D(512, (3, 3),  padding='same')(output_11)
output_12b = BatchNormalization()(output_12a)
output_12 = Activation('relu')(output_12b)
output_13a = Conv2D(512, (3, 3),  padding='same')(output_12)
output_13b = BatchNormalization()(output_13a)
output_13 = Activation('relu')(output_13b)
output_14, argmax4 = MaxPoolingWithArgmax2D((2, 2))(output_13)
output_15a = Conv2D(512, (3, 3),  padding='same')(output_14)
output_15b = BatchNormalization()(output_15a)
output_15 = Activation('relu')(output_15b)
output_16a = Conv2D(512, (3, 3),  padding='same')(output_15)
output_16b = BatchNormalization()(output_16a)
output_16 = Activation('relu')(output_16b)
output_17a = Conv2D(512, (3, 3),  padding='same')(output_16)
output_17b = BatchNormalization()(output_17a)
output_17 = Activation('relu')(output_17b)
output_18, argmax5 = MaxPoolingWithArgmax2D((2, 2))(output_17)

output_19 = MaxUnpooling2D((2,2))([output_18, argmax5])
output_20a = Conv2D(512, (3, 3),  padding='same')(output_19)
output_20b = BatchNormalization()(output_20a)
output_20 = Activation('relu')(output_20b)
output_21a = Conv2D(512, (3, 3),  padding='same')(output_20)
output_21b = BatchNormalization()(output_21a)
output_21 = Activation('relu')(output_21b)
output_22a = Conv2D(512, (3, 3),  padding='same')(output_21)
output_22b = BatchNormalization()(output_22a)
output_22 = Activation('relu')(output_22b)
output_23 = MaxUnpooling2D((2,2))([output_22, argmax4])
output_24a = Conv2D(512, (3, 3),  padding='same')(output_23)
output_24b = BatchNormalization()(output_24a)
output_24 = Activation('relu')(output_24b)
output_25a = Conv2D(512, (3, 3),  padding='same')(output_24)
output_25b = BatchNormalization()(output_25a)
output_25 = Activation('relu')(output_25b)
output_26a = Conv2D(256, (3, 3),  padding='same')(output_25)
output_26b = BatchNormalization()(output_26a)
output_26 = Activation('relu')(output_26b)
output_27 = MaxUnpooling2D((2,2))([output_26, argmax3])
output_28a = Conv2D(256, (3, 3),  padding='same')(output_27)
output_28b = BatchNormalization()(output_28a)
output_28 = Activation('relu')(output_28b)
output_29a = Conv2D(256, (3, 3),  padding='same')(output_28)
output_29b = BatchNormalization()(output_29a)
output_29 = Activation('relu')(output_29b)
output_30a = Conv2D(128, (3, 3),  padding='same')(output_29)
output_30b = BatchNormalization()(output_30a)
output_30 = Activation('relu')(output_30b)
output_31 = MaxUnpooling2D((2,2))([output_30, argmax2])
output_32a = Conv2D(128, (3, 3),  padding='same')(output_31)
output_32b = BatchNormalization()(output_32a)
output_32 = Activation('relu')(output_32b)
output_33a = Conv2D(64, (3, 3),  padding='same')(output_32)
output_33b = BatchNormalization()(output_33a)
output_33 = Activation('relu')(output_33b)
output_34 = MaxUnpooling2D((2,2))([output_33, argmax1])
output_35a = Conv2D(64, (3, 3),  padding='same')(output_34)
output_35b = BatchNormalization()(output_35a)
output_35 = Activation('relu')(output_35b)
output_36a = Conv2D(66, (3, 3),  padding='same')(output_35)
output_36b = BatchNormalization()(output_36a)
predictions = Activation('softmax')(output_36b)


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
            out_train[-1] = np.expand_dims(out_train[-1], axis=-1)
            out_train[-1] = np.transpose(out_train[-1], (1, 0, 2))

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
        in_valid[-1] = np.array(in_valid[-1].resize((img_width, img_height)))
        in_valid[-1] = np.transpose(in_valid[-1], (1, 0, 2))

    print('checkpoint 5')

    for i in os.listdir(dataset_dir + 'validation/labels/'):
        # load the image
        out_valid.append(Image.open(dataset_dir + 'validation/labels/' + i))

        # resize image (nearest neighbors) and divide dataset into input and desired output images
        out_valid[-1] = np.array(out_valid[-1].resize((img_width, img_height), Image.NEAREST))
        out_valid[-1] = np.expand_dims(out_valid[-1], axis=-1)
        out_valid[-1] = np.transpose(out_valid[-1], (1, 0, 2))

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
    in_test[-1] = np.array(in_test[-1].resize((img_width, img_height)))
    in_test[-1] = np.transpose(in_test[-1], (1, 0, 2))

for i in os.listdir(dataset_dir + 'training/labels/')[16000:]:
    # load the image
    out_test.append(Image.open(dataset_dir + 'testing/labels/' + i))

    # resize image (nearest neighbors) and divide dataset into input and desired output images
    out_test[-1] = np.array(out_test[-1].resize((img_width, img_height), Image.NEAREST))
    out_test[-1] = np.expand_dims(out_test[-1], axis=-1)
    out_test[-1] = np.transpose(out_test[-1], (1, 0, 2))

in_test = np.array([x for x in in_test])
out_test = np.array([x for x in out_test])

test_loss, test_acc = model.evaluate(in_test,  out_test, verbose=2)