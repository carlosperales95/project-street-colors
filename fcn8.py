import numpy as np
import pandas as pd
import requests
import json
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
import sklearn.model_selection
from tensorflow.python.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose
from tensorflow.python.keras.models import Model
from keras.backend.tensorflow_backend import set_session

from PIL import Image

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
tf.config.experimental.set_virtual_device_configuration(tf.config.experimental.list_physical_devices('GPU')[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10024)])
# np.set_printoptions(threshold=sys.maxsize)
# np.set_printoptions(threshold=1000)

image=[]
in_image=[]
out_image=[]
img_width = 600
img_height = 400

for i in os.listdir('./small-dset/'):
    # load the image
    image.append(Image.open('./small-dset/'+i))

    # summarize some details about the image
    #print(image.format)
    #print(image.mode)
    #print(image.size)

    # show the image
    #image.show()

    # crop image
    #width,height=image.size
    #im1=image.crop(((width-3264)/2,(height-1836)/2,(width-3264)/2+3264,(height-1836)/2+1836))
    #im1.show()

    # resize image (nearest neighbors) and divide dataset into input and desired output images
    if i[-3:] == 'jpg':
        in_image.append(np.array(image[-1].resize((img_width,img_height),Image.NEAREST)))
        #in_image[-1]=np.expand_dims(in_image[-1],axis=-1)
        #in_image[-1]=np.transpose(in_image[-1], (3, 1, 0, 2))
        in_image[-1] = np.transpose(in_image[-1], (1, 0, 2))
    else:
        out_image.append(np.array(image[-1].resize((img_width,img_height), Image.NEAREST)))
        #out_image[-1] = np.expand_dims(out_image[-1], axis=-1)
        #out_image[-1]=np.transpose(out_image[-1], (2, 1, 0))
        out_image[-1] = np.transpose(out_image[-1], (1, 0))

# np.array(out_image[0])
#out_image[-1].shape

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

in_train, in_test, out_train, out_test = sklearn.model_selection.train_test_split(in_image,out_image,test_size=0.2)

in_train = np.array([x for x in in_train])
out_train = np.array([x for x in out_train])
in_test = np.array([x for x in in_test])
out_test = np.array([x for x in out_test])


modelzero = model.fit(in_train, out_train, epochs=10, batch_size=2, validation_data=(in_test, out_test))

plt.plot(modelzero.history['accuracy'], label='accuracy')
plt.plot(modelzero.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(in_test,  out_test, verbose=2)