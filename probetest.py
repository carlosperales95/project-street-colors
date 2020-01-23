import numpy as np
import pandas as pd
import requests
import json
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
import sklearn.model_selection

from PIL import Image

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

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

model = models.Sequential()
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(600, 400, 3)))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((3, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(4096, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(4096, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(66, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2DTranspose(66, (3, 3), strides=(2,2), activation='relu'))
model.add(layers.Conv2DTranspose(66, (3, 3), strides=(2,2), activation='relu', padding='same'))
model.add(layers.Conv2DTranspose(66, (3, 3), strides=(12,8), activation='softmax', padding='same'))


model.summary()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

in_train, in_test, out_train, out_test = sklearn.model_selection.train_test_split(in_image,out_image,test_size=0.2)

in_train = np.array([x for x in in_train])
out_train = np.array([x for x in out_train])
in_test = np.array([x for x in in_test])
out_test = np.array([x for x in out_test])


modelzero = model.fit(in_train, out_train, epochs=10, validation_data=(in_test, out_test))

plt.plot(modelzero.history['accuracy'], label='accuracy')
plt.plot(modelzero.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(in_test,  out_test, verbose=2)
