import numpy as np
import pandas as pd
import requests
import json
import os
import tensorflow as tf
from tensorflow import keras
import sklearn.model_selection
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Dense, BatchNormalization, Activation, Reshape
from tensorflow.keras.models import Model
from keras import optimizers
from layers import GroupNormalization, InstanceNormalization
#from layers import MaxPoolingWithArgmax2D, MaxUnpooling2D
from tensorflow.compat.v1.nn import max_pool_with_argmax as MaxPoolingWithArgmax2D
import pickle
import random
#from sklearn.metrics import roc_auc_score
from PIL import Image

# WINDOWS
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
#  try:
#    tf.config.experimental.set_virtual_device_configuration(
#        gpus[0],
#        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4500)])
#    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#  except RuntimeError as e:
#    # Virtual devices must be set before GPUs have been initialized
#    print(e)


tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)

#def auroc(y_true, y_pred):
#    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)

dataset_dir = '/home/leonardo/Documenti/mapillarydb/'
img_width = 288 #576
img_height = 192 #384
n_epochs = 10
runs = 20 #dataset subdivisions #40
train_batch = 800 #400
accuracy_hist = []
val_accuracy_hist = []
loss_hist = []
val_loss_hist = []
test_accuracy_hist = []
test_loss_hist = []
in_valid = []
out_valid = []

trainingset = [os.path.splitext(filename)[0] for filename in sorted(os.listdir(dataset_dir+'training/images_lin/'))[:16000]]

def MaxUnpooling2D(pool, ind, ksize=[1, 2, 2, 1], name=None):
    with tf.compat.v1.variable_scope('name') as scope:
        input_shape = tf.shape(pool)
        output_shape = [input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3]]

        flat_input_size = tf.compat.v1.cumprod(input_shape)[-1]
        flat_output_shape = tf.stack([output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]])

        pool_ = tf.reshape(pool, tf.stack([flat_input_size]))
        batch_range = tf.reshape(tf.range(tf.cast(output_shape[0], tf.int64), dtype=ind.dtype),
                                 shape=tf.stack([input_shape[0], 1, 1, 1]))
        b = tf.ones_like(ind) * batch_range
        b = tf.reshape(b, tf.stack([flat_input_size, 1]))
        ind_ = tf.reshape(ind, tf.stack([flat_input_size, 1]))
        ind_ = ind_ - b * tf.cast(flat_output_shape[1], tf.int64)
        ind_ = tf.concat([b, ind_], 1)

        ret = tf.scatter_nd(ind_, pool_, shape=tf.cast(flat_output_shape, tf.int64))
        ret = tf.reshape(ret, tf.stack(output_shape))

        set_input_shape = pool.get_shape()
        set_output_shape = [set_input_shape[0], set_input_shape[1] * ksize[1], set_input_shape[2] * ksize[2], set_input_shape[3]]
        ret.set_shape(set_output_shape)
    return ret

#keras.backend.set_learning_phase(1)

images =  Input(shape=(img_width, img_height, 3))
output_1a = Conv2D(64, (3, 3),  padding='same')(images)
output_1 = Activation('relu')(output_1a)
output_2a = Conv2D(64, (3, 3),  padding='same')(output_1)
output_2 = Activation('relu')(output_2a)
output_3, argmax1 = MaxPoolingWithArgmax2D(output_2,ksize=(1, 2,2,1),strides=[1,2,2,1],padding='SAME')
output_4a = Conv2D(128, (3, 3),  padding='same')(output_3)
output_4 = Activation('relu')(output_4a)
output_5a = Conv2D(128, (3, 3),  padding='same')(output_4)
output_5 = Activation('relu')(output_5a)
output_6, argmax2 = MaxPoolingWithArgmax2D(output_5,ksize=(1, 2,2,1),strides=[1,2,2,1],padding='SAME')
output_7a = Conv2D(256, (3, 3),  padding='same')(output_6)
output_7 = Activation('relu')(output_7a)
output_8a = Conv2D(256, (3, 3),  padding='same')(output_7)
output_8 = Activation('relu')(output_8a)
output_9a = Conv2D(256, (3, 3),  padding='same')(output_8)
output_9 = Activation('relu')(output_9a)
output_10, argmax3 = MaxPoolingWithArgmax2D(output_9,ksize=(1, 2,2,1),strides=[1,2,2,1],padding='SAME')
output_11a = Conv2D(512, (3, 3),  padding='same')(output_10)
output_11 = Activation('relu')(output_11a)
output_12a = Conv2D(512, (3, 3),  padding='same')(output_11)
output_12 = Activation('relu')(output_12a)
output_13a = Conv2D(512, (3, 3),  padding='same')(output_12)
output_13 = Activation('relu')(output_13a)
output_14, argmax4 = MaxPoolingWithArgmax2D(output_13,ksize=(1, 2,2,1),strides=[1,2,2,1],padding='SAME')
output_15a = Conv2D(512, (3, 3),  padding='same')(output_14)
output_15 = Activation('relu')(output_15a)
output_16a = Conv2D(512, (3, 3),  padding='same')(output_15)
output_16 = Activation('relu')(output_16a)
output_17a = Conv2D(512, (3, 3),  padding='same')(output_16)
output_17 = Activation('relu')(output_17a)
output_18, argmax5 = MaxPoolingWithArgmax2D(output_17,ksize=(1, 2,2,1),strides=[1,2,2,1],padding='SAME')

output_19 = MaxUnpooling2D(output_18, argmax5)
output_20a = Conv2D(512, (3, 3),  padding='same')(output_19)
output_20 = Activation('relu')(output_20a)
output_21a = Conv2D(512, (3, 3),  padding='same')(output_20)
output_21 = Activation('relu')(output_21a)
output_22a = Conv2D(512, (3, 3),  padding='same')(output_21)
output_22 = Activation('relu')(output_22a)
output_23 = MaxUnpooling2D(output_22, argmax4)
output_24a = Conv2D(512, (3, 3),  padding='same')(output_23)
output_24 = Activation('relu')(output_24a)
output_25a = Conv2D(512, (3, 3),  padding='same')(output_24)
output_25 = Activation('relu')(output_25a)
output_26a = Conv2D(256, (3, 3),  padding='same')(output_25)
output_26 = Activation('relu')(output_26a)
output_27 = MaxUnpooling2D(output_26, argmax3)
output_28a = Conv2D(256, (3, 3),  padding='same')(output_27)
output_28 = Activation('relu')(output_28a)
output_29a = Conv2D(256, (3, 3),  padding='same')(output_28)
output_29 = Activation('relu')(output_29a)
output_30a = Conv2D(128, (3, 3),  padding='same')(output_29)
output_30 = Activation('relu')(output_30a)
output_31 = MaxUnpooling2D(output_30, argmax2)
output_32a = Conv2D(128, (3, 3),  padding='same')(output_31)
output_32 = Activation('relu')(output_32a)
output_33a = Conv2D(64, (3, 3),  padding='same')(output_32)
output_33 = Activation('relu')(output_33a)
output_34 = MaxUnpooling2D(output_33, argmax1)
output_35a = Conv2D(64, (3, 3),  padding='same')(output_34)
output_35 = Activation('relu')(output_35a)
output_36a = Conv2D(66, (3, 3),  padding='same')(output_35)
predictions = Activation('softmax')(output_36a)


model = Model(inputs=images, outputs=predictions)

HRVSProp=keras.optimizers.RMSprop(learning_rate=0.0001, rho=0.9)

model.compile(optimizer=HRVSProp,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
#sparse categorical is ok for output where 1 class is true and the others are false

#sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.summary()

#model.save_weights('default.h5')
#model.load_weights('default.h5')
#model.load_weights('segnet g1.h5')

def ImportValidation():
    in_valid = []
    out_valid = []

    for i in sorted(os.listdir(dataset_dir + 'validation/images/')):
        # load the image
        in_valid.append(Image.open(dataset_dir + 'validation/images/' + i))
        print('checkpoint 4.5 img '+str(i))
        # resize image (nearest neighbors) and divide dataset into input and desired output images
        in_valid[-1] = np.array(in_valid[-1].resize((img_width, img_height)))
        in_valid[-1] = np.transpose(in_valid[-1], (1, 0, 2))

    print('checkpoint 5')

    for i in sorted(os.listdir(dataset_dir + 'validation/labels/')):
        # load the image
        out_valid.append(Image.open(dataset_dir + 'validation/labels/' + i))

        # resize image (nearest neighbors) and divide dataset into input and desired output images
        out_valid[-1] = np.array(out_valid[-1].resize((img_width, img_height), Image.NEAREST))
        out_valid[-1] = np.expand_dims(out_valid[-1], axis=-1)
        out_valid[-1] = np.transpose(out_valid[-1], (1, 0, 2))

    in_valid = np.array([x for x in in_valid])
    out_valid = np.array([x for x in out_valid])
    return in_valid, out_valid

#in_valid, out_valid = ImportValidation()

for epochs in range(n_epochs):
    #epochs=0
    se=0
    random.shuffle(trainingset)
    print('checkpoint 0')
    for r in range(runs):
        #r=0
        in_train=[]
        out_train=[]
        train_size = train_batch*(r+1)
        print('checkpoint 1, run '+str(r)+'/'+str(runs))

        for i in trainingset[train_size-train_batch:train_size]:
            # load the image
            in_train.append(Image.open(dataset_dir+'training/images_lin/'+i+'.jpg'))

            # resize image (nearest neighbors) and divide dataset into input and desired output images
            in_train[-1] = np.array(in_train[-1].resize((img_width,img_height)))
            in_train[-1] = np.transpose(in_train[-1], (1, 0, 2))

        print('checkpoint 2')

        for i in trainingset[train_size-train_batch:train_size]:
            # load the image
            out_train.append(Image.open(dataset_dir+'training/labels/'+i+'.png'))

            # resize image (nearest neighbors) and divide dataset into input and desired output images
            out_train[-1] = np.array(out_train[-1].resize((img_width,img_height), Image.NEAREST))
            out_train[-1] = np.expand_dims(out_train[-1], axis=-1)
            out_train[-1] = np.transpose(out_train[-1], (1, 0, 2))

        print('checkpoint 3')
        #in_train, in_test, out_train, out_test = sklearn.model_selection.train_test_split(in_image,out_image,test_size=0.2)

        in_train = np.array([x for x in in_train])
        out_train = np.array([x for x in out_train])

        modelzero = model.fit(in_train, out_train, epochs=1, batch_size=1, validation_data=[in_train,out_train])

        accuracy_hist += modelzero.history['accuracy']
        loss_hist += modelzero.history['loss']
        print('checkpoint 4')

    if in_valid==[]:
        in_valid, out_valid = ImportValidation()

    val_loss, val_accuracy = model.evaluate(in_train, out_train, batch_size=1, verbose=1)

    #valaccuracy_hist += modelzero.history['val_accuracy']
    val_accuracy_hist += [val_accuracy]
    val_loss_hist += [val_loss]

    print(val_loss, val_accuracy)

    ssee=se+epochs
    model.save_weights('segnet g'+str(ssee)+'.h5')
    if val_loss==min(val_loss_hist):
        model.save_weights('segnet.h5')

    with open('vah.pickle', 'wb') as f:
        pickle.dump(val_accuracy_hist, f, pickle.HIGHEST_PROTOCOL)

    with open('vlh.pickle', 'wb') as f:
        pickle.dump(val_loss_hist, f, pickle.HIGHEST_PROTOCOL)


#with open('vah.pickle', 'rb') as f:
    #val_accuracy_hist = pickle.load(f)

#with open('vlh.pickle', 'rb') as f:
    #val_loss_hist = pickle.load(f)

plt.plot(accuracy_hist, label='accuracy')
plt.plot(val_accuracy_hist, label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
#plt.ylim([0.0, 1])
plt.legend(loc='lower right')


in_test = []
out_test = []

for i in sorted(os.listdir(dataset_dir + 'training/images_lin/'))[16000:]:
    # load the image
    in_test.append(Image.open(dataset_dir + 'training/images_lin/' + i))

    # resize image (nearest neighbors) and divide dataset into input and desired output images
    in_test[-1] = np.array(in_test[-1].resize((img_width, img_height)))
    in_test[-1] = np.transpose(in_test[-1], (1, 0, 2))

for i in sorted(os.listdir(dataset_dir + 'training/labels/'))[16000:]:
    # load the image
    out_test.append(Image.open(dataset_dir + 'training/labels/' + i))

    # resize image (nearest neighbors) and divide dataset into input and desired output images
    out_test[-1] = np.array(out_test[-1].resize((img_width, img_height), Image.NEAREST))
    out_test[-1] = np.expand_dims(out_test[-1], axis=-1)
    out_test[-1] = np.transpose(out_test[-1], (1, 0, 2))

in_test = np.array([x for x in in_test])
out_test = np.array([x for x in out_test])

test_loss, test_acc = model.evaluate(in_test,  out_test, batch_size=32, verbose=1)
#up to 16 batch
print(test_loss, test_acc)

test_accuracy_hist += [test_acc]
test_loss_hist += [test_loss]

 model.load_weights('segnet g1.h5')

np.mean(accuracy_hist)
