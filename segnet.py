import numpy as np
import pandas as pd
import requests
import json
import os
import tensorflow as tf
from tensorflow import keras
import sklearn.model_selection
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Dense, BatchNormalization, Activation, Reshape, Concatenate
from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.models import Model
from keras import optimizers
#from layers import GroupNormalization, InstanceNormalization
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
weights_dir = 'weights/segnet c'
img_width = 576 #576
img_height = 384 #384
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
output_16a = Conv2D(512, (3, 3),  padding='same')(output_15)
output_16b = InstanceNormalization()(output_16a)
output_16 = Activation('relu')(output_16b)
output_17a = Conv2D(512, (3, 3),  padding='same')(output_16)
output_17b = InstanceNormalization()(output_17a)
output_17 = Activation('relu')(output_17b)
output_18 = MaxPooling2D((2, 2))(output_17)

output_19a = Conv2DTranspose(512,kernel_size=(2,2), strides=(2,2), padding='same')(output_18)
output_19b = InstanceNormalization()(output_19a)
output_19 = Activation('relu')(output_19b)
#output_20a = Conv2D(512, (3, 3),  padding='same')(output_19)
#output_20b = InstanceNormalization()(output_20a)
#output_20 = Activation('relu')(output_20b)
output_20 = Concatenate()([output_19,output_17])
output_21a = Conv2D(512, (3, 3),  padding='same')(output_20)
output_21b = InstanceNormalization()(output_21a)
output_21 = Activation('relu')(output_21b)
output_22a = Conv2D(512, (3, 3),  padding='same')(output_21)
output_22b = InstanceNormalization()(output_22a)
output_22 = Activation('relu')(output_22b)
output_23a = Conv2DTranspose(512,kernel_size=(2,2), strides=(2,2), padding='same')(output_22)
output_23b = InstanceNormalization()(output_23a)
output_23 = Activation('relu')(output_23b)
#output_24a = Conv2D(512, (3, 3),  padding='same')(output_23)
#output_24b = InstanceNormalization()(output_24a)
#output_24 = Activation('relu')(output_24b)
output_24 = Concatenate()([output_23,output_13])
output_25a = Conv2D(512, (3, 3),  padding='same')(output_24)
output_25b = InstanceNormalization()(output_25a)
output_25 = Activation('relu')(output_25b)
output_26a = Conv2D(512, (3, 3),  padding='same')(output_25)
output_26b = InstanceNormalization()(output_26a)
output_26 = Activation('relu')(output_26b)
output_27a = Conv2DTranspose(256,kernel_size=(2,2), strides=(2,2), padding='same')(output_26)
output_27b = InstanceNormalization()(output_27a)
output_27 = Activation('relu')(output_27b)
#output_28a = Conv2D(256, (3, 3),  padding='same')(output_27)
#output_28b = InstanceNormalization()(output_28a)
#output_28 = Activation('relu')(output_28b)
output_28 = Concatenate()([output_27,output_9])
output_29a = Conv2D(256, (3, 3),  padding='same')(output_28)
output_29b = InstanceNormalization()(output_29a)
output_29 = Activation('relu')(output_29b)
output_30a = Conv2D(256, (3, 3),  padding='same')(output_29)
output_30b = InstanceNormalization()(output_30a)
output_30 = Activation('relu')(output_30b)
output_31a = Conv2DTranspose(128,kernel_size=(2,2), strides=(2,2), padding='same')(output_30)
output_31b = InstanceNormalization()(output_31a)
output_31 = Activation('relu')(output_31b)
#output_32a = Conv2D(128, (3, 3),  padding='same')(output_31)
#output_32b = InstanceNormalization()(output_32a)
#output_32 = Activation('relu')(output_32b)
output_32 = Concatenate()([output_31,output_5])
output_33a = Conv2D(128, (3, 3),  padding='same')(output_32)
output_33b = InstanceNormalization()(output_33a)
output_33 = Activation('relu')(output_33b)
output_34a = Conv2DTranspose(64,kernel_size=(2,2), strides=(2,2), padding='same')(output_33)
output_34b = InstanceNormalization()(output_34a)
output_34 = Activation('relu')(output_34b)
#output_35a = Conv2D(64, (3, 3),  padding='same')(output_34)
#output_35b = InstanceNormalization()(output_35a)
#output_35 = Activation('relu')(output_35b)
output_35 = Concatenate()([output_34,output_2])
output_36a = Conv2D(39, (1, 1),  padding='same')(output_35)
output_36b = InstanceNormalization()(output_36a)
predictions = Activation('softmax')(output_36a)

model = Model(inputs=images, outputs=predictions)

HRVSProp=keras.optimizers.Adam(learning_rate=0.0001)

model.compile(optimizer=HRVSProp,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
#sparse categorical is ok for output where 1 class is true and the others are false

#sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.summary()

#model.save_weights('default.h5')
#model.load_weights('default.h5')
model.load_weights('weights/segnet c14.tf')

def LabelCompact(image_label,class_from,class_to):
    class_matches = (image_label[ :, :, 0] == class_from)
    image_label[class_matches, 0] = class_to
    return image_label

def ClassCompact(image_label):
    couples = [(0,30),(1,19),(8,7),(9,2),(11,7),(12,7),(21,20),(22,20),(25,30),(26,30),(31,30),(33,32),(34,32),(35,32),(37,32),(38,32),(39,32),(40,32)
        ,(41,36),(49,32),(51,32),(52,20),(56,54),(57,20),(60,54),(61,54),(62,54)
        ,(65,0),(64,1),(63,8),(59,9),(58,11),(55,12),(54,21),(53,22),(50,25),(48,26),(47,31),(46,33),(45,34),(44,35),(43,37),(42,38)]
    couples[1][1]
    for j in couples:
        image_label = LabelCompact(image_label,j[0],j[1])
    return image_label

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
        out_valid[-1] = ClassCompact(out_valid[-1])


    in_valid = np.array([x for x in in_valid])
    out_valid = np.array([x for x in out_valid])
    return in_valid, out_valid

with open('snf_acc.pickle', 'rb') as f:
    val_accuracy_hist = pickle.load(f)

with open('snf_loss.pickle', 'rb') as f:
    val_loss_hist = pickle.load(f)

in_valid, out_valid = ImportValidation()

for epochs in range(n_epochs):
    #epochs=0
    se=15
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
            out_train[-1] = ClassCompact(out_train[-1])

        print('checkpoint 3')
        #in_train, in_test, out_train, out_test = sklearn.model_selection.train_test_split(in_image,out_image,test_size=0.2)

        in_train = np.array([x for x in in_train])
        out_train = np.array([x for x in out_train])

        modelzero = model.fit(in_train, out_train, epochs=1, batch_size=1, validation_data=[in_valid,out_valid])

        val_accuracy_hist += modelzero.history['val_accuracy']
        val_loss_hist += modelzero.history['val_loss']

        accuracy_hist += modelzero.history['accuracy']
        loss_hist += modelzero.history['loss']
        print('checkpoint 4')

    if in_valid==[]:
        in_valid, out_valid = ImportValidation()

    #val_loss, val_accuracy = model.evaluate(in_valid, out_valid, batch_size=1, verbose=1)

    #val_accuracy_hist += modelzero.history['val_accuracy']
    #val_loss_hist += modelzero.history['val_loss']
    #val_accuracy_hist += [val_accuracy]
    #val_loss_hist += [val_loss]

    #print(val_loss, val_accuracy)

    ssee=se+epochs
    #model.save_weights('segnet g'+str(ssee)+'.h5')
    model.save_weights(weights_dir+str(ssee)+'.tf')

    #if val_loss==min(val_loss_hist):
    #    model.save_weights('segnet.h5')

    with open('snf_acc.pickle', 'wb') as f:
        pickle.dump(val_accuracy_hist, f, pickle.HIGHEST_PROTOCOL)

    with open('snf_loss.pickle', 'wb') as f:
        pickle.dump(val_loss_hist, f, pickle.HIGHEST_PROTOCOL)


plt.plot(accuracy_hist, label='accuracy')
plt.plot(val_accuracy_hist, label = 'val_accuracy')
plt.plot(val_loss_hist, label= 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
#plt.ylim([0.0, 1])
plt.legend(loc='upper right')
plt.show()

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
    out_test[-1] = ClassCompact(out_test[-1])

in_test = np.array([x for x in in_test])
out_test = np.array([x for x in out_test])

test_loss, test_acc = model.evaluate(in_test,  out_test, batch_size=1, verbose=1)
#up to 16 batch
print(test_loss, test_acc)

test_accuracy_hist += [test_acc]
test_loss_hist += [test_loss]

model.load_weights('segnet g4.tf')

np.mean(accuracy_hist)

weirdel=model.predict(in_valid[6:7],verbose=1)

weirdel[0,0,0]
plt.imshow(weirdel[0,:,:,10])
plt.imshow(out_valid[1,:,:0])
plt.imshow(in_valid[1,:,:,0])

plt.imshow(np.argmax(weirdel, axis=-1)[0,:,:])

cor_valid=np.zeros([66,66],dtype=int)

for i in range(2000):
    pred_valid=model.predict(in_valid[i:i+1],verbose=1)
    pmax_valid=np.argmax(pred_valid, axis=-1)[0,:,:]
    for pxw in range(img_width):
        for pxh in range(img_height):
            cor_valid[out_valid[i,pxw,pxh,0],pmax_valid[pxw,pxh]]+=1

np.savetxt("cor_valid.csv", cor_valid, delimiter=";", fmt="%u")
pd.DataFrame(cor_valid).to_csv("cor_valid.csv",)