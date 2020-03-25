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
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Dense, BatchNormalization, Activation, Reshape, Concatenate, SeparableConv2D, Add, AvgPool2D, Dropout
from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.models import Model
from keras import optimizers
#from layers import GroupNormalization, InstanceNormalization
#from layers import MaxPoolingWithArgmax2D, MaxUnpooling2D
from tensorflow.compat.v1.nn import separable_conv2d
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
weights_dir = 'weights/d121 a'
img_width = 576 #576
img_height = 384 #384
n_epochs = 15
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

def DenseBlock(input, name, block,bb):
    db0 = InstanceNormalization()(input)
    db1 = Activation('relu')(db0)
    db2 = Conv2D((128),(1,1),padding='same')(db1)
    db3 = Dropout(0.15)(db2)
    db4 = InstanceNormalization()(db3)
    db5 = Activation('relu')(db4)
    db6 = Conv2D(32,(3,3),padding='same')(db5)
    db7 = Dropout(0.15)(db6)
    db8 = Concatenate()([input,db7])
    return db8

def Transition(input, name, block):
    t0 = InstanceNormalization()(input)
    t1 = Activation('relu')(t0)
    t2 = Conv2D((128*(2**block)),(1,1),padding='same')(t1)
    t3 = Dropout(0.15)(t2)
    t4 = AvgPool2D((2,2),strides=2)(t3)
    return t4

l000 = Input(shape=(img_width, img_height, 3))

l011 = Conv2D(64,(7,7),strides=2,padding='same')(l000)
l012 = MaxPooling2D((3,3),strides=2,padding='same')(l011)

l021 = DenseBlock(l011,'db1',0,0)
l022 = DenseBlock(l021,'db2',1,0)
l023 = DenseBlock(l022,'db3',2,0)
l024 = DenseBlock(l023,'db4',3,0)
l025 = DenseBlock(l024,'db5',4,0)
l026 = DenseBlock(l025,'db6',5,0)

l031 = Transition(l026,'t1',0)

l041 = DenseBlock(l031,'db7',0,1)
l042 = DenseBlock(l041,'db8',1,1)
l043 = DenseBlock(l042,'db9',2,1)
l044 = DenseBlock(l043,'db10',3,1)
l045 = DenseBlock(l044,'db11',4,1)
l046 = DenseBlock(l045,'db12',5,1)
l047 = DenseBlock(l046,'db13',6,1)
l048 = DenseBlock(l047,'db14',7,1)
l049 = DenseBlock(l048,'db15',8,1)
l050 = DenseBlock(l049,'db16',9,1)
l051 = DenseBlock(l050,'db17',10,1)
l052 = DenseBlock(l051,'db18',11,1)

l061 = Transition(l052,'t2',1)

l071 = DenseBlock(l061,'db19',0,2)
l072 = DenseBlock(l071,'db20',1,2)
l073 = DenseBlock(l072,'db21',2,2)
l074 = DenseBlock(l073,'db22',3,2)
l075 = DenseBlock(l074,'db23',4,2)
l076 = DenseBlock(l075,'db24',5,2)
l077 = DenseBlock(l076,'db25',6,2)
l078 = DenseBlock(l077,'db26',7,2)
l079 = DenseBlock(l078,'db27',8,2)
l080 = DenseBlock(l079,'db28',9,2)
l081 = DenseBlock(l080,'db29',10,2)
l082 = DenseBlock(l081,'db30',11,2)
l091 = DenseBlock(l082,'db31',12,2)
l092 = DenseBlock(l091,'db32',13,2)
l093 = DenseBlock(l092,'db33',14,2)
l094 = DenseBlock(l093,'db34',15,2)
l095 = DenseBlock(l094,'db35',16,2)
l096 = DenseBlock(l095,'db36',17,2)
l097 = DenseBlock(l096,'db37',18,2)
l098 = DenseBlock(l097,'db38',19,2)
l099 = DenseBlock(l098,'db39',20,2)
l100 = DenseBlock(l099,'db40',21,2)
l101 = DenseBlock(l100,'db41',22,2)
l102 = DenseBlock(l101,'db42',23,2)

l111 = Transition(l102,'t3',2)

l121 = DenseBlock(l111,'db43',0,3)
l122 = DenseBlock(l121,'db44',1,3)
l123 = DenseBlock(l122,'db45',2,3)
l124 = DenseBlock(l123,'db46',3,3)
l125 = DenseBlock(l124,'db47',4,3)
l126 = DenseBlock(l125,'db48',5,3)
l127 = DenseBlock(l126,'db49',6,3)
l128 = DenseBlock(l127,'db50',7,3)
l129 = DenseBlock(l128,'db51',8,3)
l130 = DenseBlock(l129,'db52',9,3)
l131 = DenseBlock(l130,'db53',10,3)
l132 = DenseBlock(l131,'db54',11,3)
l141 = DenseBlock(l132,'db55',12,3)
l142 = DenseBlock(l141,'db56',13,3)
l143 = DenseBlock(l142,'db57',14,3)
l144 = DenseBlock(l143,'db58',15,3)

d011 = Conv2D(512,(1,1),padding='same')(l102)
d012 = InstanceNormalization()(d011)
d013 = Activation('relu')(d012)
d013d = Dropout(0.15)(d013)
d014 = Conv2DTranspose(512,kernel_size=(2,2), strides=(2,2), padding='same')(l144)
d015 = InstanceNormalization()(d014)
d016 = Activation('relu')(d015)
d016d = Dropout(0.15)(d016)
d010 = Add()([d016d,d013d])

d021 = Conv2D(256,(1,1),padding='same')(l052)
d022 = InstanceNormalization()(d021)
d023 = Activation('relu')(d022)
d023d = Dropout(0.15)(d023)
d024 = Conv2DTranspose(256,kernel_size=(2,2), strides=(2,2), padding='same')(d010)
d025 = InstanceNormalization()(d024)
d026 = Activation('relu')(d025)
d026d = Dropout(0.15)(d026)
d020 = Add()([d026d,d023d])

d031 = Conv2D(128,(1,1),padding='same')(l026)
d032 = InstanceNormalization()(d031)
d033 = Activation('relu')(d032)
d033d = Dropout(0.15)(d033)
d034 = Conv2DTranspose(128,kernel_size=(2,2), strides=(2,2), padding='same')(d020)
d035 = InstanceNormalization()(d034)
d036 = Activation('relu')(d035)
d036d = Dropout(0.15)(d036)
d030 = Add()([d036d,d033d])

d041 = Conv2D(64,(1,1),padding='same')(l000)
d042 = InstanceNormalization()(d041)
d043 = Activation('relu')(d042)
d043d = Dropout(0.15)(d043)
d044 = Conv2DTranspose(64,kernel_size=(2,2), strides=(2,2), padding='same')(d030)
d045 = InstanceNormalization()(d044)
d046 = Activation('relu')(d045)
d046d = Dropout(0.15)(d046)
d040 = Add()([d046d,d043d])

#l999 = Activation('softmax')(l251)

d997 = Dense(39)(d040)
d998 = InstanceNormalization()(d997)

#l151 = AvgPool2D((7,7),strides=7,padding='same')(l144)
#l152 = Dense(39)(l151)

d999 = Activation('softmax')(d998)

model = Model(inputs=l000, outputs=d999)

HRVSProp=keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=HRVSProp,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
#sparse categorical is ok for output where 1 class is true and the others are false

#sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.summary()

#model.save_weights('default.h5')
#model.load_weights('default.h5')
model.load_weights('weights/d121 a21.tf')

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

with open('snf_acc.pickle', 'rb') as f:
    accuracy_hist = pickle.load(f)

with open('snf_loss.pickle', 'rb') as f:
    loss_hist = pickle.load(f)

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

        modelzero = model.fit(in_train, out_train, epochs=1, batch_size=1)

        #val_accuracy_hist += modelzero.history['val_accuracy']
        #val_loss_hist += modelzero.history['val_loss']

        accuracy_hist += modelzero.history['accuracy']
        loss_hist += modelzero.history['loss']
        print('checkpoint 4')

    if in_valid==[]:
        in_valid, out_valid = ImportValidation()

    val_loss, val_accuracy = model.evaluate(in_valid, out_valid, batch_size=1, verbose=1)

    #val_accuracy_hist += modelzero.history['val_accuracy']
    #val_loss_hist += modelzero.history['val_loss']
    val_accuracy_hist += [val_accuracy]
    val_loss_hist += [val_loss]

    #print(val_loss, val_accuracy)

    ssee=se+epochs
    #model.save_weights('segnet g'+str(ssee)+'.h5')
    model.save_weights(weights_dir+str(ssee)+'.tf')

    #if val_loss==min(val_loss_hist):
    #    model.save_weights('segnet.h5')

    with open('s121a_val_acc.pickle', 'wb') as f:
        pickle.dump(val_accuracy_hist, f, pickle.HIGHEST_PROTOCOL)

    with open('s121a_val_loss.pickle', 'wb') as f:
        pickle.dump(val_loss_hist, f, pickle.HIGHEST_PROTOCOL)

    with open('s121a_acc.pickle', 'wb') as f:
        pickle.dump(accuracy_hist, f, pickle.HIGHEST_PROTOCOL)

    with open('s121a_loss.pickle', 'wb') as f:
        pickle.dump(loss_hist, f, pickle.HIGHEST_PROTOCOL)


plt.plot(val_loss_hist, label='val_loss')
plt.plot(val_accuracy_hist, label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
#plt.ylim([0.0, 1])
plt.legend(loc='lower left')
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

model.save_weights(weights_dir +'init.tf')
model.load_weights(weights_dir +'init.tf')

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