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


l000 = Input(shape=(img_width, img_height, 3))
l001 = Conv2D(32,(3,3),strides=1, padding='same')(l000)
l002 = InstanceNormalization()(l001)
l003 = Activation('relu')(l002)

#block1
l011 = Conv2D(32,(3,3),strides=2, padding='same')(l000)
l012 = InstanceNormalization()(l011)
l010 = Activation('relu')(l012)

l021 = Conv2D(64,(3,3),padding='same')(l010)
l022 = InstanceNormalization()(l021)
l020 = Activation('relu')(l022)

l031 = Conv2D(128,(1,1), strides=2,name='resid1')(l020)
l030 = InstanceNormalization()(l031)

#block2
l041 = SeparableConv2D(128,(3,3),padding='same',name='block2')(l020)
l042 = InstanceNormalization()(l041)
l040 = Activation('relu')(l042)

l051 = SeparableConv2D(128,(3,3),padding='same')(l040)
l052 = InstanceNormalization()(l051)
l053 = MaxPooling2D((3,3),strides=2,padding='same')(l052)
l050 = Add()([l030,l053])

l061 = Conv2D(256,(1,1), strides=2,name='resid2')(l050)
l060 = InstanceNormalization()(l061)

#block3
l071 = Activation('relu',name='block3')(l050)
l072 = SeparableConv2D(256,(3,3),padding='same')(l071)
l073 = InstanceNormalization()(l072)
l070 = Activation('relu')(l073)

l081 = SeparableConv2D(256,(3,3),padding='same')(l070)
l082 = InstanceNormalization()(l081)
l083 = MaxPooling2D((3,3),strides=2,padding='same')(l082)
l080 = Add()([l060,l083])

l091 = Conv2D(728,(1,1), strides=2,name='resid3')(l080)
l090 = InstanceNormalization()(l091)

#block4
l101 = Activation('relu',name='block4')(l080)
l102 = SeparableConv2D(728,(3,3),padding='same')(l101)
l103 = InstanceNormalization()(l102)
l100 = Activation('relu')(l103)

l111 = SeparableConv2D(728,(3,3),padding='same')(l100)
l112 = InstanceNormalization()(l111)
l113 = MaxPooling2D((3,3),strides=2,padding='same')(l112)
l110 = Add()([l090,l113])

#middleflow
l121 = Activation('relu')(l110)
l122 = SeparableConv2D(728,(3,3),padding='same')(l121)
l123 = InstanceNormalization()(l122)
l124 = Activation('relu')(l123)
l124d = Dropout(0.15)(l124)
l125 = SeparableConv2D(728,(3,3),padding='same')(l124d)
l126 = InstanceNormalization()(l125)
l127 = Activation('relu')(l126)
l127d = Dropout(0.15)(l127)
l128 = SeparableConv2D(728,(3,3),padding='same')(l127d)
l129 = InstanceNormalization()(l128)
l120 = Activation('relu')(l129)
l120d = Dropout(0.15)(l120)
l12 = Add()([l110,l120d])

l131 = Activation('relu')(l12)
l132 = SeparableConv2D(728,(3,3),padding='same')(l131)
l133 = InstanceNormalization()(l132)
l134 = Activation('relu')(l133)
l134d = Dropout(0.15)(l134)
l135 = SeparableConv2D(728,(3,3),padding='same')(l134d)
l136 = InstanceNormalization()(l135)
l137 = Activation('relu')(l136)
l137d = Dropout(0.15)(l137)
l138 = SeparableConv2D(728,(3,3),padding='same')(l137d)
l139 = InstanceNormalization()(l138)
l130 = Activation('relu')(l139)
l130d = Dropout(0.15)(l130)
l13 = Add()([l12,l130d])

l141 = Activation('relu')(l13)
l142 = SeparableConv2D(728,(3,3),padding='same')(l141)
l143 = InstanceNormalization()(l142)
l144 = Activation('relu')(l143)
l144d = Dropout(0.15)(l144)
l145 = SeparableConv2D(728,(3,3),padding='same')(l144d)
l146 = InstanceNormalization()(l145)
l147 = Activation('relu')(l146)
l147d = Dropout(0.15)(l147)
l148 = SeparableConv2D(728,(3,3),padding='same')(l147)
l149 = InstanceNormalization()(l148)
l140 = Activation('relu')(l149)
l140d = Dropout(0.15)(l140)
l14 = Add()([l13,l140d])

l151 = Activation('relu')(l14)
l152 = SeparableConv2D(728,(3,3),padding='same')(l151)
l153 = InstanceNormalization()(l152)
l154 = Activation('relu')(l153)
l154d = Dropout(0.15)(l154)
l155 = SeparableConv2D(728,(3,3),padding='same')(l154d)
l156 = InstanceNormalization()(l155)
l157 = Activation('relu')(l156)
l157d = Dropout(0.15)(l157)
l158 = SeparableConv2D(728,(3,3),padding='same')(l157d)
l159 = InstanceNormalization()(l158)
l150 = Activation('relu')(l159)
l150d = Dropout(0.15)(l150)
l15 = Add()([l14,l150d])

l161 = Activation('relu')(l15)
l162 = SeparableConv2D(728,(3,3),padding='same')(l161)
l163 = InstanceNormalization()(l162)
l164 = Activation('relu')(l163)
l164d = Dropout(0.15)(l164)
l165 = SeparableConv2D(728,(3,3),padding='same')(l164d)
l166 = InstanceNormalization()(l165)
l167 = Activation('relu')(l166)
l167d = Dropout(0.15)(l167)
l168 = SeparableConv2D(728,(3,3),padding='same')(l167d)
l169 = InstanceNormalization()(l168)
l160 = Activation('relu')(l169)
l160d = Dropout(0.15)(l160)
l16 = Add()([l15,l160d])

l171 = Activation('relu')(l16)
l172 = SeparableConv2D(728,(3,3),padding='same')(l171)
l173 = InstanceNormalization()(l172)
l174 = Activation('relu')(l173)
l174d = Dropout(0.15)(l174)
l175 = SeparableConv2D(728,(3,3),padding='same')(l174d)
l176 = InstanceNormalization()(l175)
l177 = Activation('relu')(l176)
l177d = Dropout(0.15)(l177)
l178 = SeparableConv2D(728,(3,3),padding='same')(l177d)
l179 = InstanceNormalization()(l178)
l170 = Activation('relu')(l179)
l170d = Dropout(0.15)(l170)
l17 = Add()([l16,l170d])

l181 = Activation('relu')(l17)
l182 = SeparableConv2D(728,(3,3),padding='same')(l181)
l183 = InstanceNormalization()(l182)
l184 = Activation('relu')(l183)
l185 = SeparableConv2D(728,(3,3),padding='same')(l184)
l186 = InstanceNormalization()(l185)
l187 = Activation('relu')(l186)
l188 = SeparableConv2D(728,(3,3),padding='same')(l187)
l189 = InstanceNormalization()(l188)
l180 = Activation('relu')(l189)
l18 = Add()([l17,l180])

l191 = Activation('relu')(l18)
l192 = SeparableConv2D(728,(3,3),padding='same')(l191)
l193 = InstanceNormalization()(l192)
l194 = Activation('relu')(l193)
l195 = SeparableConv2D(728,(3,3),padding='same')(l194)
l196 = InstanceNormalization()(l195)
l197 = Activation('relu')(l196)
l198 = SeparableConv2D(728,(3,3),padding='same')(l197)
l199 = InstanceNormalization()(l198)
l190 = Activation('relu')(l199)
l19 = Add()([l18,l190])

#exitflow
#l201 = Conv2D(1024,(1,1), strides=2)(l19)
l201 = Conv2D(1024,(1,1),padding='same')(l17) # tolti 2 layer
l200 = InstanceNormalization()(l201)

l211 = Activation('relu')(l17) # tolti due layer
l212 = SeparableConv2D(728,(3,3),padding='same')(l211)
l213 = InstanceNormalization()(l212)
l210 = Activation('relu')(l213)
l210d = Dropout(0.15)(l210)

l221 = SeparableConv2D(1024,(3,3),padding='same')(l210d)
l222 = InstanceNormalization()(l221)
#l223 = MaxPooling2D((3,3),strides=2,padding='same')(l222)
l220 = Add()([l200,l222])

l231 = SeparableConv2D(1536,(3,3),padding='same')(l220)
l232 = InstanceNormalization()(l231)
l230 = Activation('relu')(l232)
l230d = Dropout(0.15)(l230)


l241 = SeparableConv2D(2048,(3,3),padding='same')(l230)
l242 = InstanceNormalization()(l241)
l240 = Activation('relu')(l242)

#l251 = AvgPool2D((10,10))(l240)
#l252 = Dense(39)(l251)
#l250 = InstanceNormalization()(l252)

#deconvolution
#d001 = Conv2D(1024,(1,1),padding='same')(l222)
#d002 = InstanceNormalization()(d001)
#d003 = Activation('relu')(d002)
#d004 = Conv2DTranspose(1024,kernel_size=(2,2), strides=(2,2), padding='same')(l240)
#d005 = InstanceNormalization()(d004)
#d006 = Activation('relu')(d005)
#d000 = Add()([d006,d003])

d011 = Conv2D(728,(1,1),padding='same')(l112)
d012 = InstanceNormalization()(d011)
d013 = Activation('relu')(d012)
d013d = Dropout(0.15)(d013)
d014 = Conv2DTranspose(728,kernel_size=(2,2), strides=(2,2), padding='same')(l230d) #tolto un layer
d015 = InstanceNormalization()(d014)
d016 = Activation('relu')(d015)
d010 = Concatenate()([d016,d013d])

d021 = Conv2D(256,(1,1),padding='same')(l082)
d022 = InstanceNormalization()(d021)
d023 = Activation('relu')(d022)
d023d = Dropout(0.15)(d023)
d024 = Conv2DTranspose(256,kernel_size=(2,2), strides=(2,2), padding='same')(d010)
d025 = InstanceNormalization()(d024)
d026 = Activation('relu')(d025)
d020 = Concatenate()([d026,d023d])

d031 = Conv2D(128,(1,1),padding='same')(l052)
d032 = InstanceNormalization()(d031)
d033 = Activation('relu')(d032)
d033d = Dropout(0.15)(d033)
d034 = Conv2DTranspose(128,kernel_size=(2,2), strides=(2,2), padding='same')(d020)
d035 = InstanceNormalization()(d034)
d036 = Activation('relu')(d035)
d030 = Concatenate()([d036,d033d])

d041 = Conv2D(64,(1,1),padding='same')(l003)
d042 = InstanceNormalization()(d041)
d043 = Activation('relu')(d042)
d043d = Dropout(0.15)(d043)
d044 = Conv2DTranspose(64,kernel_size=(2,2), strides=(2,2), padding='same')(d030)
d045 = InstanceNormalization()(d044)
d046 = Activation('relu')(d045)
d040 = Concatenate()([d046,d043d])

#l999 = Activation('softmax')(l251)
d994 = Conv2D(39,(1,1),padding='same')(d040)
d995 = InstanceNormalization()(d994)
d996 = Activation('relu')(d995)
d997 = Conv2D(39,(3,3),padding='same')(d996)
d998 = InstanceNormalization()(d997)
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
model.load_weights('weights/segnet xd12.tf')

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

with open('xcc_val_acc.pickle', 'rb') as f:
    val_accuracy_hist = pickle.load(f)

with open('xcc_val_loss.pickle', 'rb') as f:
    val_loss_hist = pickle.load(f)

with open('xcd_acc.pickle', 'rb') as f:
    accuracy_hist = pickle.load(f)

with open('xcd_loss.pickle', 'rb') as f:
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

########################### Test visualization

with open('config.json') as config_file:
    config = json.load(config_file)
# in this example we are only interested in the labels
labels = config['labels']

palette=[]
for label_id, label in enumerate(labels):
    palette.append([[label_id]+label["color"]])

palette=np.array(palette)
palette[:,:,0]=ClassCompact(palette[:,:,:])[:,:,0]
palette=palette[palette[:,0,0].argsort()][::-1]

for j in range(65):
    if palette[(65-j),0,0]==palette[(65-j-1),0,0]:
        palette=np.delete(palette,(65-j),0)

labelrgb=matplotlib.colors.ListedColormap((palette[:,0,1:4]/255)[::-1])


weirdel=model.predict(in_test[3:4],verbose=1)

slot=0

plt.imshow(out_test[slot,:,:,0],cmap=labelrgb)
plt.imshow(in_test[slot,:,:,0])

plt.imshow(np.argmax(weirdel, axis=-1)[slot,:,:], labelrgb)


############################### Accuracy heatmap

cor_valid=np.zeros([39,39],dtype=int)
for i in range(2000):
    pred_valid=model.predict(in_test[i:i+1],verbose=1)
    pmax_valid=np.argmax(pred_valid, axis=-1)[0,:,:]
    for pxw in range(img_width):
        for pxh in range(img_height):
            cor_valid[out_test[i,pxw,pxh,0],pmax_valid[pxw,pxh]]+=1

np.savetxt("dn121_test.csv", cor_valid, delimiter=";", fmt="%u")
#pd.DataFrame(cor_valid).to_csv("xcd_test.csv",)

############################# Prediction

in_pred = []
in_pred.append(Image.open("/home/leonardo/Scaricati/bargellino.jpeg"))
in_pred[-1] = np.array(in_pred[-1].resize((img_width, img_height)))
in_pred[-1] = np.transpose(in_pred[-1], (1, 0, 2))

in_pred = np.array([x for x in in_pred])

zitadel=model.predict(in_pred[0:1],verbose=1)

plt.imshow(np.argmax(zitadel, axis=-1)[slot,:,:], labelrgb)
plt.imshow(in_pred[slot,:,:,0])