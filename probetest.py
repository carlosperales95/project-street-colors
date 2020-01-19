import numpy as np
import pandas as pd
import requests
import json
import os

from PIL import Image

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
        in_image.append(image[-1].resize((img_width,img_height),Image.NEAREST))
    else:
        out_image.append(image[-1].resize((img_width,img_height), Image.NEAREST))



