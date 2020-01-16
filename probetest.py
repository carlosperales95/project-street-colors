import numpy as np
import pandas as pd
import requests
import json

from PIL import Image
# load the image
image = Image.open('./small-dset/aSqVUgt36gddhmJdI1lXNA.jpg')
# summarize some details about the image
print(image.format)
print(image.mode)
print(image.size)
# show the image
#image.show()
