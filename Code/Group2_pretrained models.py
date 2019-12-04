import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import math
import cv2
import matplotlib.pyplot as plt
import os
import seaborn as sns
from PIL import Image
from scipy import misc
from os import listdir
from os.path import isfile, join
import numpy as np
from scipy import misc
from matplotlib.pyplot import imread
from random import shuffle
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils.np_utils import to_categorical

os.chdir('UTKFace')
onlyfiles = os.listdir()
shuffle(onlyfiles)
age = [i.split('_')[0] for i in onlyfiles]

classes = []
for i in age:
    if i == "model.hdf5":
        continue
    i = int(i)
    if i <= 6:
        classes.append(0)
    if (i > 6) and (i < 13):
        classes.append(1)
    if (i >= 13) and (i < 20):
        classes.append(2)
    if (i >= 20) and (i < 40):
        classes.append(3)
    if (i >=40) and (i< 60):
        classes.append(4)
    if i >= 60:
        classes.append(5)

X_data =[]
for file in onlyfiles:
    if file == "model.hdf5":
        continue
    face = cv2.imread(file)
    face =cv2.resize(face, (32, 32))
    X_data.append(face)

X = np.array(X_data)
categorical_labels = to_categorical(classes, num_classes =6)

(x_train, y_train), (x_test, y_test) = (X[:15000], categorical_labels[:15000]), (X[15000:], categorical_labels[15000:])
(x_valid, y_valid) = (x_test[:7000], y_test[:7000])
(x_test, y_test) = (x_test[7000:], y_test[7000:])

import keras
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers import GlobalAveragePooling2D
import numpy as np

from keras.models import Model
vgg16_model = keras.applications.vgg16.VGG16(weights="imagenet", include_top=False)
x = vgg16_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(6, activation='softmax')(x)
model = Model(inputs=vgg16_model.input, outputs=predictions)
#model.add(Dense(units=5000,activation="relu"))
#model.add(Dense(units=5000,activation="relu"))
#model.add(Dense(units=6, activation="softmax"))
from keras.optimizers import Adam
opt = Adam(lr=0.0001)

from keras.applications.vgg16 import preprocess_input
x_train= preprocess_input(x_train)

model.compile(loss='categorical_crossentropy',
              optimizer= opt,
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, batch_size=2)

score = model.evaluate(x_test, y_test)
print('Accuracy:', score)
