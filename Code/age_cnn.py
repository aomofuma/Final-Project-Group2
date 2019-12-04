import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
import cv2
import matplotlib.pyplot as plt
import os
import seaborn as sns
#import umap
from PIL import Image
from scipy import misc
from os import listdir
from os.path import isfile, join
import numpy as np
from scipy import misc
from random import shuffle
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils.np_utils import to_categorical
import PIL
from keras.optimizers import Adam


os.chdir('UTKFace')
im = Image.open('1_0_0_20161219140623097.jpg.chip.jpg').resize((128,128))

onlyfiles = os.listdir()
#onlyfiles1 = os.listdir()
#onlyfiles = onlyfiles1[0:7000]
shuffle(onlyfiles)
age = [i.split('_')[0] for i in onlyfiles]

classes = []
for i in age:
    if i == 'model.hdf5':
        continue

    if i == "plot.jpg":
        continue

    i = int(i)
    if i <= 6:
        classes.append(0)
    if (i>6) and (i<13):
        classes.append(1)
    if (i>=13) and (i<20):
        classes.append(2)
    if (i>=20) and (i<40):
        classes.append(3)
    if (i>=40) and (i<60):
        classes.append(4)
    if i>=60:
        classes.append(5)

X_data =[]
for file in onlyfiles:
    if file == "model.hdf5":
        continue
    face = cv2.imread(file)
    face = cv2.resize(face, (32, 32) )
    X_data.append(face)

X = np.squeeze(X_data)
# X.shape

# normalize data
X = X.astype('float32')
X /= 255

categorical_labels = to_categorical(classes, num_classes=6)
(x_train, y_train), (x_test, y_test) = (X[:15000],categorical_labels[:15000]), (X[15000:], categorical_labels[15000:])
(x_valid , y_valid) = (x_test[:5780], y_test[:5780])
(x_test, y_test) = (x_test[5770:], y_test[5770:])

model = tf.keras.Sequential()

# Must define the input shape in the first layer of the neural network
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(32,32,3)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(6, activation='softmax'))

# Take a look at the model summary
model.summary()

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy',
             optimizer=optimizer,
             metrics=['accuracy'])

model.fit(x_train,
         y_train,
         batch_size=10,
         epochs=25,
         validation_data=(x_valid, y_valid))

# Evaluate the model on test set
score = model.evaluate(x_test, y_test, verbose=0)

# Print test accuracy
print('\n', 'Test accuracy:', score[1])

labels =["Infants",  # index 0
        "Children",      # index 1
        "Teenagers",     # index 2
        "Young Adults",        # index 3
        "Adults",         # index 4
        "Senior Citizens" #index 5
        ]

y_hat = model.predict(x_test)

# Plot a random sample of 10 test images, their predicted labels and ground truth
figure = plt.figure(figsize=(20, 8))
for i, index in enumerate(np.random.choice(x_test.shape[0], size=15, replace=False)):
    ax = figure.add_subplot(3, 5, i + 1, xticks=[], yticks=[])
    # Display each image
    ax.imshow(np.squeeze(x_test[index]))
    predict_index = np.argmax(y_hat[index])
    true_index = np.argmax(y_test[index])
    # Set the title for each image
    ax.set_title("{} ({})".format(labels[predict_index],
                                  labels[true_index]),
                                  color=("green" if predict_index == true_index else "red"))
plt.plot()
plt.savefig("plot_age.jpg")
plt.show()