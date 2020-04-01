# -*- coding: utf-8 -*-
"""
Created on Sat May 25 19:57:07 2019

@author: Aspire V5
"""

# Convolutional Neural Network

import numpy as np
import pandas as pd
import re

import pandas as pd
import os
import shutil

# Importing the dataset
train_dataset = pd.read_csv('train.csv')
train_X = train_dataset.iloc[:, :-1].values
dfX = pd.DataFrame(train_X);
train_Y = train_dataset.iloc[:, 1].values
dfY = pd.DataFrame(train_Y);

#test_dataset = pd.read_csv('test_ApKoW4T.csv')
#test_X = test_dataset.values
#dfY = pd.DataFrame(test_X)

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 5, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from sklearn.cross_validation import train_test_split

X_train, Y_train, X_val, Y_val = train_test_split(dfX, dfY, test_size=0.10, random_state = np.random.randint(1,1000, 1)[0])

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

valid_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale = 1./255)

train_set = train_datagen.flow_from_directory('D:\Hackathon\Image Classification\train\images\train',
                                              target_size = (64, 64),
                                              batch_size = 32,
                                              classes = ['Cargo', 'Military', 'Carrier', 'Cruise', 'Tankers'],
                                              class_mode = 'categorical')

valid_set = valid_datagen.flow_from_directory('D:\Hackathon\Image Classification\train\images\valid',
                                              target_size=(64, 64),
                                              batch_size=16,
                                              classes = ['Cargo', 'Military', 'Carrier', 'Cruise', 'Tankers'],
                                              class_mode='categorical')

test_set = test_datagen.flow_from_directory('D:\Hackathon\Image Classification\train\images\test',
                                            target_size = (64, 64),
                                            batch_size = 1,
                                            classes = ['Cargo', 'Military', 'Carrier', 'Cruise', 'Tankers'],
                                            class_mode = 'categorical')

# Save Checkpoints
filepath="weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

classifier.fit_generator(train_set,
                         steps_per_epoch = 8000,
                         epochs = 25,
                         validation_data = valid_set,
                         validation_steps = 2000)

# Save Mordel
classifier.save_weights("model.h5")