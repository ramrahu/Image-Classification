# -*- coding: utf-8 -*-
"""
Created on Sat May 25 21:04:40 2019

@author: Aspire V5
"""
"""
This file is used to separate train and test set into
different folders based on given csv files

"""
# Import the packages
import pandas as pd
import os
import shutil

path = []

# Read CSV file for test set images
test_dataset = pd.read_csv("train.csv")

# Divide into train & test sets
for root, dirs, files in os.walk("images/"):
    for image in files:
        if os.path.splitext(image)[1].lower() in ('.jpg', '.jpeg'):
            for i in test_dataset['image'].tolist():
                if image == i:
                    path.append(os.path.join(root, image))
                
if not os.path.exists("images/test") and not os.path.exists("images/train"):
    os.mkdir("images/test")
    os.mkdir("images/train")

for file in path:
    shutil.copy(file, "images/train")

# Read CSV file for test set images
train_dataset = pd.read_csv("train.csv")

# Divide into train & test sets
for root, dirs, files in os.walk("images/train/"):
    for image in files:
        if os.path.splitext(image)[1].lower() in ('.jpg', '.jpeg'):
            path.append(os.path.join(root, image))

categories = ['Cargo', 'Military', 'Carrier', 'Cruise', 'Tankers']

for ctg in categories:
    if not os.path.exists(os.path.join("images/train", ctg)):
        os.mkdir(os.path.join("images/train", ctg))

for file in path:
    for i in train_dataset['image'].tolist():
        if file.split('/')[1] == i:
            category = train_dataset['category'].tolist()[train_dataset['image'].tolist().index(file.split('/')[1])]
            shutil.copy(file, os.path.join("images/train/", categories[category-1]))