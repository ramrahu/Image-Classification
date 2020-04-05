# Image-Classification
CNN based classifier to classify images of vessels into 5 different categories. This solution was for a hackathon organized by Analytics Vidhya.

# Problem Statement
Ship or vessel detection has a wide range of applications, in the areas of maritime safety, fisheries management, marine pollution, defence and maritime security, protection from piracy, illegal migration, etc.

Keeping this in mind, a Governmental Maritime and Coastguard Agency is planning to deploy a computer vision based automated system to identify ship type only from the images taken by the survey boats. You have been hired as a consultant to build an efficient model for this project.

There are 5 classes of ships to be detected which are as follows:
1. Cargo
2. Military
3. Tanker
4. Carrier
5. Cruise

# Dataset Description
There are 6252 train images and 2680 images in the test set.he categories of ships and their corresponding codes in the dataset are as follows:

'Cargo' -> 1
'Military' -> 2
'Carrier' -> 3
'Cruise' -> 4
'Tankers' -> 5

There are 2 files provided to us along with the dataset:
1. train.csv: Train dataset
2 .test_ApKoW4T.csv: Test dataset

Code Files:
1. Data_create.py: Divide dataset into train and test folders based on csv files
2. Classify.py: CNN architecture to classify the images

# Evaluation Metric
The evaluation metrics for this competition was weighted F1 score
