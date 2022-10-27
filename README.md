# Poster Classifier
Determining the genre of each movie given only its poster by both supervised learning and unsupervised learning.  
Use a convolutional neural network (CNN) to classify posters by genre in the supervised setting.  
Then explore unsupervised pretraining, work exclusively with the image data to learn a feature representation (i.e., a continuous-valued feature vector) for each input image.

# Dataset
This dataset contains 10,000 PNG image files of 10 different genres of movie posters. These images are named as 00000.png through 09999.png, and stored under data/images/. Each image contains 3 color channels (red, green, blue) and is of size 128 × 128 × 3. These images have already been divided into 3 class-balanced data partitions: a training set, a validation set, and a test set. The metadata containing the label and data partition for each image is documented in data/posters.csv. Note that the test labels have been removed from this file. As in project 1, you will include your challenge model’s predictions on this data partition as a part of your submission.