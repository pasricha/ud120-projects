#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

import time

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

print 'Number of features: ', len(features_train[0])
print

# Create a Decision Tree classifier.
classifier = DecisionTreeClassifier(min_samples_split=40)

# Train the classifier.
start_time = time.time()
classifier.fit(features_train, labels_train)
end_time = time.time()

print 'Training time: ', end_time - start_time, 'seconds'

# Make predictions.
start_time = time.time()
predictions = classifier.predict(features_test)
end_time = time.time()

print 'Prediction time: ', end_time - start_time, 'seconds'

# Compute accuracy of the predictions.
accuracy = accuracy_score(labels_test, predictions)
print 'Accuracy: ', accuracy

#########################################################


