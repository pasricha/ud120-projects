#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
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

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Create an SVM classifier.
classifier = SVC(kernel='linear')

# Use a smaller dataset to speed up training.
features_train_small = features_train[:len(features_train)/100]
labels_train_small = labels_train[:len(labels_train)/100]

# Train the classifier using a linear kernel.
start_time = time.time()
classifier.fit(features_train_small, labels_train_small)
end_time = time.time()

print 'Training time (linear kernel): ', end_time - start_time, 'seconds'

# Make predictions.
start_time = time.time()
predictions = classifier.predict(features_test)
end_time = time.time()

print 'Prediction time: ', end_time - start_time, 'seconds'

# Compute accuracy of the predictions.
accuracy = accuracy_score(labels_test, predictions)
print 'Accuracy: ', accuracy

print
#########################################################

# Train the classifier using rbf kernel.
classifier = SVC(kernel='rbf')

start_time = time.time()
classifier.fit(features_train_small, labels_train_small)
end_time = time.time()

print 'Training time (rbf kernel): ', end_time - start_time, 'seconds'

# Make predictions.
start_time = time.time()
predictions = classifier.predict(features_test)
end_time = time.time()

print 'Prediction time: ', end_time - start_time, 'seconds'

# Compute accuracy of the predictions.
accuracy = accuracy_score(labels_test, predictions)
print 'Accuracy: ', accuracy

print
#########################################################

# Try different values of C.
C_values = [10., 100., 1000., 10000.]

for C in C_values:
    classifier = SVC(kernel='rbf', C=C)
    classifier.fit(features_train_small, labels_train_small)
    predictions = classifier.predict(features_test)
    accuracy = accuracy_score(predictions, labels_test)
    print 'C: ', C, ',', 'Accuracy: ', accuracy

print
#########################################################

# Train on the full data set again, after optimizing C.
classifier = SVC(kernel='rbf', C=10000.)

start_time = time.time()
classifier.fit(features_train, labels_train)
end_time = time.time()

print 'Training time (rbf & C=10000): ', end_time - start_time, 'seconds'

# Make predictions.
start_time = time.time()
predictions = classifier.predict(features_test)
end_time = time.time()

print 'Prediction time: ', end_time - start_time, 'seconds'

# Compute accuracy of the predictions.
accuracy = accuracy_score(labels_test, predictions)
print 'Accuracy: ', accuracy

print
#########################################################

# Count emails predicted to be by Chris.
chris = 0

for p in predictions:
    if p == 1:
        chris += 1

print 'Emails predicted to be by Chris: ', chris

print
#########################################################
# Predctions for 10th, 26th and 50th elements of test set using small dataset.
classifier = SVC(kernel='rbf', C=10000.)

start_time = time.time()
classifier.fit(features_train_small, labels_train_small)
end_time = time.time()

print 'Training time: ', end_time - start_time, 'seconds'

predictions = classifier.predict(features_test)
print 'Prediction for 10: ', predictions[10]
print 'Prediction for 26: ', predictions[26]
print 'Prediction for 50: ', predictions[50]

print
#########################################################
