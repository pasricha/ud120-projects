#!/usr/bin/python

import numpy as np
import os
import sys
import pickle
import matplotlib.pyplot as plt

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

from parse_out_email_text import parseOutText

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'salary', 'bonus', 'total_stock_value', 'exercised_stock_options',
                'from_poi_to_this_person', 'from_this_person_to_poi', 'shared_receipt_with_poi']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
# Remove the TOTAL outlier
data_dict.pop("TOTAL", 0)

# Visualize the salary and bonus in context of being POI.
for person in data_dict.keys():
    salary = data_dict[person][features_list[1]]
    bonus = data_dict[person][features_list[2]]

    c = 'red' if data_dict[person]['poi'] else 'blue'
    plt.scatter(salary, bonus, color=c)

plt.xlabel('salary')
plt.ylabel('bonus')
plt.show()

# Visualize the total stock value and exercised stock options in context of being POI.
for person in data_dict.keys():
    total_stock_value = data_dict[person][features_list[3]]
    exercised_stock_options = data_dict[person][features_list[4]]

    c = 'red' if data_dict[person]['poi'] else 'blue'
    plt.scatter(total_stock_value, exercised_stock_options, color=c)

plt.xlabel('total stock value')
plt.ylabel('exercised stock options')
plt.show()

# Visualize the number of emails received from poi and emails sent to poi.
for person in data_dict.keys():
    emails_from_poi = data_dict[person][features_list[5]]
    email_to_poi = data_dict[person][features_list[6]]

    c = 'red' if data_dict[person]['poi'] else 'blue'
    plt.scatter(emails_from_poi, email_to_poi, color=c)

plt.xlabel('emails from poi')
plt.ylabel('email to poi')
plt.show()

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3, weights='distance')

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

# Fit the classifier with the training data.
clf.fit(features_train, labels_train)

predictions = clf.predict(features_test)

from sklearn.metrics import accuracy_score, precision_score, recall_score
print 'accuracy on test set: ', accuracy_score(labels_test, predictions)
print 'precision on test set: ', precision_score(labels_test, predictions)
print 'recall on test set: ', recall_score(labels_test, predictions)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)