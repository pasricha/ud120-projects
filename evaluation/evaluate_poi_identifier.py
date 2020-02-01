#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here

### Split data into training and test set.
from sklearn.model_selection import train_test_split

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

print 'Number of POIs in the test set:', sum(labels_test)
print 'Number of people in the test set: ', len(labels_test)
print 'Accuracy for predicting everyone as non-POI: ', \
    (len(labels_test) - sum(labels_test)) / len(labels_test)

### Fit a Decision Tree Classifier.
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Create a decision tree classifier.
classifier = DecisionTreeClassifier()

# Fit the classifier on the training data.
classifier.fit(features_train, labels_train)

# Make predictions.
predictions = classifier.predict(features_test, labels_test)

# Compute the accuracy of the model.
accuracy = accuracy_score(predictions, labels_test)
print 'Accuracy on the test set:', accuracy

# Find true positives among the predictions.
true_positives = 0
for i in range(len(predictions)):
    if predictions[i] == 1.0 and labels_test[i] == 1.0:
        true_positives += 1
print 'True positives: ', true_positives

# Compute the precision of the model.
precision = precision_score(predictions, labels_test)
print 'Precision on the test set:', precision

# Compute the recall of the model.
recall = recall_score(predictions, labels_test)
print 'Recall on the test set:', recall

# Made-up predictions and true labels for a hypothetical test set.
predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]

true_positives = 0
true_negatives = 0
false_positives = 0
false_negatives = 0

for i in range(len(predictions)):
    if predictions[i] == 1 and true_labels[i] == 1:
        true_positives += 1
    if predictions[i] == 0 and true_labels[i] == 0:
        true_negatives += 1
    if predictions[i] == 1 and true_labels[i] == 0:
        false_positives += 1
    if predictions[i] == 0 and true_labels[i] == 1:
        false_negatives += 1
print 'True positives in the hypothetical test set:', true_positives
print 'True negatives in the hypothetical test set:', true_negatives
print 'False positives in the hypothetical test set:', false_positives
print 'False negatives in the hypothetical test set:', false_negatives
print 'Precision on the hypothetical test set:', float(true_positives) / (true_positives + false_positives)
print 'Recall on the hypothetical test set:', float(true_positives) / (true_positives + false_negatives)

# My true positive rate is high, which means that when a "POI" is present in the test data,
# I am good at flagging him or her.

# My identifier doesn't have great "Precision", but it does have good "Recall". That means that, nearly
# every time a POI shows up in my test set, I am able to identify him or her.
# The cost of this is that I sometimes get some false positives, where non-POIs get flagged.

# My identifier doesn't have great "Recall", but it does have good "Precision". That means that whenever a POI
# gets flagged in my test set, I know with a lot of confidence that it's very likely to be a real POI
# and not a false alarm. On the other hand, the price I pay for this is that I sometimes miss real POIs,
# since I'm effectively reluctant to pull the trigger on edge cases.

# My identifier has a really great "F1 score".
# This is the best of both worlds. Both my false positive and false negative rates are "low", which means that
# I can identify POI's reliably and accurately. If my identifier finds a POI then the person is almost certainly
# a POI, and if the identifier does not flag someone, then they are almost certainly not a POI.
