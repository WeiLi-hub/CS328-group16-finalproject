# -*- coding: utf-8 -*-
"""
This is the script used to train an activity recognition 
classifier on accelerometer data.

"""

import os
import sys
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import export_graphviz
from features import extract_features
from util import slidingWindow, reorient, reset_vars
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import pickle

import labels


# %%---------------------------------------------------------------------------
#
#		                 Load Data From Disk
#
# -----------------------------------------------------------------------------

print("Loading data...")
sys.stdout.flush()
data_file = 'data/all_labeled_data.csv'
data = np.genfromtxt(data_file, delimiter=',')
print("Loaded {} raw labelled activity data samples.".format(len(data)))
sys.stdout.flush()

sys.stdout.flush()
data_file_gyro = 'data/all_labeled_data_gyro.csv'
data_gyro = np.genfromtxt(data_file_gyro, delimiter=',')
print("Loaded {} raw labelled activity data samples.".format(len(data_gyro)))
sys.stdout.flush()

# %%---------------------------------------------------------------------------
#
#		                    Pre-processing
#
# -----------------------------------------------------------------------------

print("Reorienting accelerometer data...")
sys.stdout.flush()
reset_vars()
reoriented = np.asarray([reorient(data[i,2], data[i,3], data[i,4]) for i in range(len(data))])
reoriented_data_with_timestamps = np.append(data[:,0:2],reoriented,axis=1)
data = np.append(reoriented_data_with_timestamps, data[:,-1:], axis=1)

data = np.nan_to_num(data)

sys.stdout.flush()
reset_vars()
reoriented = np.asarray([reorient(data_gyro[i,2], data_gyro[i,3], data_gyro[i,4]) for i in range(len(data_gyro))])
reoriented_data_gyro_with_timestamps = np.append(data_gyro[:,0:2],reoriented,axis=1)
data_gyro = np.append(reoriented_data_gyro_with_timestamps, data_gyro[:,-1:], axis=1)

data_gyro = np.nan_to_num(data_gyro)

# %%---------------------------------------------------------------------------
#
#		                Extract Features & Labels
#
# -----------------------------------------------------------------------------

window_size = 200
step_size = 20

# sampling rate should be about 100 Hz (sensor logger app); you can take a brief window to confirm this
n_samples = 1000
time_elapsed_seconds = (data[n_samples,1] - data[0,1])
sampling_rate = n_samples / time_elapsed_seconds

print("Sampling Rate: " + str(sampling_rate))

# TODO: list the class labels that you collected data for in the order of label_index (defined in labels.py)
class_names = labels.activity_labels

print("Extracting features and labels for window size {} and step size {}...".format(window_size, step_size))
sys.stdout.flush()

X = []
Y = []

# accel
feature_names = []
for i,window_with_timestamp_and_label in slidingWindow(data, window_size, step_size):
    window = window_with_timestamp_and_label[:,2:-1]
    # print("window = ")
    # print(window)
    feature_names, x = extract_features(window)
    X.append(x)
    Y.append(window_with_timestamp_and_label[10, -1])
# print(np.array(X).shape)

temp_x = []
fn = []

# gyro
for i,window_with_timestamp_and_label in slidingWindow(data_gyro, window_size, step_size):
    window = window_with_timestamp_and_label[:,2:-1]
    fn, x = extract_features(window)
    temp_x.append(x)
# print(np.array(temp_x).shape)


# append gyro features to the feature name
for f in fn:
    feature = f + '_gyro'
    feature_names.append(feature)

# append gyro features with the accelerometer features

for i in range(len(X)):
    accels_x = X[i]
    for j in range(len(temp_x[0])):
        accels_x.append(temp_x[i][j])
    
X = np.asarray(X)
Y = np.asarray(Y)
n_features = len(X)
# print(X.shape)


print("Finished feature extraction over {} windows".format(len(X)))
print("Unique labels found: {}".format(set(Y)))
print("\n")
sys.stdout.flush()

# %%---------------------------------------------------------------------------
#
#		                Train & Evaluate Classifier
#
# -----------------------------------------------------------------------------


"""
TODO: iterating over each fold, fit a decision tree classifier on the training set.
Then predict the class labels for the test set and compute the confusion matrix
using predicted labels and ground truth values. Print the accuracy, precision and recall
for each fold.
"""

# TODO: calculate and print the average accuracy, precision and recall values over all 10 folds



# print("\n")
# print("---------------------- Random Forest Classifier -------------------------")
# total_accuracy = 0.0
# total_precision = [0.0, 0.0, 0.0]
# total_recall = [0.0, 0.0, 0.0]

# cv = sklearn.model_selection.KFold(n_splits=10, random_state=None, shuffle=True)
# for i, (train_index, test_index) in enumerate(cv.split(X)):
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = Y[train_index], Y[test_index]
#     print("Fold {} : Training Random Forest classifier over {} points...".format(i, len(y_train)))
#     sys.stdout.flush()
#     clf = RandomForestClassifier(n_estimators=100)
#     clf.fit(X_train, y_train)

#     print("Evaluating classifier over {} points...".format(len(y_test)))
#     # predict the labels on the test data
#     y_pred = clf.predict(X_test)

#     # show the comparison between the predicted and ground-truth labels
#     conf = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
#     # print(conf)

#     accuracy = np.sum(np.diag(conf)) / float(np.sum(conf))
#     precision = np.nan_to_num(np.diag(conf) / np.sum(conf, axis=0))
#     recall = np.nan_to_num(np.diag(conf) / np.sum(conf, axis=1))
#     print("Precision = ",precision)
#     print("Recall = ", recall)
#     total_accuracy += accuracy
#     total_precision += precision
#     total_recall += recall

# print("The average accuracy is {}".format(total_accuracy / 10.0))
# print("The average precision is {}".format(total_precision / 10.0)) # added sum to remove type error
# # (might have to change 10)
# print("The average recall is {}".format(total_recall / 10.0)) # added sum() to remove type error
# # (might have to change 10)

# # TODO: (optional) train other classifiers and print the average metrics using 10-fold cross-validation

# # Set this to the best model you found, trained on all the data:
# best_classifier = RandomForestClassifier(n_estimators=100)
# best_classifier.fit(X, Y)

# print("saving classifier model...")
# with open('classifier.pickle', 'wb') as f:
#     pickle.dump(best_classifier, f)


# # # %%



# TODO: split data into train and test datasets using 10-fold cross validation

# """
# TODO: iterating over each fold, fit a decision tree classifier on the training set.
# Then predict the class labels for the test set and compute the confusion matrix
# using predicted labels and ground truth values. Print the accuracy, precision and recall
# for each fold.
# """

# TODO: calculate and print the average accuracy, precision and recall values over all 10 folds
accuracy = 0
precision = 0
recall = 0
f1 = 0


cv = sklearn.model_selection.KFold(n_splits=10, random_state=None, shuffle=True)

tree = sklearn.tree.DecisionTreeClassifier(criterion="entropy", max_depth=4)

for i, (train_index, test_index) in enumerate(cv.split(X, Y)):
    
    X_train, X_test, y_train, y_test = X[train_index], X[test_index], Y[train_index], Y[test_index]
    tree.fit(X_train, y_train)
    
    y_pred = tree.predict(X_test)
    cm = sklearn.metrics.confusion_matrix(y_true=y_test, y_pred=y_pred)

    accuracy += accuracy_score(y_test, y_pred)
    precision += precision_score(y_test, y_pred, average='macro')
    recall +=  recall_score(y_test, y_pred, average='macro')
    f1 +=f1_score(y_test, y_pred, average='macro')

    print(cm)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels.activity_labels)
    # disp.plot()
    # plt.show()
    # plt.savefig('confusion_matrix.png')

    print('accuracy: ', accuracy_score(y_test, y_pred))
    print('precision: ', precision_score(y_test, y_pred, average='macro'))
    print('recall: ', recall_score(y_test, y_pred, average='macro'))
    print('f1: ', f1_score(y_test, y_pred, average='macro'))


print("The average accuracy is {}".format(accuracy / 10.0))
print("The average precision is {}".format(precision / 10.0))
print("The average recall is {}".format(recall / 10.0))
print("The average f1 is {}".format(f1 / 10.0))




# TODO: train the decision tree classifier on entire dataset
tree.fit(X, Y)

y_pred = tree.predict(X)
cm = sklearn.metrics.confusion_matrix(y_true=Y, y_pred=y_pred)

accuracy += accuracy_score(Y, y_pred)
precision += precision_score(Y, y_pred, average='macro')
recall +=  recall_score(Y, y_pred, average='macro')
f1 +=f1_score(Y, y_pred, average='macro')

print(cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels.activity_labels)
disp.plot()
# plt.show()
plt.savefig('confusion_matrix.png') 



# TODO: Save the decision tree visualization to disk - replace 'tree' with your decision tree and run the below line
export_graphviz(tree, out_file='tree.dot', feature_names = feature_names)
# print(feature_names)
# dot -Tpng tree.dot -o tree.png

# TODO: Save the classifier to disk - replace 'tree' with your decision tree and run the below line
print("saving classifier model...")
with open('classifier.pickle', 'wb') as f:
    pickle.dump(tree, f)
# %%
