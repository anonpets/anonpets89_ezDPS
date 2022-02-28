from load_data import unpickle
from time import time
import logging
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.utils import shuffle
import numpy as np
from extract_feature import wavelets_f


train_path = 'cifar-100-python/train'
test_path = 'cifar-100-python/test'
train = unpickle(train_path)
test = unpickle(test_path)

# for item in test:
#     print(item)

print(train['data'].shape)
# print(train['fine_labels'])

train_data = np.array(train['data'])
train_label = np.array(train['fine_labels'])
test_data = np.array(test['data'])
test_label = np.array(test['fine_labels'])

# ######################################################################
# control the number of the classes involved in the training procedure
num_class = 100
train_sub_data = []
train_sub_label = []
test_sub_data = []
test_sub_label = []
for i in range(len(train_label)):
    if train_label[i] < num_class:
        train_sub_label.append(train_label[i])
        train_sub_data.append(train_data[i])
for i in range(len(test_label)):
    if test_label[i] < num_class:
        test_sub_label.append(test_label[i])
        test_sub_data.append(test_data[i])
print(len(train_sub_label))
print(np.array(train_sub_data).shape)

# ######################################################################
# control the number of samples in each class
num_samples_each = 20
num_test_samples = 200
train_sub_sub_data = []
train_sub_sub_label = []
test_sub_sub_data = []
test_sub_sub_label = []
for i in range(num_class):
    print(i)
    count = 0
    for j in range(len(train_sub_label)):
        if train_sub_label[j] == i and count < num_samples_each:
            train_sub_sub_label.append(train_sub_label[j])
            train_sub_sub_data.append(train_sub_data[j])
            count = count + 1
print(np.array(train_sub_sub_data).shape)
# train_sub_sub_data = np.array(train_sub_sub_data)
# train_sub_sub_label = np.array(train_sub_sub_label)

for i in range(num_test_samples):
    test_sub_sub_data.append(test_sub_data[i])
    test_sub_sub_label.append(test_sub_label[i])
print(np.array(test_sub_sub_data).shape)

# mixed dataset
total_test_sample = 80
test_mix_data = []
test_mix_label = []

train_ratio = 0.6
for i in range(int(total_test_sample * train_ratio)):
    test_mix_data.append(train_sub_sub_data[i])
    test_mix_label.append(train_sub_sub_label[i])
for i in range(int(total_test_sample * (1 - train_ratio))):
    test_mix_data.append(test_sub_sub_data[i])
    test_mix_label.append(test_sub_sub_label[i])


# #######################################################################
# pca
# n_components = 0.9
# t0 = time()
# pca = PCA(n_components=n_components, whiten=True).fit(train_sub_sub_data)
# print("done in %0.3fs" % (time() - t0))
#
# filename_pac = 'pca_4.sav'
# pickle.dump(pca, open(filename_pac, 'wb'))
# print('pca components saved!')
# print("pca components: ", pca.components_.shape)
#
#
# print("Projecting the input data on the eigenfaces orthonormal basis")
# t0 = time()
# X_train_pca = pca.transform(train_sub_sub_data)
#
# print("done in %0.3fs" % (time() - t0))

# ########################################################################
# svc
print("Fitting the classifier to the training set")
t0 = time()
clf = SVC(C=1000.0, cache_size=200, class_weight='balanced', coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=0.01, kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
# param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
#               'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
# clf = GridSearchCV(
#     SVC(kernel='rbf', class_weight='balanced'), param_grid
# )
print(np.array(train_sub_sub_label).shape)
clf = clf.fit(train_sub_sub_data, train_sub_sub_label)
# save the model to disk
filename = 'svm_16.sav'
pickle.dump(clf, open(filename, 'wb'))
print("svc parameters saved!")

print("done in %0.3fs" % (time() - t0))
# print("Best estimator found by grid search:")
# print(clf.best_estimator_)
print("support vectors ", clf.support_vectors_.shape)

# #############################################################################
# Quantitative evaluation of the model quality on the test set

print("Predicting people's names on the test set")
t0 = time()
# X_test_dwt = wavelets_f(test_sub_sub_data)
# X_test_pca = pca.transform(X_test_dwt)
y_pred = clf.predict(test_sub_sub_data)
print("done in %0.3fs" % (time() - t0))
print("Accuracy:", accuracy_score(test_sub_sub_label, y_pred))

print("Predicting people's names on the training set")
t0 = time()
# X_train_dwt = wavelets_f(train_sub_sub_data)
# X_train_pca_new = pca.transform(X_train_dwt)
y_pred_train = clf.predict(train_sub_sub_data)
print("done in %0.3fs" % (time() - t0))
print("Accuracy:", accuracy_score(train_sub_sub_label, y_pred_train))

# #############################################################################
# accuracy on the mix dataset
print("Predicting on the mix dataset")
# X_test_mix_dwt = wavelets_f(test_mix_data)
# X_test_mix_pca = pca.transform(X_test_mix_dwt)
y_pred_mix = clf.predict(test_mix_data)
print("Accuracy:", accuracy_score(test_mix_label, y_pred_mix))

