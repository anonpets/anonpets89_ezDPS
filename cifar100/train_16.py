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
test_data_or = np.array(test['data'])
test_label_or = np.array(test['fine_labels'])
print(test_label_or)

# data = []
# label = []
# ######################################################################
# control the number of the classes involved in the training procedure
num_class = 16
train_sub_data = []
train_sub_label = []
test_sub_data = []
test_sub_label = []
for i in range(len(train_label)):
    if train_label[i] < num_class:
        train_sub_label.append(train_label[i])
        train_sub_data.append(train_data[i])
for i in range(len(test_label_or)):
    if test_label_or[i] < num_class:
        test_sub_label.append(test_label_or[i])
        test_sub_data.append(test_data_or[i])
print(len(train_sub_label))
print(np.array(train_sub_data).shape)
########################################################################
test_data = []
test_label = []
ratio = 0.04
shuf = np.loadtxt('shuffledata_16.txt', dtype=int)
for i in shuf:
    print(i)
    test_data.append(test_sub_data[i])
    test_label.append(test_sub_label[i])
num = int(ratio * len(test_sub_label))
count = 0
for i in range(len(test_sub_label)):
    if i not in shuf and count < num:
        test_data.append(test_sub_data[i])
        test_label.append(test_sub_label[i])
        count = count + 1
# ######################################################################

# #######################################################################
# pca
# n_components = 0.93
# t0 = time()
# pca = PCA(n_components=n_components, whiten=True).fit(train_sub_data)
# print("done in %0.3fs" % (time() - t0))

# filename_pac = 'pca_16.sav'
# pickle.dump(pca, open(filename_pac, 'wb'))
# print('pca components saved!')
# print("pca components: ", pca.components_.shape)


# print("Projecting the input data on the eigenfaces orthonormal basis")
# t0 = time()
pca = pickle.load(open('pca_16.sav', 'rb'))
# X_train_pca = pca.transform(train_sub_data)

# print("done in %0.3fs" % (time() - t0))

# ########################################################################
# svc
# print("Fitting the classifier to the training set")
# t0 = time()
# clf = SVC(C=1000.0, cache_size=200, class_weight='balanced', coef0=0.0,
#     decision_function_shape='ovr', degree=3, gamma=0.01, kernel='rbf',
#     max_iter=-1, probability=False, random_state=None, shrinking=True,
#     tol=0.001, verbose=False)
# param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
#               'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
# clf = GridSearchCV(
#     SVC(kernel='rbf', class_weight='balanced'), param_grid
# )
# print(np.array(train_label).shape)
# clf = clf.fit(X_train_pca, train_sub_label)
# save the model to disk
# filename = 'pipeline_16.sav'
# pickle.dump(clf, open(filename, 'wb'))
# print("svc parameters saved!")

# print("done in %0.3fs" % (time() - t0))
# print("Best estimator found by grid search:")
# print(clf.best_estimator_)
# print("support vectors ", clf.support_vectors_.shape)
clf = pickle.load(open('pipeline_16.sav', 'rb'))
# #############################################################################
# Quantitative evaluation of the model quality on the test set


print("Predicting on the test set")
t0 = time()
X_test_dwt = wavelets_f(test_data)
X_test_pca = pca.transform(X_test_dwt)
y_pred = clf.predict(X_test_pca)
print("done in %0.3fs" % (time() - t0))
print("Accuracy:", accuracy_score(test_label, y_pred))
