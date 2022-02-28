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
from extract_feature import wavelets_f
from sklearn.utils import shuffle
import numpy as np
from count_class import count_class

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


# #############################################################################
# Download the data, if not already on disk and load it as numpy arrays

lfw_people = fetch_lfw_people(min_faces_per_person=7, resize=0.7)

# introspect the images arrays to find the shapes (for plotting)
n_samples, h, w = lfw_people.images.shape
print("h", h)
print("w", w)

# for machine learning we use the 2 data directly (as relative pixel
# positions info is ignored by this model)
X = lfw_people.data
n_features = X.shape[1]

# the label to predict is the id of the person
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]
# print(y)

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)

# #############################################################################
# use the subset that contains the power-of-two number of classes
count = 0
X_sub = []
Y_sub = []

cur = 0  # cur指示当前项
lab = []
while count != 0:
    if y[cur] not in lab:
        lab.append(y[cur])
        print(lab)
        count = count - 1
    cur += 1

for i in range(len(y)):
    if y[i] not in lab:
        X_sub.append(X[i])
        Y_sub.append(y[i])

print(np.array(X_sub).shape)
print(np.array(Y_sub).shape)
print(count_class(Y_sub))

# #############################################################################
# Split into a training set and a test set using a stratified k fold

# split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(
    np.array(X_sub), np.array(Y_sub), test_size=0.01, random_state=42)

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.03, random_state=42)
print("shape of X_train: ", X_train.shape)
print("shape of X_test: ", X_test.shape)
X_train, y_train = shuffle(X_train, y_train)
# #############################################################################
# Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
# dataset): unsupervised feature extraction / dimensionality reduction
n_components = 0.85

print("Extracting the top %d eigenfaces from %d faces"
      % (n_components, X_train.shape[0]))
t0 = time()
pca = PCA(n_components=n_components, whiten=True).fit(X_train)
print("done in %0.3fs" % (time() - t0))

filename_pac = 'pca.sav'
pickle.dump(pca, open(filename_pac, 'wb'))
print('pca components saved!')
print("pca components: ", pca.components_.shape)


print("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time()
X_train_pca = pca.transform(X_train)

print("done in %0.3fs" % (time() - t0))


# #############################################################################
# Train a SVM classification model

print("Fitting the classifier to the training set")
t0 = time()
clf = SVC(C=1000.0, cache_size=200, class_weight='balanced', coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=0.005, kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
# param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
#               'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
# clf = GridSearchCV(
#     SVC(kernel='rbf', class_weight='balanced'), param_grid
# )
clf = clf.fit(X_train_pca, y_train)
# save the model to disk
filename = 'pipeline_10_04.sav'
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
X_test_dwt = wavelets_f(X_test)
X_test_pca = pca.transform(X_test_dwt)
y_pred = clf.predict(X_test_pca)
print("done in %0.3fs" % (time() - t0))

# print(classification_report(y_test, y_pred, target_names=target_names))
# print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))
print("Accuracy:", accuracy_score(y_test, y_pred))

# X_train_dwt = wavelets_f(X_train)
#
# X_train_dwt_pca = pca.transform(X_train_dwt)
# y_train_pred = clf.predict(X_train_dwt_pca)
# print("Accuracy for train: ", accuracy_score(y_train, y_train_pred))

X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(
    X, y, test_size=0.15, random_state=42)

X_test_new_dwt = wavelets_f(X_test_new)
X_test_new_dwt_pca = pca.transform(X_test_new_dwt)
y_test_new_pred = clf.predict(X_test_new_dwt_pca)
print("Accuracy for new test: ", accuracy_score(y_test_new, y_test_new_pred))
