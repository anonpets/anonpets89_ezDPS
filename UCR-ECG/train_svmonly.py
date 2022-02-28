import load_data
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from time import time
from extract_feature import wavelets_f
import numpy as np
from sklearn.model_selection import GridSearchCV

train_data, train_label, test_data, test_label = load_data.load_data(42)
train_data = np.array(train_data)
train_label = np.array(train_label)
test_data = np.array(test_data)
test_label = np.array(test_label)
print("Data load successfully!")
print("The train data shape is: ")
print(train_data.shape)
print("The test data shape is: ")
print(test_data.shape)
# ########################################################################
# svc
print("Fitting the classifier to the training set")
t0 = time()
clf = SVC(C=1000.0, class_weight='balanced', gamma=0.2)
# param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
#               'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
# clf = GridSearchCV(
#     SVC(kernel='rbf', class_weight='balanced'), param_grid
# )
clf = clf.fit(train_data, train_label)
# save the model to disk
# filename = 'pipeline_4.sav'
# pickle.dump(clf, open(filename, 'wb'))
# print("svc parameters saved!")

print("done in %0.3fs" % (time() - t0))
# print("Best estimator found by grid search:")
# print(clf.best_estimator_)
print("support vectors ", clf.support_vectors_.shape)

# #############################################################################
# Quantitative evaluation of the model quality on the test set

print("Predicting on the test set")
t0 = time()

y_pred = clf.predict(test_data)
print("done in %0.3fs" % (time() - t0))
print("Accuracy:", accuracy_score(test_label, y_pred))
