import load_data
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import tree
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

# #######################################################################
# dwt
train_data = wavelets_f(train_data)

# #######################################################################
# pca
n_components = 0.97
t0 = time()
pca = PCA(n_components=n_components, whiten=True).fit(train_data)
print("done in %0.3fs" % (time() - t0))

# filename_pac = 'pca_4.sav'
# pickle.dump(pca, open(filename_pac, 'wb'))
# print('pca components saved!')
print("pca components: ", pca.components_.shape)


print("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time()
X_train_pca = pca.transform(train_data)

print("done in %0.3fs" % (time() - t0))

# ########################################################################
# ########################################################################
# dt
print("Fitting the classifier to the training set")
# 获得一个决策树分类器
clf = tree.DecisionTreeClassifier()
# 拟合
clf.fit(X_train_pca, train_label)
test_data = wavelets_f(test_data)
test_data = pca.transform(test_data)
prediction = clf.predict(test_data)
accuracy = clf.score(test_data, test_label)

print('accurancy : ', accuracy)
print('dep: ', clf.get_depth())
print('N: ', clf.get_n_leaves())

