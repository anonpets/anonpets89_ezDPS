from sklearn import tree
import numpy as np
from sklearn.datasets import fetch_lfw_people
from count_class import count_class
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from time import time
import pickle
from extract_feature import wavelets_f
from PIL import Image

lfw_people = fetch_lfw_people(min_faces_per_person=11, resize=0.7)

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
count = 15
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
    np.array(X_sub), np.array(Y_sub), test_size=0.02, random_state=42)

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.03, random_state=42)
print("shape of X_train: ", X_train.shape)
print("shape of X_test: ", X_test.shape)
X_train, y_train = shuffle(X_train, y_train)


if __name__ == '__main__':
    # pca处理数据
    n_components = 0.99

    print("Extracting the top %d eigenfaces from %d faces"
          % (n_components, X_train.shape[0]))
    t0 = time()
    pca = PCA(n_components=n_components, whiten=True).fit(X_train)
    print("done in %0.3fs" % (time() - t0))

    filename_pac = 'pca_dt4.sav'
    pickle.dump(pca, open(filename_pac, 'wb'))
    print('pca components saved!')
    print("pca components: ", pca.components_.shape)

    print("Projecting the input data on the eigenfaces orthonormal basis")
    t0 = time()
    X_train = pca.transform(X_train)

    print("done in %0.3fs" % (time() - t0))

    # 获得一个决策树分类器
    clf = tree.DecisionTreeClassifier()
    # 拟合
    clf.fit(X_train, y_train)
    # dwt
    X_test = wavelets_f(X_test)
    # pca
    X_test = pca.transform(X_test)
    # 预测
    prediction = clf.predict(X_test)
    print(prediction)

    accurancy = np.sum(np.equal(prediction, y_test)) / len(X_test)
    print('prediction : ', prediction)
    print('accurancy : ', accurancy)
    print('dep: ', clf.get_depth())
    print('N: ', clf.get_n_leaves())