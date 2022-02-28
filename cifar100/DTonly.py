from sklearn import tree
import numpy as np
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import load_data
import pickle
from load_data import unpickle
from extract_feature import wavelets_f
from sklearn.decomposition import PCA
from time import time

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
num_class = 32
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
num_samples_each = 70
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

if __name__ == '__main__':

    # 获得一个决策树分类器
    clf = tree.DecisionTreeClassifier()
    # 拟合
    clf.fit(train_sub_sub_data, train_sub_sub_label)
    # 预测
    prediction = clf.predict(test_sub_sub_data)

    print(prediction)

    accuracy = clf.score(test_sub_sub_data, test_sub_sub_label)
    # accurancy = np.sum(np.equal(prediction, test_label)) / len(test_label)
    print('prediction : ', prediction)
    print('accurancy : ', accuracy)
    print('dep: ', clf.get_depth())
    print('N: ', clf.get_n_leaves())
