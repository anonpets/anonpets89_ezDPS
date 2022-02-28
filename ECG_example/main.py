import wfdb
import matplotlib.pyplot as plt
import numpy as np
import pywt
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

from read_ecg import read_ecg
from plot_ecg import plot_ecg
from extract_data import extract_data_from_train_file, extract_data_from_test_file
from sklearn.metrics import confusion_matrix, accuracy_score
from extract_feature import simple_f, wavelets_f
import joblib
import my_pca_infer
import os

# read all the training data and put them in one array
x_total = []
y_total = []
for i in range(100, 110):
    path = "data/" + str(i)
    full_path = path + ".hea"
    if not os.path.exists(full_path):
        continue
    else:
        # print(str(i))
        x, y = extract_data_from_train_file(path)
        x_total.extend(x)
        y_total.extend(y)

x_total = np.array(x_total)
y_total = np.array(y_total)
print(len(x_total.shape))
print(x_total.shape)
# feature extraction: simple feature and wavelets feature
n = len(x_total[:, 1])

trainingdata2 = wavelets_f(x_total)
x_feature = trainingdata2

# shuffle data
x_feature, y_total = shuffle(x_feature, y_total)
idx = int(len(x_feature) * 0.7)
x_train = x_feature[:idx]
x_valid = x_feature[idx:]
y_train = y_total[:idx]
y_valid = y_total[idx:]

# cross-validation for choosing model (svm)

# print('dimension of the raw data: ', x_train.shape)
clf_rbf = SVC(kernel='rbf', gamma='auto', shrinking=False)  # 该函数default值是ovr策略
pca = PCA(n_components=12)
model = make_pipeline(pca, clf_rbf)

# scores_rbf = cross_val_score(model, x_feature, y_total, cv=10, scoring='accuracy')
# print(scores_rbf.mean())

# machine learning model(PCA+SVM)
model.fit(x_train, y_train)
# joblib.dump(model, 'model_size2.pkl')
# np.set_printoptions(threshold=np.inf)

# load the well-trained model
# model = joblib.load('model_nosimple_mywavelet.pkl')
# print('the model step is:', model.get_params)

pred_valid = model.predict(x_valid)
print(accuracy_score(y_valid, pred_valid))
pca = model.get_params()['pca']
svc = model.get_params()['svc']
print("support_vectors: ", svc.support_vectors_.shape)

# print('classification report: ', classification_report(y_valid, pred_valid))
# fpr, tpr, thresholds = metrics.roc_curve(y_valid, pred_valid)
# print('auc: ', metrics.auc(fpr, tpr))

# 3. The assigned 'V' beat info shall be exported to WFDB format (*.test),
# and sent back to Biofourmis.
# for index in range(232, 233):
#     path = "data/" + str(232)
#     x_test, location = extract_data_from_test_file(path)
#     n = len(x_test)
#     x_test = np.array(x_test)
    # testingdata1 = simple_f(x_test, n)
    # print(len(x_test[0]))
    # np.savetxt('values/dwt/dwt_input.txt', x_test[0], fmt='%s', newline='\n')
    # testingdata2 = wavelets_f(x_test)

    # np.savetxt('values/dwt/dwt_result.txt', testingdata2[0], fmt='%s', newline='\n')
    # print(len(testingdata2[0]))
    # x_feature_test = np.hstack((testingdata1, testingdata2))
    # x_feature_test = testingdata2
    # x_feature_test_center = x_feature_test - np.mean(x_feature_test, axis=0)
    # x_feature_test_center = x_feature_test_center[0]
    # print(x_feature_test.shape)

    # x_feature_test = testingdata2
    # pca = model.get_params()['pca']
    # svc = model.get_params()['svc']
    # print(svc)

    # print(pca.components_.transpose())
    # my_result = my_pca_infer.my_pca_infer(pca.components_, x_feature_test_center, pca.mean_)
    # x_feature_test_center = np.array(x_feature_test_center).reshape((1, -1))
    # my_result_pca_sklearn = pca.transform(x_feature_test_center)
    # decision_function = svc.decision_function(my_result)
    # print(decision_function.shape)
    # predict = svc.predict(my_result)
    # decision_function = svc.decision_function(my_result)


"""
    the parameters of the pca function in sklearn
"""
# components_ : ndarray of shape (n_components, n_features)
    # Principal axes in feature space, representing the directions of maximum variance in the data.
    # The components are sorted by explained_variance_.
    # print(pca.components_.shape)
    # mean_:Per-feature empirical mean, estimated from the training set.
    # print(pca.mean_)
    # save the parameters
    # np.savetxt('values/pca_input.txt', x_feature_test_center, fmt='%s', newline='\n')
    # np.savetxt('values/pca_components.txt', pca.components_, fmt='%s', newline='\n')
    # np.savetxt('values/pca_result_sklearn.txt', my_result_pca_sklearn, fmt='%s', newline='\n')
    # np.savetxt('values/pca_mean_.txt', pca.mean_, fmt='%s', newline='\n')
    # np.savetxt('values/my_pca_result.txt', my_result, fmt='%s', newline='\n')


"""
    the parameters of the svc function in sklearn
"""
    # print(predict)
    # # n_support_:Number of support vectors for each class.
    # print("num_support vector: ", svc.n_support_)
    # # support_vectors_
    # print("support_vectors: ", svc.support_vectors_.shape)
    # # intercept_ndarray of shape (n_classes * (n_classes - 1) / 2,)： Constants in decision function.
    # print("b: ", svc.intercept_)
    # # gamma
    # print("gamma: ", svc.gamma)
    # # class_weight_：Multipliers of parameter C for each class. Computed based on the class_weight parameter.
    # print("class_weight: ", svc.class_weight_)
    # # dual_coef_ndarray of shape (n_classes -1, n_SV): Dual coefficients of the support vector in the decision function,
    # # multiplied by their targets. For multiclass, coefficient for all 1-vs-1 classifiers.
    # # The layout of the coefficients in the multiclass case is somewhat non-trivial.
    # print("dual_coef: ", svc.dual_coef_.shape)
    # # Evaluates the decision function for the samples in X.
    # print("decision_value: ", decision_function)

    # save the parameters
    # np.savetxt('values/support_vectors.txt', svc.support_vectors_, fmt='%s', newline='\n')
    # np.savetxt('values/dual_coef.txt', svc.dual_coef_, fmt='%s', newline='\n')
    # np.savetxt('values/my_pca_result.txt', my_result, fmt='%s', newline='\n')  # result after pca, before svc
    # np.savetxt('values/decision_function.txt', decision_function, fmt='%s', newline='\n')

    # print(pca.components_.shape)
    # np.savetxt('pca_para.txt', pca.components_, newline='\n')

    # print(pca.components_)
    # pca_result = pca.transform(x_feature_test)
    # np.savetxt('pca_output.txt', pca_result, fmt='%s', newline='\n')

    # predicted_labels = model.predict(x_feature_test)
    # ecg_sig, ecg_type, ecg_peak = read_ecg(path)
    # for i in range(len(location)):
    #     if predicted_labels[i] == 1:
    #         ecg_type[location[i]] = "V"
    #
    # name = "b" + str(index)
    # wfdb.wrann(name, 'test', ecg_peak, ecg_type, write_dir='result/')
