import numpy as np


def my_pca_infer(para, X, x_center):
    len_para = para.shape
    # print(len_para)
    len_x = X.shape
    # print(len_x)
    # if len_x[1] != len_para[1]:
    #     return -1
    # else:
    X = np.reshape(X, (1, len_x[0]))
    x_center = np.reshape(x_center, (1, len_x[0]))
    # print(X.shape)
    len_x = X.shape
    out = np.zeros([len_x[0], len_para[0]])
    # print(out.shape)
    for i in range(1):
        out[i] = np.dot(X[i]-x_center[i], para.transpose())

    return out
