
import numpy as np

alpha = [1, 2, 3, 4, 5, 6, 7, 8, 9]


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


# compute the mediate values in pca
def compute_pca_mv(para, X, x_center):
    out = np.zeros(len(X))
    for i in range(0, 170):
        temp1 = 0
        for j in range(0, 9):
            temp1 += para[j][i] * alpha[j]
        temp2 = X[i] - x_center[i]
        out[i] = temp2 * temp1
    return out


def validation_pca(temp, output):
    left = 0
    for i in range(0, 170):
        left += temp[i]
    right = 0
    for i in range(0, 9):
        right += output[0][i] * alpha[i]
    return [left, right]


if __name__ == '__main__':
    X = np.loadtxt('values/pca_fixpoint/pca_input_fixpoint.txt', delimiter=' ')
    x_center = np.loadtxt('values/pca_fixpoint/pca_mean_fixpoint.txt', delimiter=' ')
    para = np.loadtxt('values/pca_fixpoint/pca_component_fixpoint.txt', delimiter=' ')
    out = my_pca_infer(para, X, x_center)
    # print(len(out[0]))
    # np.savetxt('values/pca_fixpoint/pca_result_fp_cp.txt', out, fmt='%d', newline='\n')
    pca_mv = compute_pca_mv(para, X, x_center)
    np.savetxt('values/pca_fixpoint/pca_mv_fp.txt', pca_mv, fmt='%d', newline='\n')
    # print(len(pca_mv))
    result = validation_pca(pca_mv, out)
    print(result)
