import wfdb
import pywt
import numpy as np

# signals, fields = wfdb.rdsamp('data/100')
# annotation = wfdb.rdann('data/100', 'atr')
# print(signals)
# print('=========================')
# print(annotation)


def wavelets_f(x_total, threshold=0.2):
    coeffs = pywt.wavedec(x_total, 'db4', level=1)
    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], threshold * max(coeffs[i]))
        print(coeffs[i].shape)
        # print(max(coeffs[i]))
    # 将信号进行小波重构
    datarec = pywt.waverec(coeffs, 'db4')
    featuredata = np.array(datarec)
    return featuredata


x = [1, 2, 3, 4, 5]
wavelets_f(x)
