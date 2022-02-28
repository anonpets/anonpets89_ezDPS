import numpy as np
import int2bit


def compute_svc_temp(x, sv):
    if len(x) != len(sv[0]):
        raise ValueError("The dimension of x and sv are not compatible")
    else:
        print(len(sv))
        print(len(sv[0]))
        temp = np.zeros((len(sv), len(sv[0])))
        for i in range(len(sv)):
            for j in range(len(sv[0])):
                temp[i][j] = (x[j] - sv[i][j]) * (x[j] - sv[i][j])
        return temp


def compute_svc_d(temp, lamb):
    d = np.zeros(len(sv))
    for i in range(len(sv)):
        tp = 0
        for j in range(len(sv[0])):
            tp += temp[i][j]
        d[i] = tp * lamb
    return d


def compute_dij(x):
    d = []
    for i in range(len(x)):
        print(i)
        bit = int2bit.int2bit(x[i], 0)
        d.append(bit)
    np.savetxt('values/svc_fixpoint/dij_fixpoint.txt', d, fmt='%d', newline='\n')


def compute_Poe():
    e = 1
    Poe = []
    Poe.append(e)
    for i in range(63):
        e = e * e
        Poe.append(e)


def compute_mij():
    m = np.ones([545, 63])
    # print(m)
    np.savetxt("values/svc_fixpoint/svc_mij.txt", m, fmt="%d", newline='\n')

if __name__ == '__main__':
    x = np.loadtxt("values/svc_fixpoint/svc_input_fixpoint.txt", delimiter=' ')
    sv = np.loadtxt("values/svc_fixpoint/support_vectors_fixpoint.txt", delimiter=' ')
    temp = compute_svc_temp(x, sv)
    # np.savetxt("values/svc_fixpoint/svc_temp.txt", temp, fmt="%d", newline='\n')
    # print(len(temp))
    lamb = 29127
    d = np.loadtxt("values/svc_fixpoint/svc_d_rust_compute.txt", delimiter=",")
    compute_dij(d)
    # compute_Poe()
    # compute_mij()
