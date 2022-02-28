import numpy as np
np.set_printoptions(threshold = np.inf)


def int2bit(a, one_more):
    binary_vec_temp = []
    if one_more:
        binary_vec = np.zeros(65)
    else:
        binary_vec = np.zeros(64)
    tag = 1
    while tag:
        bit = a % 2
        binary_vec_temp.append(bit)
        a = (a - bit) // 2
        if a == 1 or a == 0:
            tag = 0
    binary_vec_temp.append(1)
    k = 0
    for i in range(len(binary_vec_temp)):
        binary_vec[k] = binary_vec_temp[i]
        k += 1
    return binary_vec


def validate_int2bit(binary_vec):
    sum = 0
    for i in range(len(binary_vec)):
        sum += binary_vec[i] * (2**i)
    return sum


if __name__ == '__main__':
    a = 445528495458246354
    b = int2bit(a, 0)
    print(b)
    # np.savetxt('values/dwt_fixpoint/lambda_bit.txt', b, fmt='%d', newline='\n')
    a_val = validate_int2bit(b)
    print(a_val - 445528495458246354)
    # 这一段是计算witness c in dwt
    abs_y_prime = np.loadtxt('values/dwt_fixpoint/abs_y_prime.txt', delimiter=' ')
    c = []
    # for i in range(len(abs_y_prime)):
    #     # print(i)
    #     bit = int2bit(abs_y_prime[i], 0)
    #     c.append(bit)
    # np.savetxt('values/dwt_fixpoint/cij_fixpoint/cij_new.txt', c, fmt='%d', newline='\n')
    # 然后计算witness d in dwt
    d = []
    # for i in range(len(abs_y_prime)):
    # for i in range(1):
    #     print(i)
        # print(int(abs_y_prime[i]))
        # ele = int(abs_y_prime[i]) - 3355443 + 2**64
        # bit = int2bit(ele, 1)
        # a = validate_int2bit(bit)
        # print(bit)
        # d.append(bit)
        # f = open('values/dwt_fixpoint/dij_fixpoint/dij.txt', 'a')
        # print(bit, file=f)
        # f.close()
    # np.savetxt('values/dwt_fixpoint/dij_fixpoint/dij_new.txt', d, fmt='%d', newline='\n')


