import numpy as np
import int2bit
# 参数h和g的值，浮点数表示:
# h3 = -0.1294095226
# h2 = 0.2241438680
# h1 = 0.8365163037
# h0 = 0.4829629131
#
# g3 = -0.4829629131
# g2 = 0.8365163037
# g1 = -0.2241438680
# g0 = -0.1294095226
#
# Ih3 = g0
# Ih2 = h0
# Ih1 = g2
# Ih0 = h2
#
# Ig3 = g1
# Ig2 = h1
# Ig1 = g3
# Ig0 = h3
#
# h_floating_p = [h0, h1, h2, h3]
# g_floating_p = [g0, g1, g2, g3]

# 参数h和g的定点数表示

# 小波的decomp
def wavelets_decomp(data, h, g):
    n = len(data)
    data_out = np.zeros(len(data))
    if n < 4:
        raise ValueError("The dimension of the data is less than 4")
    else:
        half = n >> 1
        temp = np.zeros(n)
        i = 0
        for j in range(0, n - 3, 2):
            temp[i] = data[j] * h[0] + data[j + 1] * h[1] + data[j + 2] * h[2] + data[j + 3] * h[3]
            temp[i + half] = data[j] * g[0] + data[j + 1] * g[1] + data[j + 2] * g[2] + data[j + 3] * g[3]
            i = i + 1
        temp[i] = data[n - 2] * h[0] + data[n - 1] * h[1] + data[0] * h[2] + data[1] * h[3]
        temp[i + half] = data[n - 2] * g[0] + data[n - 1] * g[1] + data[0] * g[2] + data[1] * g[3]
        for i in range(n):
            data_out[i] = temp[i]
        # level = level - 1
    return data_out


# 小波的thresholding
def my_threshold(data, thre):
    data_out = []
    for i in range(len(data)):
        if data[i] >= 0:
            abs_data = data[i]
        else:
            abs_data = -data[i]
        temp = data[i] // abs_data
        comp = abs_data - thre
        if comp > 0:
            data_out.append(temp * comp)
        else:
            data_out.append(0)
    return data_out


# 小波变换的逆变换
def my_iwave2dec(data, Ih, Ig):
    n = len(data)
    if n < 4:
        raise ValueError("The dimension of the data is less than 4")
    else:
        half = n >> 1
        halfplus = half + 1
        temp = np.zeros(n)
        temp[0] = data[half-1] * int(Ih[0]) + data[n-1] * int(Ih[1]) + data[0] * int(Ih[2]) + data[half] * int(Ih[3])
        temp[1] = data[half - 1] * int(Ig[0]) + data[n - 1] * int(Ig[1]) + data[0] * int(Ig[2]) + data[half] * int(Ig[3])
        j = 2
        for i in range(0, half-1):
            temp[j] = data[i] * int(Ih[0]) + data[i+half] * int(Ih[1]) + data[i+1] * int(Ih[2]) + data[i+halfplus] * int(Ih[3])
            j = j + 1
            temp[j] = data[i] * int(Ig[0]) + data[i + half] * int(Ig[1]) + data[i + 1] * int(Ig[2]) + data[i + halfplus] * int(Ig[3])
            j = j + 1
        for i in range(0, n):
            data[i] = temp[i]
    return data


# 计算中间值temp0-7
def dwt_decom_mv(x, h, g):
    alp = 1
    alpha = []
    alpha.append(alp)
    for i in range(0, 85):
        alp = alp * 1
        alpha.append(alp)
    temp = []
    sum1 = 0
    sum2 = 0
    for k in range(0, 85):
        sum1 += x[2*k] * alpha[k]
        sum2 += x[2*k+1] * alpha[k]
    temp.append((alpha[1] * h[0] + h[2] * sum1))
    temp.append((alpha[1] * h[1] + h[3]) * sum2)
    temp.append((alpha[1] * g[0] + g[2]) * sum1)
    temp.append((alpha[1] * g[1] + g[3]) * sum2)
    temp.append((alpha[85] - 1) * h[2] * x[0])
    temp.append((alpha[85] - 1) * h[3] * x[1])
    temp.append((alpha[85] - 1) * g[2] * x[0])
    temp.append((alpha[85] - 1) * g[3] * x[1])
    return temp


# 计算abs变量和sign变量
def compute_abs(x):
    result = np.zeros(len(x))
    sign = np.zeros(len(x))
    for i in range(len(x)):
        if x[i] >= 0:
            result[i] = x[i]
            sign[i] = 1
        else:
            result[i] = -x[i]
            sign[i] = -1
    return result, sign


def compute_e(sign, abs_y, lamb, d64, y_prime):
    e1 = np.zeros(85)
    e2 = np.zeros(85)
    e3 = np.zeros(85)
    for i in range(85):
        e1[i] = sign[i] * (abs_y[i] - lamb)
        e2[i] = d64[i] * (y_prime[i] - e1[i])
        if e2[i] == 0:
            e3[i] = 0
        else:
            e3[i] = -e2[i]
    return e1, e2, e3


# 计算temp8-temp15
def compute_temp8to15(y, Ih, Ig):
    alp = 1
    alpha = []
    alpha.append(alp)
    for i in range(0, 85):
        alp = alp * 1
        alpha.append(alp)
    temp = []
    sum1 = 0
    sum2 = 0
    for k in range(0, 85):
        sum1 += y[k] * alpha[k]
        sum2 += y[k + 85] * alpha[k]
    temp.append((alpha[1] * Ih[0] + Ih[2]) * sum1)
    temp.append((alpha[1] * Ih[1] + Ih[3]) * sum2)
    temp.append((alpha[1] * Ig[0] + Ig[2]) * sum1)
    temp.append((alpha[1] * Ig[1] + Ig[3]) * sum2)
    temp.append((alpha[85] - 1) * Ih[2] * y[0])
    temp.append((alpha[85] - 1) * Ih[3] * y[85])
    temp.append((alpha[85] - 1) * Ig[2] * y[0])
    temp.append((alpha[85] - 1) * Ig[3] * y[85])
    return temp


if __name__ == '__main__':
    # data_floatingpoint = np.loadtxt('values/dwt/dwt_input.txt', delimiter=' ')
    # print(data)
    # 检查一下函数的正确性
    # decom_result = wavelets_decomp(data, h_fp, g_fp)
    # print(decom_result)
    data_fixpoint = np.loadtxt('values/dwt_fixpoint/dwt_input_fixpoint.txt', delimiter=' ')
    hg = np.loadtxt('values/dwt_fixpoint/dwt_hg_fixpoint.txt', delimiter=' ')
    h = hg[0:4]
    g = hg[4:8]
    # decom_result_fixpoint = wavelets_decomp(data_fixpoint, h, g)
    # np.savetxt('values/dwt_fixpoint/dwt_dec_result_fixpoint.txt', decom_result_fixpoint, fmt='%d', newline='\n')
    # temp0to7 = dwt_decom_mv(data_fixpoint, h, g)
    # np.savetxt('values/dwt_fixpoint/dwt_temp0to7_fixpoint.txt', temp0to7, fmt='%d', newline='\n')
    data_dec_result = np.loadtxt('values/dwt_fixpoint/dwt_dec_result_fixpoint.txt', delimiter=' ')
    abs_y_prime, sign_y_prime = compute_abs(data_dec_result[85:])
    y = data_dec_result[:85]

    # np.savetxt('values/dwt_fixpoint/abs_y_prime.txt', abs_y_prime, fmt='%d', newline='\n')
    # np.savetxt('values/dwt_fixpoint/sign_y_prime.txt', sign_y_prime, fmt='%d', newline='\n')

    # y prime的值，仅包含index 85-169
    data_thre_result = my_threshold(data_dec_result[85:], 3355443)
    # np.savetxt('values/dwt_fixpoint/dwt_thr_result_fixpoint.txt', data_thre_result, fmt='%d', newline='\n')

    # 导入d64的值，表示所有abs - lambda + p的二进制最高位
    d = np.loadtxt('values/dwt_fixpoint/dij_fixpoint/dij_new.txt', delimiter=' ')
    d64 = []
    for i in range(85):
        d64.append(d[i][64])
    # 计算witness e1, e2, e3, 并保存
    e1, e2, e3 = compute_e(sign_y_prime, abs_y_prime, 3355443, d64, data_thre_result)
    # np.savetxt('values/dwt_fixpoint/dwt_e1.txt', e1, fmt='%d', newline='\n')
    # np.savetxt('values/dwt_fixpoint/dwt_e2.txt', e2, fmt='%d', newline='\n')
    # np.savetxt('values/dwt_fixpoint/dwt_e3.txt', e3, fmt='%d', newline='\n')
    # 计算recons的结果x
    y_prime_total = np.concatenate((data_dec_result[:85], data_thre_result), axis=0)
    np.savetxt('values/dwt_fixpoint/test.txt', y_prime_total, fmt='%d', newline='\n')
    print(y_prime_total)
    IhIg = np.loadtxt('values/dwt_fixpoint/dwt_IhIg_fixpoint.txt', delimiter=' ')
    Ih = IhIg[0:4]
    for i in range(4):
        Ih[i] = int(Ih[i])
    Ig = IhIg[4:8]
    for i in range(4):
        Ig[i] = int(Ig[i])
    data_recons_result = my_iwave2dec(y_prime_total, Ih, Ig)
    # print(data_recons_result[0])

    # 计算temp8-temp15

    # temp8to15 = compute_temp8to15(y_prime_total, Ih, Ig)
    # print(temp8to15[0])

    # Ih0 = 3760510
    # Ih2 = 8102773
    # result = (Ig0 + Ih2) * int(summ)
    # print(result)
    # np.savetxt('values/dwt_fixpoint/dwt_temp8to15_fixpoint.txt', temp8to15, fmt='%d', newline='\n')


