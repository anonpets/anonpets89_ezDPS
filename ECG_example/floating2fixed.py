import xlrd
import xlwt
import pandas as pd
import numpy as np

# Specify total size of your fixed-point and size of fractions part
FP_Size = 32
FR_Size = 18

# Input to be converted
# inp = 1
"""
对二维数据进行fixpoint处理
"""
# from numpy import loadtxt
# input_data = loadtxt('values/svc/support_vectors.txt', delimiter=' ')
# print(len(input_data))
# print(len(input_data[0]))
#
# output = np.zeros(shape=(len(input_data), len(input_data[0])))
#
# for row in range(0, len(input_data)):
#     for col in range(0, len(input_data[0])):
#         result = np.zeros(shape=(1, 32))
#         flag = 0
#         inp = input_data[row][col]
#         if inp < 0:
#             flag = 1
#             inp = inp * -1
#         for i in range(FP_Size - FR_Size - 1, -FR_Size - 1, -1):
#             if inp >= 2 ** i:
#                 result[0, FR_Size + i] = 1
#                 inp -= 2 ** i
#         temp = 0
#         for i in range(0, 32, 1):
#             temp += (2 ** i) * result[0, i]
#         if flag == 1:
#             output[row][col] = -temp
#         else:
#             output[row][col] = temp
#     np.savetxt('values/svc_fixpoint/support_vectors_fixpoint.txt', output, fmt='%d', newline='\n')


"""
对一维数据进行fixpoint处理
"""
# from numpy import loadtxt
# input_data = loadtxt('values/svc/dual_coef.txt', delimiter=' ')
# print(len(input_data))
#
# output = []
#
# for inp in input_data:
#     result = np.zeros(shape=(1, 32))
#     flag = 0
#     if inp < 0:
#         flag = 1
#         inp = inp * -1
#     for i in range(FP_Size - FR_Size - 1, -FR_Size - 1, -1):
#         if inp >= 2 ** i:
#             result[0, FR_Size + i] = 1
#             inp -= 2 ** i
#     temp = 0
#     for i in range(0, 32, 1):
#         temp += (2 ** i) * result[0, i]
#     if flag == 1:
#         output.append(-temp)
#     else:
#         output.append(temp)
# np.savetxt('values/svc_fixpoint/dual_coef_fixpoint.txt', output, fmt='%d', newline='\n')

"""
一个数的小test
"""
inp = -0.5
print(inp)
result = np.zeros(shape=(1, 32))
flag = 0
if inp < 0:
    flag = 1
    inp = inp * -1
for i in range(FP_Size - FR_Size - 1, -FR_Size - 1, -1):
    if inp >= 2 ** i:
        result[0, FR_Size + i] = 1
        inp -= 2 ** i

temp = 0
for i in range(0, 32, 1):
    temp += (2 ** i) * result[0, i]
print("result in uint32:")
if flag == 1:
    print(-int(temp))
else:
    print(int(temp))

print("absolute result in binary:")
print(np.flip(result))
