import numpy as np


def count_class(y):
    count = 1
    label = []
    label.append(y[0])
    for i in range(len(y)):
        tag = 0
        for j in label:
            if y[i] == j:
                tag = 1
        if tag == 0:
            count += 1
            label.append(y[i])
    return count


