#!/home/.pyenv/versions/3.6.6/python3.6
# -*- coding: utf-8 -*-
# Edited by bighead 18-11-25

import numpy as np

def load_training_data():
    """
    get the trainig datas and translate they to np.array
    param: None
    return: x, y(both np.array)
    """
    train_data = "data/Algorithm_book/data.txt"
    with open(train_data, 'r', encoding='utf-8')as f:
        x = list()
        y = list()
        for data in f.readlines():
            x_tmp, y_tmp = data.split('\t')[:-1], data.split('\t')[-1]
            x_tmp_tmp = list()
            for tmp in x_tmp:
                x_tmp_tmp.append(np.float(tmp))
            x.append(x_tmp_tmp)
            y.append(np.float(y_tmp))
    x = np.array(x)
    y = np.array(y)
    return x.T, y

def load_test_data():
    """
    get the test datas and translate them to np.array
    param: None
    return: x(np.array)
    """
    test_data = "data/Algorithm_book/test_data"
    with open(test_data, 'r', encoding='utf-8')as f:
        x = list()
        for data in f.readlines():
            x_tmp = data.split('\t')
            x_tmp_tmp = list()
            for tmp in x_tmp:
                x_tmp_tmp.append(np.float(tmp))
            x.append(x_tmp_tmp)
        x = np.array(x)
    return x.T

if __name__ == "__main__":
    x = load_test_data()
    print(type(x))
