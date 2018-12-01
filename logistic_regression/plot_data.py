# !/bin/.pyenv/versions/2.7.15/bin/python2.7
# -*- coding: utf-8 -*-
# Edited by bighead 18-12-1
# using python 2.7.15 pyenv version

import matplotlib.pyplot as plt
from util import load_training_data
from sklearn.linear_model import LogisticRegression

x, y = load_training_data()
x = x.T
#plt.show()

clf = LogisticRegression()
clf.fit(x, y)
y_p = clf.predict(x)
plt.scatter(x[:, 0], x[:, 1], c=y_p, s=40, cmap=plt.cm.Spectral)
plt.show()
