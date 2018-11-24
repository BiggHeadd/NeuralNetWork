#!/home/.pyenv/versions/3.6.6/python3
# -*- coding: utf-8 -*-
# Edited by bighead 18-11-24

class logistic_regression:

    def __init__(self, layers):
        self.layers = layers

    def print_layers(self):
        print(self.layers)

if __name__ == "__main__":
    log_reg = logistic_regression(1)
    log_reg.print_layers()
