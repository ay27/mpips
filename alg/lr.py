#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import sys, time

import sys

# sys.path.append('..')
# sys.path.append('../mpips/')

from mpips import MpiPs
import numpy as np
from scipy.stats import logistic


def line_trans(line):
    lst = list(map(float, line.strip().split(',')))
    lst.insert(0, 1.0)
    x, y = np.array(lst[:-1]), lst[-1]
    return x, y


class LR(MpiPs):
    def __init__(self, ps_num,
                 input_file,
                 learning_rate,
                 iter_num):
        self.input_file = input_file
        self.learning_rate = learning_rate
        self.iter_num = iter_num
        super(LR, self).__init__(ps_num)

    def load_data(self):
        self.train_set = list(map(line_trans, self.load_input(self.input_file)))
        self.Ndim = len(self.train_set[0][0])

    def train(self):
        self.theta = np.random.rand(self.Ndim)
        if self.wk_rank == 0:
            self.push_vector("theta", self.theta)
        self.sync()
        for i in range(self.iter_num):
            # print('iter: %d' % i)
            self.local_theta = np.zeros(self.Ndim)
            for x, y in self.train_set:
                coef = self.learning_rate * (y - logistic.cdf(np.inner(x, self.theta)))
                self.local_theta += (coef * x)
            self.push_vector("theta", self.local_theta)
            self.sync()
            self.theta = self.pull_vector("theta")
        if self.wk_rank == 0:
            print(self.theta)


if __name__ == "__main__":
    lr = LR(2, "data/lr", 0.05, 20)
    lr.load_data()
    lr.train()
    lr.end()
