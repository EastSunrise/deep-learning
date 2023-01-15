#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
A two-layer network.

@Author Kingen
"""
import numpy as np

import functions


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        """
        初始化权重和偏置.
        :param input_size: 输入层大小
        :param hidden_size: 隐藏层大小
        :param output_size: 输出层大小
        :param weight_init_std: 权重初始值标准
        """
        self.w1 = weight_init_std * np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros(hidden_size)
        self.w2 = weight_init_std * np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros(output_size)

    def predict(self, x):
        """
        识别输入.
        :param x: 输入值
        :return: 预测值
        """
        z1 = np.dot(x, self.w1) + self.b1
        a1 = functions.sigmoid(z1)
        z2 = np.dot(a1, self.w2) + self.b2
        y = functions.softmax(z2)
        return y

    def loss(self, x, t):
        """
        计算损失函数的值.
        :param x: 输入值
        :param t: 正解
        :return: 预测误差
        """
        y = self.predict(x)
        return functions.cross_entropy_error(y, t)

    def numerical_gradient(self, x, t):
        """
        根据数值微分计算梯度.
        :param x: 输入值
        :param t: 正解
        :return: 梯度
        """
        loss_func = lambda r: self.loss(x, t)
        gw1 = functions.numerical_gradient(loss_func, self.w1)
        gb1 = functions.numerical_gradient(loss_func, self.b1)
        gw2 = functions.numerical_gradient(loss_func, self.w2)
        gb2 = functions.numerical_gradient(loss_func, self.b2)
        return gw1, gb1, gw2, gb2

    def calc_accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        return np.sum(y == t) / float(x.shape[0])

    def update_args(self, grad, learning_rate):
        self.w1 -= learning_rate * grad[0]
        self.b1 -= learning_rate * grad[1]
        self.w2 -= learning_rate * grad[2]
        self.b2 -= learning_rate * grad[3]
