#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
两层神经网络的实现.

@Author Kingen
"""
import abc

import numpy as np
from numpy import ndarray

from common import functions
from common.layers import AffineLayer, ReluLayer, SoftmaxWithLossLayer


class TwoLayerNet(abc.ABC):
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

    @abc.abstractmethod
    def predict(self, x: ndarray) -> ndarray:
        """
        识别输入.
        :param x: 输入值
        :return: 最终预测值
        """
        pass

    @abc.abstractmethod
    def loss(self, x: ndarray, t: ndarray) -> float:
        """
        计算预测误差.
        :param x: 输入值
        :param t: 正解
        :return: 预测误差
        """
        pass

    @abc.abstractmethod
    def gradient(self, x: ndarray, t: ndarray) -> tuple:
        """
        计算误差对各参数的梯度.
        :param x: 输入值
        :param t: 正解
        :return: 梯度
        """
        pass

    def calc_accuracy(self, x, t) -> float:
        """
        计算当前参数预测的准确度.
        :param x: 输入值
        :param t: 正解
        :return: 准确度
        """
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        return np.sum(y == t) / float(x.shape[0])

    def update_args(self, grad, learning_rate):
        """
        更新当前参数.
        :param grad: 更新的梯度
        :param learning_rate: 学习率
        :return:
        """
        self.w1 -= learning_rate * grad[0]
        self.b1 -= learning_rate * grad[1]
        self.w2 -= learning_rate * grad[2]
        self.b2 -= learning_rate * grad[3]


class TwoLayerNetNumerical(TwoLayerNet):
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        super().__init__(input_size, hidden_size, output_size, weight_init_std)

    def predict(self, x: ndarray) -> ndarray:
        z1 = np.dot(x, self.w1) + self.b1
        a1 = functions.sigmoid(z1)
        z2 = np.dot(a1, self.w2) + self.b2
        y = functions.softmax(z2)
        return y

    def loss(self, x: ndarray, t: ndarray) -> float:
        y = self.predict(x)
        return functions.cross_entropy_error(y, t)

    def gradient(self, x: ndarray, t: ndarray) -> tuple:
        """
        利用数值微分计算梯度.
        """
        loss_func = lambda r: self.loss(x, t)
        dw1 = functions.numerical_gradient(loss_func, self.w1)
        db1 = functions.numerical_gradient(loss_func, self.b1)
        dw2 = functions.numerical_gradient(loss_func, self.w2)
        db2 = functions.numerical_gradient(loss_func, self.b2)
        return dw1, db1, dw2, db2


class TwoLayerNetBackward(TwoLayerNet):
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        super().__init__(input_size, hidden_size, output_size, weight_init_std)
        self.layers = [
            AffineLayer(self.w1, self.b1),
            ReluLayer(),
            AffineLayer(self.w2, self.b2)
        ]
        self.last_layer = SoftmaxWithLossLayer()

    def predict(self, x: ndarray) -> ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def loss(self, x: ndarray, t: ndarray) -> float:
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def gradient(self, x: ndarray, t: ndarray) -> tuple:
        """
        利用反向传播计算梯度.
        """
        self.loss(x, t)
        dy = self.last_layer.backward()
        for layer in reversed(self.layers):
            dy = layer.backward(dy)
        return self.layers[0].dw, self.layers[0].db, self.layers[2].dw, self.layers[2].db
