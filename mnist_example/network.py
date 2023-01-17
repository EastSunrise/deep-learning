#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
两层神经网络的实现.

@Author Kingen
"""
import abc
from typing import List

import numpy as np
from numpy import ndarray

from common import functions
from common.layers import AffineLayer, ReluLayer, SoftmaxWithLossLayer


class Arg:
    def __init__(self, w, b):
        self.w = w
        self.b = b


class MultiLayerNet(abc.ABC):
    def __init__(self, input_size, output_size, hidden_size, learning_rate=0.01, weight_init_std=0.01):
        """
        初始化权重和偏置.
        :param input_size: 输入层大小
        :param output_size: 输出层大小
        :param hidden_size: 隐藏层大小，可以是多个
        :param learning_rate: 学习率
        :param weight_init_std: 权重初始值标准
        """
        sizes = [input_size]
        if isinstance(hidden_size, int):
            sizes.append(hidden_size)
        else:
            sizes.extend(hidden_size)
        sizes.append(output_size)
        self.params = []
        for i in range(1, len(sizes)):
            w = weight_init_std * np.random.randn(sizes[i - 1], sizes[i])
            b = np.zeros(sizes[i])
            self.params.append(Arg(w, b))
        self.learning_rate = learning_rate

    def train(self, x: ndarray, t: ndarray):
        """
        对给定的输入和正解进行一次训练.
        :param x: 输入值
        :param t: 正解
        """
        grads = self.gradient(x, t)
        for i, grad in enumerate(grads):
            self.params[i].w -= self.learning_rate * grad.w
            self.params[i].b -= self.learning_rate * grad.b

    def calc_accuracy(self, x: ndarray, t: ndarray) -> float:
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

    @abc.abstractmethod
    def gradient(self, x: ndarray, t: ndarray) -> List[Arg]:
        """
        计算误差对各参数的梯度.
        :param x: 输入值
        :param t: 正解
        :return: 梯度
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
    def predict(self, x: ndarray) -> ndarray:
        """
        识别输入.
        :param x: 输入值
        :return: 最终预测值
        """
        pass


class MultiLayerNetNumerical(MultiLayerNet):
    def __init__(self, input_size, output_size, hidden_size, learning_rate=0.01, weight_init_std=0.01):
        super().__init__(input_size, output_size, hidden_size, learning_rate, weight_init_std)

    def predict(self, x: ndarray) -> ndarray:
        a = x
        for arg in self.params[:-1]:
            z = np.dot(a, arg.w) + arg.b
            a = functions.sigmoid(z)
        z = np.dot(a, self.params[-1].w) + self.params[-1].b
        return functions.softmax(z)

    def loss(self, x: ndarray, t: ndarray) -> float:
        y = self.predict(x)
        return functions.cross_entropy_error(y, t)

    def gradient(self, x: ndarray, t: ndarray) -> List[Arg]:
        """
        利用数值微分计算梯度.
        """
        loss_func = lambda r: functions.cross_entropy_error(self.predict(x), t)
        grads = []
        for arg in self.params:
            dw = functions.numerical_gradient(loss_func, arg.w)
            db = functions.numerical_gradient(loss_func, arg.b)
            grads.append(Arg(dw, db))
        return grads


class MultiLayerNetBackward(MultiLayerNet):
    def __init__(self, input_size, output_size, hidden_size, learning_rate=0.01, weight_init_std=0.01):
        super().__init__(input_size, output_size, hidden_size, learning_rate, weight_init_std)
        self.layers = []
        for arg in self.params[:-1]:
            self.layers.append(AffineLayer(arg.w, arg.b))
            self.layers.append(ReluLayer())
        self.layers.append(AffineLayer(self.params[-1].w, self.params[-1].b))
        self.last_layer = SoftmaxWithLossLayer()

    def predict(self, x: ndarray) -> ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def loss(self, x: ndarray, t: ndarray) -> float:
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def gradient(self, x: ndarray, t: ndarray) -> List[Arg]:
        """
        利用反向传播计算梯度.
        """
        self.loss(x, t)
        dy = self.last_layer.backward()
        for layer in reversed(self.layers):
            dy = layer.backward(dy)
        return [Arg(layer.dw, layer.db) for i, layer in enumerate(self.layers) if i % 2 == 0]
