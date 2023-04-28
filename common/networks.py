#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
多层神经网络的实现.

@Author Kingen
"""
from typing import List

import numpy as np
from numpy import ndarray

import common.gradient
from common.layers import AffineLayer, ReluLayer, SoftmaxWithLossLayer, SigmoidLayer
from common.optimizers import SGD

activations = {'sigmoid': SigmoidLayer, 'relu': ReluLayer}
# 权重初始化标准差系数
weight_init_std = {'sigmoid': 1, 'relu': 2}


class MultiLayerNet:
    def __init__(self, input_size, output_size, hidden_size, activation='relu', optimizer=SGD(0.01)):
        """
        初始化多层神经网络.
        Parameters
        ----------
        input_size 输入层大小
        output_size 输出层大小
        hidden_size 隐藏层大小，可以是多个
        activation 激活函数，'relu'或'sigmoid'
        optimizer 参数优化算法
        """
        sizes = [input_size]
        if isinstance(hidden_size, int):
            sizes.append(hidden_size)
        else:
            sizes.extend(hidden_size)
        sizes.append(output_size)
        self.params = []
        for i in range(1, len(sizes)):
            init_weight = np.sqrt(weight_init_std[activation] / sizes[i - 1])
            self.params.append(init_weight * np.random.randn(sizes[i - 1], sizes[i]))
            self.params.append(np.zeros(sizes[i]))

        self.layers = []
        for i in range(0, len(self.params) - 2, 2):
            self.layers.append(AffineLayer(self.params[i], self.params[i + 1]))
            self.layers.append(activations[activation]())
        self.layers.append(AffineLayer(self.params[-2], self.params[-1]))
        self.last_layer = SoftmaxWithLossLayer()

        self.optimizer = optimizer

    def train(self, x: ndarray, t: ndarray):
        """
        对给定的输入和正解进行一次训练.
        :param x: 输入值
        :param t: 正解
        """
        grads = self.gradient(x, t)
        self.optimizer.update(self.params, grads)

    def gradient(self, x: ndarray, t: ndarray) -> List[ndarray]:
        """
        利用反向传播计算梯度.
        :param x: 输入值
        :param t: 正解
        :return: 梯度
        """
        self.loss(x, t)
        dy = self.last_layer.backward()
        for layer in reversed(self.layers):
            dy = layer.backward(dy)
        return [x for i, layer in enumerate(self.layers) if i % 2 == 0 for x in [layer.dw, layer.db]]

    def numerical_gradient(self, x: ndarray, t: ndarray) -> List[ndarray]:
        """
        利用数值微分计算梯度.
        """
        return [common.gradient.numerical_gradient(lambda r: self.loss(x, t), p) for p in self.params]

    def loss(self, x: ndarray, t: ndarray) -> float:
        """
        计算预测误差.
        :param x: 输入值
        :param t: 正解
        :return: 预测误差
        """
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def predict(self, x: ndarray) -> ndarray:
        """
        识别输入.
        :param x: 输入值
        :return: 最终预测值
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x

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
