#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
神经网络常用的层节点实现.

@Author Kingen
"""
import abc

import numpy as np
from numpy import ndarray

from common import functions


class Layer(abc.ABC):
    @abc.abstractmethod
    def forward(self, x: ndarray) -> ndarray:
        """
        正向传播.
        :param x: 输入值
        :return: 输出值
        """
        pass

    @abc.abstractmethod
    def backward(self, dy: ndarray) -> ndarray:
        """
        反向传播.
        :param dy: 输出值的误差
        :return: 输入值的误差
        """
        pass


class ReluLayer(Layer):
    def __init__(self):
        self.mask = None

    def forward(self, x: ndarray) -> ndarray:
        self.mask = x <= 0
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dy: ndarray) -> ndarray:
        dy[self.mask] = 0
        return dy


class SigmoidLayer(Layer):
    def __init__(self):
        self.out = None

    def forward(self, x: ndarray) -> ndarray:
        self.out = functions.sigmoid(x)
        return self.out

    def backward(self, dy: ndarray) -> ndarray:
        return dy * (1.0 - self.out) * self.out


class AffineLayer(Layer):
    """
    仿射变换.
    """

    def __init__(self, w: ndarray, b: ndarray):
        self.w = w
        self.b = b
        self.x = None
        self.original_x_shape = None
        self.dw = None
        self.db = None

    def forward(self, x: ndarray) -> ndarray:
        self.original_x_shape = x.shape
        self.x = x
        return np.dot(x, self.w) + self.b

    def backward(self, dy: ndarray) -> ndarray:
        dx = np.dot(dy, self.w.T)
        self.dw = np.dot(self.x.T, dy)
        self.db = np.sum(dy, axis=0)
        return dx.reshape(*self.original_x_shape)


class SoftmaxWithLossLayer:
    def __init__(self):
        self.y = None
        self.t = None

    def forward(self, x: ndarray, t: ndarray) -> float:
        self.t = t
        self.y = functions.softmax(x)
        return functions.cross_entropy_error(self.y, t)

    def backward(self) -> ndarray:
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:
            return (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            return dx / batch_size
