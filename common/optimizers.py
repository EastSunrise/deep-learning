#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
参数优化.

@author: Kingen
"""
import abc
import math
from typing import List

import numpy as np
from numpy import ndarray


class Optimizer(abc.ABC):

    @abc.abstractmethod
    def update(self, params: List[ndarray], grads: List[ndarray]):
        """
        参数更新
        Parameters
        ----------
        params 原参数
        grads 梯度
        """
        pass


class SGD(Optimizer):
    """
    随机梯度下降法.
    """

    def __init__(self, learning_rate=0.01) -> None:
        self.learning_rate = learning_rate

    def update(self, params: List[ndarray], grads: List[ndarray]):
        for i in range(len(params)):
            params[i] -= self.learning_rate * grads[i]


class Momentum(Optimizer):
    """
    动量法.
    """

    def __init__(self, learning_rate=0.01, momentum=0.9) -> None:
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.v = None

    def update(self, params: List[ndarray], grads: List[ndarray]):
        if self.v is None:
            self.v = [np.zeros_like(x) for x in params]

        for i in range(len(params)):
            self.v[i] = self.momentum * self.v[i] - self.learning_rate * grads[i]
            params[i] += self.v[i]


class AdaGrad(Optimizer):

    def __init__(self, learning_rate=0.01) -> None:
        self.learning_rate = learning_rate
        self.h = None

    def update(self, params: List[ndarray], grads: List[ndarray]):
        if self.h is None:
            self.h = [np.zeros_like(x) for x in params]

        for i in range(len(params)):
            self.h[i] += grads[i] * grads[i]
            params[i] -= self.learning_rate * grads[i] / (np.sqrt(self.h[i]) + 1e-7)


class RMSProp(Optimizer):
    def __init__(self, learning_rate=0.01, decay_rate=0.99) -> None:
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.h = None

    def update(self, params: List[ndarray], grads: List[ndarray]):
        if self.h is None:
            self.h = [np.zeros_like(x) for x in params]

        for i in range(len(params)):
            self.h[i] = self.decay_rate * self.h[i] + (1 - self.decay_rate) * grads[i] * grads[i]
            params[i] -= self.learning_rate * grads[i] / (np.sqrt(self.h[i]) + 1e-7)


class Adam(Optimizer):

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999) -> None:
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.t = 0
        self.m = None
        self.v = None

    def update(self, params: List[ndarray], grads: List[ndarray]):
        if self.t == 0:
            self.m = [np.zeros_like(x) for x in params]
            self.v = [np.zeros_like(x) for x in params]

        self.t += 1
        alpha = self.learning_rate * math.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)
        for i in range(len(params)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grads[i] * grads[i]
            params[i] -= alpha * self.m[i] / (np.sqrt(self.v[i]) + 1e-7)
