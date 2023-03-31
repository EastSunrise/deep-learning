#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
神经网络常用函数.

@Author Kingen
"""

import numpy as np
from numpy import ndarray


# 激活函数 #


def step_function(x: ndarray):
    """
    单位阶跃函数.
    """
    return (x > 0).astype(int)


def sigmoid(x: ndarray):
    """
    Sigmoid 函数.
    """
    return 1 / (1 + np.exp(-x))


def relu(x: ndarray):
    """
    ReLU 函数.
    :param x: 每层的输入
    """
    return np.maximum(x, 0)


def softmax(x: ndarray):
    """
    softmax 函数.
    :param x: 输出层的输入
    """
    if x.ndim == 2:
        x = x.T
        exp_x = np.exp(x - np.max(x, axis=0))
        y = exp_x / np.sum(exp_x, axis=0)
        return y.T
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)


# 损失函数 #

def mean_squared_error(y: ndarray, t: ndarray):
    """
    均方误差.
    :param y: 预测值
    :param t: 正解
    """
    return 0.5 * np.sum((y - t) ** 2)


def cross_entropy_error(y: ndarray, t: ndarray):
    """
    交叉熵误差，计算时加上一个极小值，防止log(0)负的无限大.
    :param y: 单个或者多个训练实例的预测值
    :param t: 正解
    """
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)
    if y.size == t.size:
        t = t.argmax(axis=1)

    size = y.shape[0]
    return -np.sum(np.log(y[np.arange(size), t] + 1e-7)) / size
