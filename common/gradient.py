#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
梯度计算.

@author: Kingen
"""
import numpy as np
from numpy import ndarray


def numerical_gradient(func, x: ndarray, delta=1e-4):
    """
    利用数值微分计算梯度.
    :param func: 多元函数
    :param x: 多元自变量
    :return: 在x处的梯度
    """
    grad = np.zeros_like(x)
    itr = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not itr.finished:
        idx = itr.multi_index
        tmp = x[idx]
        x[idx] = float(tmp) - delta
        y1 = func(x)
        x[idx] = float(tmp) + delta
        y2 = func(x)
        grad[idx] = (y2 - y1) / (2 * delta)
        x[idx] = tmp
        itr.iternext()
    return grad


def gradient_descent(func, x0, eta=0.01, step_num=100):
    """
    梯度下降法.
    :param func: 函数
    :param x0: 初始值
    :param eta: 学习率
    :param step_num: 重复次数
    :return: x更新结果
    """
    x = x0
    for i in range(step_num):
        grad = numerical_gradient(func, x)
        x -= eta * grad
    return x
