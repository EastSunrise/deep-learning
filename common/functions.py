#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
神经网络常用函数.

@Author Kingen
"""
import numpy as np
from numpy import ndarray


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


def numerical_diff(func, x):
    """
    利用数值微分计算导数.
    :param func: 函数
    :param x: 自变量
    :return: 在x处近似导数
    """
    delta = 1e-4
    return (func(x + delta) - func(x - delta)) / (2 * delta)


def numerical_gradient(func, x: ndarray):
    """
    利用数值微分计算梯度.
    :param func: 多元函数
    :param x: 多元自变量
    :return: 在x处的梯度
    """
    delta = 1e-4
    grad = np.zeros_like(x)
    itr = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not itr.finished:
        idx = itr.multi_index
        tmp = x[idx]
        x[idx] = float(tmp) - delta
        y1 = func(x)
        x[idx] = tmp + delta
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
