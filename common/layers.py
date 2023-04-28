#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
神经网络常用的层节点实现.

@Author Kingen
"""
import abc

import numpy as np
from numpy import ndarray

from common import functions, utils


class Layer(abc.ABC):
    @abc.abstractmethod
    def forward(self, x: ndarray, train_flag=True) -> ndarray:
        """
        正向传播.
        Parameters
        ----------
        x 输入值
        train_flag 训练还是测试

        Returns
        -------
        输出值
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

    def forward(self, x: ndarray, train_flag=True) -> ndarray:
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

    def forward(self, x: ndarray, train_flag=True) -> ndarray:
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

    def forward(self, x: ndarray, train_flag=True) -> ndarray:
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


class Dropout(Layer):
    def __init__(self, dropout_ratio=0.5) -> None:
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x: ndarray, train_flag=True) -> ndarray:
        if train_flag:
            self.mask = np.random.rand(*x.shape)
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dy: ndarray) -> ndarray:
        return dy * self.mask


class ConvolutionLayer(Layer):
    def __init__(self, w: ndarray, b: ndarray, stride=1, pad=0) -> None:
        self.w = w
        self.b = b
        self.stride = stride
        self.pad = pad

        self.x = None
        self.col = None
        self.col_w = None

        self.dw = None
        self.db = None

    def forward(self, x: ndarray, train_flag=True) -> ndarray:
        fn, c, fh, fw = self.w.shape
        n, c, h, w = x.shape
        output_h = 1 + int((h + 2 * self.pad - fh) / self.stride)
        output_w = 1 + int((w + 2 * self.pad - fw) / self.stride)

        self.col = utils.im2col(x, fh, fw, self.stride, self.pad)
        self.col_w = self.w.reshape(fn, -1).T

        output = np.dot(self.col, self.col_w) + self.b
        output = output.reshape(n, output_h, output_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        return output

    def backward(self, dy: ndarray) -> ndarray:
        fn, c, fh, fw = self.w.shape
        dy = dy.transpose((0, 2, 3, 1)).reshape(-1, fn)

        self.db = np.sum(dy, axis=0)
        self.dw = np.dot(self.col.T, dy)
        self.dw = self.dw.transpose((1, 0)).reshape(fn, c, fh, fw)

        dcol = np.dot(dy, self.col_w.T)
        return utils.col2im(dcol, self.x.shape, fh, fw, self.stride, self.pad)


class Pooling(Layer):
    def __init__(self, pool_h, pool_w, stride=2, pad=0) -> None:
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

        self.x = None
        self.arg_max = None

    def forward(self, x: ndarray, train_flag=True) -> ndarray:
        n, c, h, w = x.shape
        output_h = int(1 + (h - self.pool_h) / self.stride)
        output_w = int(1 + (w - self.pool_w) / self.stride)

        col = utils.im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h * self.pool_w)

        self.x = x
        self.arg_max = np.argmax(col, axis=1)
        output = np.max(col, axis=1)
        return output.reshape(n, output_h, output_w, c).transpose(0, 3, 1, 2)

    def backward(self, dy: ndarray) -> ndarray:
        dy = dy.transpose((0, 2, 3, 1))

        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dy.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flattern()] = dy.flatten()
        dmax = dmax.reshape(dy.shape + (pool_size,))

        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        return utils.col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
