#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
工具类.

@author: Kingen
"""
import numpy as np


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    n, c, h, w = input_data.shape
    out_h = (h + 2 * pad - filter_h) // stride + 1
    out_w = (w + 2 * pad - filter_w) // stride + 1

    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((n, c, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose((0, 4, 5, 1, 2, 3)).reshape(n * out_h * out_w, -1)
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    n, c, h, w = input_shape
    out_h = (h + 2 * pad - filter_h) // stride + 1
    out_w = (w + 2 * pad - filter_w) // stride + 1
    col = col.reshape(n, out_h, out_w, c, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((n, c, h + 2 * pad + stride - 1, w + 2 * pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:h + pad, pad:w + pad]
