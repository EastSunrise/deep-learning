#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
优化算法比较.

@author: Kingen
"""

import matplotlib.pyplot as plt

from common.networks import MultiLayerNet
from common.optimizers import *
from dataset import mnist

x_train, t_train, x_test, t_test = mnist.load_mnist(normalize=True, one_hot_label=True)

train_size = x_train.shape[0]  # 训练集大小
input_size = x_train[0].shape[0]
batch_size = 128
max_iterations = 2000

optimizers = {'SGD': SGD(), 'Momentum': Momentum(), 'AdaGrad': AdaGrad(), 'RMSProp': RMSProp(), 'Adam': Adam()}

networks = {}
train_loss = {}
hidden_size = [100, 100, 100, 100]
for key, optimizer in optimizers.items():
    networks[key] = MultiLayerNet(input_size=784, output_size=10, hidden_size=hidden_size, optimizer=optimizer)
    train_loss[key] = []


def smooth_curve(x):
    window_len = 11
    s = np.r_[x[window_len - 1:0:-1], x, x[-1:-window_len:-1]]
    w = np.kaiser(window_len, 2)
    y = np.convolve(w / w.sum(), s, mode='valid')
    return y[5:len(y) - 5]


if __name__ == '__main__':
    for i in range(max_iterations):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        for key in optimizers.keys():
            grads = networks[key].gradient(x_batch, t_batch)
            optimizers[key].update(networks[key].params, grads)
            loss = networks[key].loss(x_batch, t_batch)
            train_loss[key].append(loss)

        if i % 100 == 0:
            print("===========" + "iteration:" + str(i) + "===========")
            for key in optimizers.keys():
                loss = networks[key].loss(x_batch, t_batch)
                print(key + ":" + str(loss))

    x = np.arange(max_iterations)
    for key in optimizers.keys():
        plt.plot(x, smooth_curve(train_loss[key]), label=key)
    plt.xlabel("iterations")
    plt.ylabel("loss")
    plt.ylim(0, 1)
    plt.legend()
    plt.show()
