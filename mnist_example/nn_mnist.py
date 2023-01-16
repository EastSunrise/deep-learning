#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
使用MNIST数据集训练.

@Author Kingen
"""
import os
import pickle
from timeit import default_timer

import numpy as np
from matplotlib import pyplot as plt

from dataset import mnist
from mnist_example.two_layer_net import TwoLayerNetNumerical, TwoLayerNet, TwoLayerNetBackward

x_train, t_train, x_test, t_test = mnist.load_mnist(normalize=True, one_hot_label=True)

learning_rate = 0.1
train_size = x_train.shape[0]  # 训练集大小
input_size = x_train[0].shape[0]


def train(network_file, train_num: int, network: TwoLayerNet = None, batch_size=100):
    """
    训练数据.
    :param network_file: 神经网络缓存pkl文件
    :param train_num: 训练次数
    :param network: 神经网络
    :param batch_size: 批次大小
    :return: 训练过程中误差变化
    """
    losses = []
    train_accuracies = []
    test_accuracies = []
    if os.path.exists(network_file):
        network = pickle.load(open(network_file, 'rb'))
    if network is None:
        raise ValueError("network must not be none")

    for i in range(train_num):
        print(f'training {i + 1}...')
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        # 计算梯度并更新参数
        grad = network.gradient(x_batch, t_batch)
        network.update_args(grad, learning_rate)
        losses.append(network.loss(x_batch, t_batch))

        if i % 10 == 0:
            train_acc = network.calc_accuracy(x_train, t_train)
            test_acc = network.calc_accuracy(x_test, t_test)
            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)
            print(f'accuracy: train={train_acc}, test={test_acc}')

        if i % 10 == 9:
            # 存储训练结果
            print("Updating network file ...")
            pickle.dump(network, open(network_file, 'wb'), -1)
            print("Done!")

    # 绘制图形
    x = np.arange(len(train_accuracies))
    plt.plot(x, train_accuracies, label='train accuracy')
    plt.plot(x, test_accuracies, label='test accuracy', linestyle='--')
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.ylim(0, 1.0)
    plt.legend(loc='lower right')
    plt.show()


def train_numerical(train_num: int):
    """
    利用数值微分训练.
    :return:
    """
    network_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'numerical.pkl')
    if os.path.exists(network_file):
        train(network_file, train_num)
    else:
        train(network_file, train_num, TwoLayerNetNumerical(input_size, hidden_size=50, output_size=10))


def train_backward(train_num: int):
    """
    利用反向传播训练.
    :return:
    """
    network_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'backward.pkl')
    if os.path.exists(network_file):
        train(network_file, train_num)
    else:
        train(network_file, train_num, TwoLayerNetBackward(input_size, hidden_size=50, output_size=10))


if __name__ == '__main__':
    start = default_timer()
    train_backward(11)
    print("cost: ", default_timer() - start)
