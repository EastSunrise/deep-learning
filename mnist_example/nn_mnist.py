#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
使用MNIST数据集学习.

@Author Kingen
"""
import os
import pickle
from timeit import default_timer

import numpy as np
from matplotlib import pyplot as plt

from dataset import mnist
from mnist_example.two_layer_net import TwoLayerNet

x_train, t_train, x_test, t_test = mnist.load_mnist(normalize=True, one_hot_label=True)

learning_rate = 0.1
train_size = x_train.shape[0]  # 训练集大小
input_size = x_train[0].shape[0]
network_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'network.pkl')


def train(train_num, batch_size=100):
    """
    训练数据.
    :param train_num: 训练次数
    :param batch_size: 批次大小
    :return: 学习过程中误差变化
    """
    losses = []
    train_accuracies = []
    test_accuracies = []

    # 初始化神经网络
    if not os.path.exists(network_file):
        network = TwoLayerNet(input_size, hidden_size=50, output_size=10)
    else:
        network = pickle.load(open(network_file, 'rb'))

    for i in range(train_num):
        print(f'training {i + 1}...')
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        # 计算梯度并更新参数
        grad = network.numerical_gradient(x_batch, t_batch)
        network.update_args(grad, learning_rate)
        losses.append(network.loss(x_batch, t_batch))

        train_acc = network.calc_accuracy(x_train, t_train)
        test_acc = network.calc_accuracy(x_test, t_test)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        print(f'accuracy: train={train_acc}, test={test_acc}')

        if i % 100 == 99:
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


if __name__ == '__main__':
    start = default_timer()
    train(50)
    print("cost: ", default_timer() - start)
