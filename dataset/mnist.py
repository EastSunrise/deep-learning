# coding: utf-8
# mnist dataset

import os
import os.path
import pickle

import numpy as np
from PIL import Image

dataset_dir = os.path.dirname(os.path.abspath(__file__))
key_filenames = {
    'train_img': 'train-images-idx3-ubyte',
    'train_label': 'train-labels-idx1-ubyte',
    'test_img': 't10k-images-idx3-ubyte',
    'test_label': 't10k-labels-idx1-ubyte'
}
save_file = dataset_dir + "/mnist.pkl"

train_num = 60000
test_num = 10000
img_dim = (1, 28, 28)
img_size = 784


def download_mnist():
    """
    Download from kaggle.com by 'kaggle datasets download -d zalando-research/fashionmnist'.
    Then unzip the file and move files to the dataset directory.
    """
    pass


def _load_img(filename):
    print("Converting " + filename + " to NumPy Array ...")
    with open(os.path.join(dataset_dir, filename), 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, img_size)
    print("Done")
    return data


def _load_label(filename):
    print("Converting " + filename + " to NumPy Array ...")
    with open(os.path.join(dataset_dir, filename), 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)
    print("Done")
    return labels


def init_mnist():
    download_mnist()
    dataset = {
        'train_img': _load_img(key_filenames['train_img']),
        'train_label': _load_label(key_filenames['train_label']),
        'test_img': _load_img(key_filenames['test_img']),
        'test_label': _load_label(key_filenames['test_label'])
    }
    print("Creating pickle file ...")
    with open(save_file, 'wb') as f:
        pickle.dump(dataset, f, -1)
    print("Done!")


def _change_one_hot_label(x):
    t = np.zeros((x.size, 10))
    for idx, row in enumerate(t):
        row[x[idx]] = 1
    return t


def load_mnist(normalize=True, flatten=True, one_hot_label=False):
    """读入MNIST数据集
    
    Parameters
    ----------
    normalize : 将图像的像素值正规化为0.0~1.0
    one_hot_label : 
        one_hot_label为True的情况下，标签作为one-hot数组返回
        one-hot数组是指[0,0,1,0,0,0,0,0,0,0]这样的数组
    flatten : 是否将图像展开为一维数组
    
    Returns
    -------
    (训练图像, 训练标签), (测试图像, 测试标签)
    """
    if not os.path.exists(save_file):
        init_mnist()

    with open(save_file, 'rb') as f:
        dataset = pickle.load(f)

    if normalize:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0

    if one_hot_label:
        dataset['train_label'] = _change_one_hot_label(dataset['train_label'])
        dataset['test_label'] = _change_one_hot_label(dataset['test_label'])

    if not flatten:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)

    return dataset['train_img'], dataset['train_label'], dataset['test_img'], dataset['test_label']


def show_image(idx=0):
    train_img, train_label, test_img, test_label = load_mnist(normalize=False, flatten=False)
    img = train_img[idx]
    img = img.reshape(28, 28)
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


if __name__ == '__main__':
    init_mnist()
