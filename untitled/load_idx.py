import numpy as np
import struct
import os
import pandas as pd
from matplotlib import pyplot as plt


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


def load_mnist_train(path, kind='train'):
    labels_path = os.path.join(path, '%s-labels.idx1-ubyte'% kind)
    images_path = os.path.join(path, '%s-images.idx3-ubyte'% kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
    return images, labels


def load_mnist_test(path, kind='t10k'):
    labels_path = os.path.join(path, '%s-labels.idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images.idx3-ubyte' % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
    return images, labels


path = 'E:\dataset\mnist'
train_images, train_labels = load_mnist_train(path)
test_images, test_labels = load_mnist_test(path)
df = pd.DataFrame(test_images).head()
print(df.shape)

list_name = ['train_images', 'train_labels', 'test_images', 'test_labels']
for temp in list_name:
    temp_str = eval(temp)
    df = pd.DataFrame(temp_str)
    df.to_csv(r"E:/dataset/mnist/%s.csv" % temp, header=False, index=False)

# for i in range(5):
#     pic = np.array((df.loc[i])).reshape(28, 28)
#     plt.imshow(pic)
#     plt.show()
#     plt.imsave("D:/%dpic.jpg" %i, arr=pic)