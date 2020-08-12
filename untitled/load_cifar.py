class Cifar10DataReader():
    import os
    import random
    import numpy as np
    import pickle

    def __init__(self, cifar_file, one_hot=False, file_number=1):
        self.batch_index = 0  # 第i批次
        self.file_number = file_number  # 第i个文件数r
        self.cifar_file = cifar_file  # 数据集所在dir
        self.one_hot = one_hot
        self.train_data = self.read_train_file()  # 一个数据文件的训练集数据，得到的是一个1000大小的list，
        self.test_data = self.read_test_data()  # 得到1000个测试集数据

    # 读取数据函数，返回dict
    def unpickle(self, file):
        with open(file, 'rb') as fo:
            try:
                dicts = self.pickle.load(fo, encoding='bytes')
            except Exception as e:
                print('load error', e)
            return dicts

    # 读取一个训练集文件,返回数据list
    def read_train_file(self, files=''):
        if files:
            files = self.os.path.join(self.cifar_file, files)
        else:
            files = self.os.path.join(self.cifar_file, 'data_batch_%d' % self.file_number)
        dict_train = self.unpickle(files)
        train_data = list(zip(dict_train[b'data'], dict_train[b'labels']))  # 将数据和对应标签打包
        self.np.random.shuffle(train_data)
        print('成功读取到训练集数据：data_batch_%d' % self.file_number)
        return train_data

    # 读取测试集数据
    def read_test_data(self):
        files = self.os.path.join(self.cifar_file, 'test_batch')
        dict_test = self.unpickle(files)
        test_data = list(zip(dict_test[b'data'], dict_test[b'labels']))  # 将数据和对应标签打包
        print('成功读取测试集数据')
        return test_data

    # 编码得到的数据，变成张量，并分别得到数据和标签
    def encodedata(self, detum):
        rdatas = list()
        rlabels = list()
        for d, l in detum:
            rdatas.append(self.np.reshape(self.np.reshape(d, [3, 1024]).T, [32, 32, 3]))
            if self.one_hot:
                hot = self.np.zeros(10)
                hot[int(l)] = 1
                rlabels.append(hot)
            else:
                rlabels.append(l)
        return rdatas, rlabels

    # 得到batch_size大小的数据和标签
    def nex_train_data(self, batch_size=100):
        assert 1000 % batch_size == 0, 'erro batch_size can not divied!'  # 判断批次大小是否能被整除

        # 获得一个batch_size的数据
        if self.batch_index < len(self.train_data) // batch_size:  # 是否超出一个文件的数据量
            detum = self.train_data[self.batch_index * batch_size:(self.batch_index + 1) * batch_size]
            datas, labels = self.encodedata(detum)
            self.batch_index += 1
        else:  # 超出了就加载下一个文件
            self.batch_index = 0
            if self.file_number == 5:
                self.file_number = 1
            else:
                self.file_number += 1
            self.read_train_file()
            return self.nex_train_data(batch_size=batch_size)
        return datas, labels

    # 随机抽取batch_size大小的训练集
    def next_test_data(self, batch_size=100):
        detum = self.random.sample(self.test_data, batch_size)  # 随机抽取
        datas, labels = self.encodedata(detum)
        return datas, labels


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    Cifar10 = Cifar10DataReader(r'E:/dataset/cifar10/cifar', one_hot=True)
    d, l = Cifar10.nex_train_data()
    print(len(d))
    print(d)
    plt.imshow(d[0])
    plt.show()