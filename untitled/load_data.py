import pickle
import pandas as pd
import gzip



# def read_gzip(files):
#     with gzip.open(files, 'rb') as fo:
#         # dicts = pd.DataFrame(fo)
#         dicts = pickle.load(fo, encoding='bytes')
#     return dicts
#
#
# def unpickle(file_):
#     with open(file_, 'rb') as foo:
#         dicts_ = pickle.load(foo, encoding='bytes')
#     return dicts_
#
#
# if __name__ == '__main__':
#
#     dicts = unpickle(r"E:/dataset/cifar10/cifar")
#     df = pd.DataFrame(dicts)
#     print(df.head())

with open(r'E:/dataset/cifar10/cifar.rt', 'rb') as fo:
    dic = pickle.load(fo, encoding='bytes')
