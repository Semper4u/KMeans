import numpy as np
import tensorflow as tf
import cv2


def text():
    data = tf.data.Dataset.from_tensor_slices({
        'a': np.array([1, 2, 4, 6, 8]),
        'b': np.random.uniform(1, 10, [5, 2])
    })
    iter = data.make_one_shot_iterator()
    one_element = iter.get_next()

    with tf.Session() as sess:
        for i in range(5):
            print(sess.run(one_element))


def parse_dataset(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string)
    image_resized = tf.image.resize_images(image_decoded, [56, 56])
    image_resized = tf.reshape(image_resized, [56, 56, 3])
    print(image_resized)
    return image_resized, label


def re_text():
    # dataset默认集成了c++的多线程，因此使用此方式读取比较方便，在以后的应用中，应用os模块获取文件夹中图片名，随后便可应用此函数读取图像，
    #   返回值为（batch—size， 长，宽， 通道数）便于训练
    filenames = tf.constant(["./data/pic/0pic.jpg",
                             "./data/pic/1pic.jpg",
                             "./data/pic/2pic.jpg",
                             "./data/pic/3pic.jpg",
                             "./data/pic/4pic.jpg"])
    labels = tf.constant([0, 1, 2, 3, 4])
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(parse_dataset)
    dataset = dataset.shuffle(buffer_size=1).batch(batch_size=2, drop_remainder=True).repeat(3)
    print(dataset)
    iter = dataset.make_one_shot_iterator()
    one_ele = iter.get_next()

    with tf.Session() as sess:
        for i in range(5):
            print(sess.run(one_ele))

            print('*'*80)
            print(one_ele)


def read_csv():            # 读取文本文件，可含csv等格式
    data = tf.data.TextLineDataset(['./data/csv.csv'])
    iter = data.make_one_shot_iterator()
    one_ele = iter.get_next()

    with tf.Session() as sess:
        for i in range(5):
            print(sess.run(one_ele))


if __name__ == '__main__':
    re_text()
