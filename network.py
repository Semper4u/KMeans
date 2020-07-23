import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("E:/dataset/mnist/", one_hot=True)


def fullconnected():             # 神经网络方法
    # 1.建立数据的占位符
    with tf.variable_scope("data"):
        x = tf.placeholder(tf.float32, [None, 784])
        y_true = tf.placeholder(tf.float32, [None, 10])

    # 2. 建立一个全连接层的神经网络  w = [784, 10]   b=[10]
    with tf.variable_scope("full_model"):
        #  随机初始化权重偏置
        weight = tf.Variable(tf.random_normal([784, 10], 0.0, 1.0, name="weight"))
        bias = tf.Variable(tf.ones([10]), name="bias")

        # 预测none个样本的输出结果
        y_predict = tf.matmul(x, weight)+bias

    #  3. 计算损失函数
    with tf.variable_scope("loss"):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=y_true, logits=y_predict))

    # 4.梯度下降优化
    with tf.variable_scope("optimizer"):
        train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    # 5.计算准确率
    with tf.variable_scope("precision"):
        equal_list = tf.equal(tf.argmax(y_true, 1), tf.argmax(y_predict, 1))
        accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))

    # 收集变量
    tf.summary.scalar("losses", loss)
    tf.summary.scalar("precision", accuracy)
    tf.summary.histogram("weightes", weight)
    tf.summary.histogram("bias", bias)
    # 合并变量
    merged = tf.summary.merge_all()

    init_op = tf.global_variables_initializer()
    # 6.开启会话训练
    with tf.Session() as sess:
        sess.run(init_op)

        # 建立events文件，写入
        filewriter = tf.summary.FileWriter("./GRAPH/", graph=sess.graph)
        # 迭代训练
        for i in range(2000):
            mnist_x, mnist_y = mnist.train.next_batch(50)
            sess.run(train_op, feed_dict={
                x: mnist_x,
                y_true: mnist_y
            })
            # 写入每步训练的值
            summary = sess.run(merged, feed_dict={x: mnist_x,
                y_true: mnist_y})
            filewriter.add_summary(summary, i)

            print("训练第%d步，准确率为：%f" % (i, sess.run(accuracy, feed_dict={x: mnist_x,
                y_true: mnist_y})))



"""以下为用卷积神经网络实现手写数字分类
"""


def weight_variables(shape):
    w = tf.Variable(tf.random_normal(shape, mean=0.0, stddev=1.0))
    return w


def bias_variables(shape):
    b = tf.Variable(tf.constant(0.0, shape=shape))
    return b


def model():
    """
    自定义卷积模型
    :return:
    """
    # 1, 定义数据占位符
    with tf.variable_scope("data"):
        x = tf.placeholder(tf.float32, [None, 784])
        y_true = tf.placeholder(tf.float32, [None, 10])

    # 2.卷积层一:  卷积核5*5*1， 32个，strids = 1， 激活， 池化
    with tf.variable_scope('conv1'):
        # 首先改变形状，使之满足卷积输入[none,784]---[none,28,28,1]
        x_reshape = tf.reshape(x, [-1, 28, 28, 1])

        # 随机初始化权重， 偏置
        w_conv1 = weight_variables([5, 5, 1, 32])
        b_conv1 = bias_variables([32])

        # 卷积层一----[none, 28, 28, 1]----[none, 28, 28, 32]
        conv1 = tf.nn.conv2d(x_reshape, filter=w_conv1, strides=[1, 1, 1, 1],
                     padding="SAME") + b_conv1
        # 激活
        relu_conv1 = tf.nn.relu(conv1)

        # 池化    [none, 28, 28, 32]-----[none, 14, 14, 32]
        x_pool1 = tf.nn.max_pool(relu_conv1, ksize=[1, 2, 2, 1],
                                 strides=[1, 2, 2, 1], padding="SAME")

    # 3.卷积层二:  卷积核5*5*32， 64个，strids = 1， 激活， 池化
    with tf.variable_scope('conv2'):
        # 随机初始化权重， 偏置
        w_conv2 = weight_variables([5, 5, 32, 64])
        b_conv2 = bias_variables([64])

        # 卷积层二， [none, 14, 14, 32]----[none, 14, 14, 64]
        conv2 = tf.nn.conv2d(x_pool1, filter=w_conv2, strides=[1, 1, 1, 1],
                     padding="SAME")+b_conv2

        # 激活
        relu_conv2 = tf.nn.relu(conv2)

        # 池化      [none, 14, 14, 64]----[none, 7, 7, 64]
        x_pool2 = tf.nn.max_pool(relu_conv2, ksize=[1, 2, 2, 1], strides=
                                 [1, 2, 2, 1], padding="SAME")

    # 4.全连接层  [none, 7, 7, 64]---[none,7*7*64]**[7*7*64, 10]----[none, 10]
    with tf.variable_scope("full_connection"):
        # 随机初始化权重， 偏置
        w_full = weight_variables([7*7*64, 10])
        b_full = bias_variables([10])

        # 计算预测值
        # 修改上一个池化层的形状
        x_pool = tf.reshape(x_pool2, [-1, 7*7*64])
        y_predict = tf.matmul(x_pool, w_full)+b_full

    return x, y_true, y_predict


def conv_fu():             # 卷积神经网络方法

    x, y_true, y_predict = model()

    #   计算损失函数
    with tf.variable_scope("loss"):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=y_true, logits=y_predict))

    # 梯度下降优化
    with tf.variable_scope("optimizer"):
        train_op = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)

    # 计算准确率
    with tf.variable_scope("precision"):
        equal_list = tf.equal(tf.argmax(y_true, 1), tf.argmax(y_predict, 1))
        accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))

    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)

        for i in range(10000):
            mnist_x, mnist_y = mnist.train.next_batch(50)
            sess.run(train_op, feed_dict={
                x: mnist_x,
                y_true: mnist_y
            })
            # 写入每步训练的值
            # summary = sess.run(merged, feed_dict={x: mnist_x,
            #                                       y_true: mnist_y})
            # filewriter.add_summary(summary, i)

            print("训练第%d步，准确率为：%f" % (i, sess.run(accuracy, feed_dict={x: mnist_x,
                                                                       y_true: mnist_y})))


if __name__ == '__main__':
    conv_fu()
