#coding=utf-8

import os
from PIL import Image
import numpy as np
import cv2
import tensorflow as tf


# 数据文件
data_dir = "data/test01"
# 训练还是测试

train = False
# 模型文件路径
model_path = "model_mask/"


# 从文件夹读取图片和标签到numpy数组
# 标签信息在文件名中，例如1_40.jpg表示该图片的标签 
def read_data(data_dir):
    datas = []
    labels = []
    fpaths = []
    num0 = 0
    for fname in os.listdir(data_dir):
        fpath = os.path.join(data_dir, fname)
        fpaths.append(fpath)
        image = Image.open(fpath)
        num0 += 1
        data = np.array(image) / 255.0
        # print(data)
        label = int(fname.split("_")[0])
        datas.append(data)
        labels.append(label)

    datas = np.array(datas)
    labels = np.array(labels)

    print("shape of datas: {}\tshape of labels: {}".format(datas.shape, labels.shape))
    return fpaths, datas, labels, num0
   


fpaths, datas, labels, num0 = read_data(data_dir)

# 计算有多少类图片
num_classes = len(set(labels))


# 定义Placeholder，存放输入和标签
datas_placeholder = tf.placeholder(tf.float32, [None, 32, 32, 3])
labels_placeholder = tf.placeholder(tf.int32, [None])

# 存放DropOut参数的容器，训练时为0.25，测试时
dropout_placeholdr = tf.placeholder(tf.float32)

# 定义卷积核， 20个卷积核, 卷积核大小为5，用Relu激活
conv0 = tf.layers.conv2d(datas_placeholder, 20, 5, activation=tf.nn.relu)
# 定义max-pooling层，pooling窗口x2，步长为2x2
pool0 = tf.layers.max_pooling2d(conv0, [2, 2], [2, 2])

# 40个卷积核, 卷积核大小为4，用Relu
conv1 = tf.layers.conv2d(pool0, 40, 4, activation=tf.nn.relu)
# 定义max-pooling层，pooling窗口x2，步长为2x2
pool1 = tf.layers.max_pooling2d(conv1, [2, 2], [2, 2])

# 特征转换为1维向
flatten = tf.layers.flatten(pool1)

# 全连接层
fc = tf.layers.dense(flatten, 400, activation=tf.nn.relu)

# 加上DropOut，防止过拟合
dropout_fc = tf.layers.dropout(fc, dropout_placeholdr)

# 未激活的输出
logits = tf.layers.dense(dropout_fc, num_classes)

predicted_labels = tf.arg_max(logits, 1)


# 用于保存和载入模型
saver = tf.train.Saver()

with tf.Session() as sess:

        print("测试模式")
        # 如果是测试，载入参数
        saver.restore(sess, model_path)
        print("从{}载入模型".format(model_path))
        # label和名称的对照关系
        label_name_dict = {
            0: "no_mask",
            1: "have_mask",
          }
        # 定义输入和Label以填充容器，测试时dropout？
        test_feed_dict = {
            datas_placeholder: datas,
            labels_placeholder: labels,
            dropout_placeholdr: 0
        }
        predicted_labels_val = sess.run(predicted_labels, feed_dict=test_feed_dict)
        # 真实label与模型预测label
        num = 0
        for fpath, real_label, predicted_label in zip(fpaths, labels, predicted_labels_val):
            # 将label id转换为label 
            real_label_name = label_name_dict[real_label]
            predicted_label_name = label_name_dict[predicted_label]
            if real_label_name == predicted_label_name:
                num += 1
            print("{}\t{} => {}".format(fpath, real_label_name, predicted_label_name))
        a = num/num0
        aa = "%.2f%%" % (a * 100)
        print("=========================")
        print("识别准确率:",aa)
        print("=========================")








