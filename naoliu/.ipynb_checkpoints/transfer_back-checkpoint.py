import numpy as np
import tensorflow as tf
from PIL import Image
import os
import matplotlib.pyplot as plt
import argparse
import cv2
import scipy.misc
from PIL import Image
import pylab
import datetime

starttime = datetime.datetime.now()
#long running

def read_and_decode(filename):
    #根据文件名生成一个队列
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   #返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [297, 159, 3])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5

    return img




init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    threads = tf.train.start_queue_runners(sess=sess)
    img = read_and_decode("./0011.tfrecords")

endtime = datetime.datetime.now()
print((endtime - starttime).seconds)
