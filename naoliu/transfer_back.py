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
import time
import base64

def decode_from_tfrecords(filename_queue): 
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   #返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       }) 
    image = tf.decode_raw(features['img_raw'],tf.int8)
    image = tf.reshape(image, [297, 159])

    return image
 

 
if __name__=='__main__':
    start = time.time()
    #long running
    a = np.fromfile('./0011.bin')
    #a = a.encode('utf-8').decode('unicode_escape')
    print(a.shape)
    #a = a.reshape((297, 159, 3))
    end = time.time()
    print(end-start)
    
'''
    train_filename = "./0011.tfrecords"
    filename_queue = tf.train.string_input_producer([train_filename]) #读入流中
    train_image = decode_from_tfrecords(filename_queue)
    
    img_path = './0011.npy'
    c = np.load(img_path)
    c = c.reshape((297, 159, 3))
'''