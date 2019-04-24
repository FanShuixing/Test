#coding=utf-8

import math 
import os 
import cv2 
import numpy as np 
import scipy.misc
from PIL import Image
import sys

meta_path = './meta_new_vaild.txt'

i = 0
size = 256
list_x =[]
list_y =[]
count = 0

while True:
    with open(meta_path) as f:
        for line in f:
            print(line)
            train_path, M_type, degred= line.split(' ')
            #train_path = train_path.replace("\"","").replace("\"","")
            #if os.path.exists(train_path)==True:
                #print(train_path)
            list_0 = os.listdir(train_path)
            length = len(list_0)
            print(length)
            if length>22:
                pic_part = []
                label_part = []
                for i in range(23):
                    train_pic = train_path + '/' + list_0[i]
                    train = Image.open(train_pic)
                    train = train.resize((size,size))
                    train = np.asarray(train)
                    pic_part.append(train)                        
                pic_part = np.asarray(pic_part)
                print(pic_part.shape)
                list_x.append(pic_part)
                list_y.append(degred)
                count += 1 
                print(count)
                if count==157:
                    print('start')
                    list_x = np.asarray(list_x)
                    print(list_x.shape)
                    list_x = list_x.reshape((count, size, size, 23))
                    list_y = np.asarray(list_y)
                    list_y = list_y.reshape((count, 1))
                    np.save('./x_vaild_new.npy', list_x)
                    np.save('./y_vaild_new.npy', list_y)
                    print(list_x.shape, list_y.shape)
                    sys.exit()