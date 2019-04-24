import numpy as np
import csv
from os import listdir
import pydicom
import scipy.misc
import SimpleITK as sitk
import cv2
import csv
from PIL import Image
import shutil
import os
from skimage import io,data
import scipy.misc
import random
from sklearn.utils import shuffle

pic_list = []
label_list = []

m1 = 0
m2 = 0
m3 = 0
m4 = 0
m5 = 0
m6 = 0
m7 = 0
m8 = 0
m9 = 0
m10 = 0

pwd = os.path.abspath('.')
list = os.listdir(pwd)
length = len(list)
for i in range(length):
    if 'M1' == list[i] or 'M2' == list[i] or 'M3' == list[i]:
        path = pwd+'/'+list[i]
        path_1 = os.listdir(path)
        for j in range(len(path_1)):
            path_2 = path + '/' + path_1[j]
            path_3 = os.listdir(path_2)
            if len(path_3) != 0:
                path_4 = path_2+'/'+path_3[0]
                path_5 = os.listdir(path_4)
                pic_part = []
                label_part = []
                for k in range(len(path_5)):
                     if '.jpg' in path_5[k]:
                        img_path = path_4 +'/'+path_5[k]
                        img = scipy.misc.imread(img_path)
                        pic_part.append(img)
                if len(pic_part) == 23:
                    label_part = np.zeros(3)
                    if list[i] == 'M1':
                        if int(path_1[j])<=1570:
                            label_part[0]=1
                            m1 += 1
                            pic_part = np.asarray(pic_part)
                            pic_list.append(pic_part)
                            label_list.append(label_part)
                    elif list[i] == 'M2':
                        if int(path_1[j])<=2372:
                            label_part[1]=1
                            m2 += 1
                            pic_part = np.asarray(pic_part)
                            pic_list.append(pic_part)
                            label_list.append(label_part)
                    elif list[i] == 'M3':
                        if int(path_1[j])<=796:
                            label_part[2]=1
                            m3 += 1
                            pic_part = np.asarray(pic_part)
                            pic_list.append(pic_part)
                            label_list.append(label_part)
                    
print('m1',m1,'m2',m2,'m3',m3)

l = len(pic_list)
pic_np = np.asarray(pic_list)
print('pic:',pic_np.shape)
pic_np = pic_np.reshape((l,23,512,512))
#np.save('train_512_fenlei123_32.npy',pic_np)

label_np = np.asarray(label_list)
label_np = label_np.reshape((l,3))
#np.save('label_512_fenlei123_32.npy',label_np)
print('label:',label_np.shape)

pic_np, label_np = shuffle(pic_np, label_np)
np.save('train_512_fenlei123_shuffle.npy',pic_np)
np.save('label_512_fenlei123_shuffle.npy',label_np)
