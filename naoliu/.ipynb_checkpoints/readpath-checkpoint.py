import os
import numpy as np
import csv
from os import listdir
import pydicom
import cv2
import scipy.misc


i = 0
rows = []
csv_reader = csv.reader(open('/home/data_001/datasets/naoliu/MR/csv_M_2.csv', "rt"))

pic_list = []
label_list = []
for row in csv_reader:
    rows.append(row)
    filename_list = listdir(row[6])
    for filename in filename_list:
        if filename.startswith('4_'):
            pic_part=[]
            image_list = listdir(row[6]+'/'+filename)
            if len(image_list)>23:
                for i in range(23):
                     in_path = row[6]+ '/' + filename + '/' + image_list[i]
                     ds = pydicom.read_file(in_path)
                     img = ds.pixel_array
                     pic = scipy.misc.imresize(img, (224,224))
                     pic_part.append(pic)
                if row[7].startswith('M'):
                     pic_list.append(pic_part)
                     label_list.append(row[7])
            else:
                continue


l =  len(pic_list)
print(l)
pic_np = np.asarray(pic_list)
label_np = np.asarray(label_list)
pic_np = pic_np.reshape((l,224,224,23))
label_np = label_np.reshape((l, 1))
print(pic_np.shape,label_np.shape)

