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
from skimage import io

i = 0
k = 0
rows = []
csv_reader = csv.reader(open('./csv_M_2.csv', "rt"))

pic_list = []
label_list = []
for row in csv_reader:
    M_type = row[6]
    M_type = str(M_type)
    M_type = M_type[2:-2]

    path_1 = row[7]
    path_1 = path_1.strip('.')
    path = './'+ M_type + path_1
    
    if os.path.exists(path)== True:  
        filename_list = listdir(path)
        for filename in filename_list:
            if filename.startswith('Ax_T1') and filename.endswith('+C'):
            #if filename.startswith('Ax_T1'):
                i += 1
                pic_part=[]
                image_path = path + '/' +filename
                image_list = listdir(image_path)
                print(path)
                print(len(image_list), i)
                if len(image_list)>22:
                    for i in range(len(image_list)):                     
                        in_path = image_path + '/' + image_list[i]  
                                                                                     
                        img = ds.pixel_array                    
                        pic = scipy.misc.imresize(img, (224,224))
                        pic_part.append(pic)
                    if row[7].startswith('M'):
                        pic_list.append(pic_part)
                        label_list.append(row[7])
                        print(row[6])               
                else:
                    continue
             


l =  len(pic_list)            
pic_np = np.asarray(pic_list)
label_np = np.asarray(label_list)    
pic_np = pic_np.reshape((l,224,224,23))
label_np = label_np.reshape((l, 1))
print(pic_np.shape,label_np.shape)

'''
