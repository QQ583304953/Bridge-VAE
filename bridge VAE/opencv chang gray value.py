#读取图片
#《Deep Learning with Python-Chollet2018.pdf》P145
import numpy as np
import cv2
import os

#为了避免误操作，故不用时让代码失效
'''
root='.\\png 512x128_GRAY\\'
for dir in os.listdir(root):
    print(root+dir)
    i=0
    for file_name in os.listdir(root+dir):
        print(i)
        this_file_name=root+dir+'\\'+file_name
        img = cv2.imread(this_file_name,-1)
        img=img*((255-i)/255)
        img=img.astype('uint8')
        cv2.imwrite(this_file_name,img)
        i +=1
'''






