#旋转图片
#《Deep Learning with Python-Chollet2018.pdf》P145
import numpy as np
import cv2
import os

# 定义旋转函数
def ImageRotate(image,angle): # 旋转角度，正为逆时针，负为顺势针
    height, width = image.shape[:2]    # height=竖向像素数目（如1920*1080的1080）, width=横向像素数目
    #print('height=',height)#height, width=128,512
    center = (width / 2, height / 2)   # 绕图片中心（x、y坐标）进行旋转
    # 获得旋转矩阵
    M = cv2.getRotationMatrix2D(center, angle, 1)
    #print('M=',M)
    # 进行仿射变换
    image_rotation = cv2.warpAffine(src=image, M=M, dsize=(width,height)) 
    return image_rotation

'''
#为了避免误操作，故不用时让代码失效
root='.\\png 512x128_GRAY\\'
for dir in os.listdir(root):
    print(root+dir)
    for file_name in os.listdir(root+dir):
        this_file_name=root+dir+'\\'+file_name
        name, extension = os.path.splitext(this_file_name) #分割文件名(如“A010125”)、后缀名(如“.png”)
        img = cv2.imread(this_file_name,-1)
        for angle in np.linspace(-0.3,0.3,3):#np.linspace(-1,1,7)=[-1.    -0.66666667 -0.33333333  0.    0.33333333  0.66666667  1. ]
            #print(angle)
            img1 = ImageRotate(img,angle)
            cv2.imwrite(name+'_Rotate='+str(round(angle,1))+extension,img1)
        os.remove(this_file_name)
'''






