#平移图片
#《Deep Learning with Python-Chollet2018.pdf》P145
import numpy as np
import cv2
import os

# 定义move函数
def ImageMove(image,offset_x,offset_y): # offset_x指水平向移动的像素（向右为正）,offset_y指竖向移动的像素（向下为正）
    height, width = image.shape[:2]    # height=竖向像素数目（如1920*1080的1080）, width=横向像素数目
    #print('height=',height)#height, width=128,512
    # 获得平移矩阵
    M = np.float32([[1,0,offset_x],[0,1,offset_y]])
    # 进行仿射变换
    image_move = cv2.warpAffine(src=image, M=M, dsize=(width,height)) 
    return image_move


'''
#为了避免误操作，故不用时让代码失效
root='.\\png 512x128_GRAY\\'
for dir in os.listdir(root):
    print(root+dir)
    for file_name in os.listdir(root+dir):
        this_file_name=root+dir+'\\'+file_name
        #print('this_file_name=',this_file_name)
        name, extension = os.path.splitext(this_file_name) #分割文件名(如“A010125”)、后缀名(如“.png”)
        #print("文件名: ", name)
        #print("后缀: ", extension)
        img = cv2.imread(this_file_name,-1)
        for offset_x in [-3,0,3]:
            for offset_y in [-3,0,3]:
                img1=ImageMove(img,offset_x,offset_y) # offset_x指水平向移动的像素（向右为正）,offset_y指竖向移动的像素（向下为正）
                #cv2.imshow('img',img1)
                #cv2.waitKey(0)
                cv2.imwrite(name+'_offset_x='+str(offset_x)+'_offset_y='+str(offset_y)+extension,img1)
        os.remove(this_file_name)
'''







