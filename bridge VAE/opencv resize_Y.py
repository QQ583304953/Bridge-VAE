#竖向放大图像
import numpy as np
import cv2
import os

'''
#为了避免误操作，故不用时让代码失效
#opencv缩放图像，必然导致图像尺寸改变，故需要剪裁图片，有点麻烦，网上查了其它图像模块都是如此麻烦。为了编程方便，这里仅放大图像，不缩小。
root='.\\png 512x128_GRAY\\'
for dir in os.listdir(root):
    print(root+dir)
    for file_name in os.listdir(root+dir):
        this_file_name=root+dir+'\\'+file_name
        name, extension = os.path.splitext(this_file_name) #分割文件名(如“A010125”)、后缀名(如“.png”)
        img = cv2.imread(this_file_name,-1)#print(img.shape[0], img.shape[1]) #img: 128 x 512
        for y_scale in np.linspace(1,1.1,5):
            #print('y_scale=',y_scale)
            y_resize=int(int(img.shape[0]/2*y_scale)*2)
            # 缩放图像
            dst = cv2.resize(img, (512,y_resize)) #横向像素数目（如1920*1080的1920）*竖向像素数目（如1920*1080的1080）
            # 显示图像
            #cv2.imshow("dst: %d x %d" % (dst.shape[0], dst.shape[1]), dst)
            # 裁剪图像
            cropped_image = dst[int((y_resize-128)/2):int((y_resize+128)/2),0:512]
            # 显示裁剪图像
            #cv2.imshow("cropped_image: %d x %d" % (cropped_image.shape[0], cropped_image.shape[1]), cropped_image)
            #cv2.waitKey(0)
            cv2.imwrite(name+'_resize_Y='+str(round(y_scale,3))+extension,cropped_image)
        os.remove(this_file_name)
'''






#为了避免误操作，故不用时让代码失效







