#读取图片的像素总和
#《Deep Learning with Python-Chollet2018.pdf》P145
import numpy as np
import cv2

a=['.\\png 512x128_GRAY\\Arch Bottom_bear\\Arch Bottom_bear0000_Rotate=0.0_resize_X=1.0_resize_Y=1.0.png',
   '.\\png 512x128_GRAY\\Arch Top_bear\\Arch Top_bear0000_Rotate=0.0_resize_X=1.0_resize_Y=1.0.png',
   '.\\png 512x128_GRAY\\Beam Three_span\\Beam Three_span0000_Rotate=0.0_resize_X=1.0_resize_Y=1.0.png',
   '.\\png 512x128_GRAY\\Beam V_type\\Beam V_type0000_Rotate=0.0_resize_X=1.0_resize_Y=1.0.png']
for ai in a:
    img = cv2.imread(ai,-1)
    print(ai+'_np.sum(img)=',np.sum(img)) #结果见“四种桥型像素均衡化的措施2023.09.26.xlsx”


'''
.\png 512x128_GRAY\Arch Bottom_bear\Arch Bottom_bear0000_Rotate=0.0_resize_X=1.0_resize_Y=1.0.png_np.sum(img)= 550375
.\png 512x128_GRAY\Arch Top_bear\Arch Top_bear0000_Rotate=0.0_resize_X=1.0_resize_Y=1.0.png_np.sum(img)= 420351
.\png 512x128_GRAY\Beam Three_span\Beam Three_span0000_Rotate=0.0_resize_X=1.0_resize_Y=1.0.png_np.sum(img)= 209638
.\png 512x128_GRAY\Beam V_type\Beam V_type0000_Rotate=0.0_resize_X=1.0_resize_Y=1.0.png_np.sum(img)= 267115
'''