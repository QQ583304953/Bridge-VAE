#2023.10.12未能够发现有用的新桥型
import time
startTime=time.time()

import numpy as np
from tensorflow.keras.models import load_model
import os
import matplotlib.pyplot as plt
from scipy.stats import norm
from tensorflow.keras.preprocessing import image
import cv2

#可变参数
kl_loss_coefficient=1 
n = 2 #每个维度取值数量只能取2，否则卡死。
max_value=5 #每个维度取值的最大值

#固定参数
latent_dim = 8
epochs=10
h5_fileName='1nd Edition - %dD 02_epochs=%d_kl_loss_coefficient=%.4f'%(latent_dim,epochs,kl_loss_coefficient) 
img_shape = (128, 512, 1) #对于1920*1080电影截屏图片，第一个1920是列（图片水平向像素数量）、第二个1080是行（图片竖向像素数量）。所以输入神经网络时，input_shape=(720 * 1980,)
png_folder='.\\VAE - %dD_img\\'%(latent_dim)

#加载模型，第一版P255、第二版P356
#vae = load_model('vae.h5', custom_objects={'CustomVariationalLayer': CustomVariationalLayer})  # 从文件*.h5 中载入模型
#vae = load_model('vae.h5')  # 这个模型含自定义的层layer或者损失函数loss，加载总是出错，上网查询了，搞不定
#vae.summary()
decoder = load_model(os.path.join(h5_fileName + '_decoder.h5'))  # 从文件*.h5 中载入模型
decoder.summary()

#P255代码清单8-28。
figure = np.zeros((img_shape[0] * n, img_shape[1] * n,3)) #因为要添加文字，故RGB格式
figure_all= np.empty(shape=(0,img_shape[1] * n,3))
grid_x = grid_y =grid_z =grid_t =grid_a =grid_b =grid_c =grid_d =np.linspace(-max_value, max_value, n)#每个维度取值数量只能取2，否则卡死。
for ai in grid_a:
    for bi in grid_b:
        for ci in grid_c:
            for di in grid_d:
                for ti in grid_t:
                    for zi in grid_z:
                        for i, xi in enumerate(grid_x): #网格坐标系与数学直角坐标系一致，左下角负负、右上角正正，水平向X轴、竖向Y轴
                            for j, yi in enumerate(grid_y):
                                z_sample = np.array([[xi, yi,zi,ti,di,ci,bi,ai]])
                                x_decoded = decoder.predict(z_sample)
                                digit = x_decoded[0]
                                figure[j * img_shape[0]: (j + 1) * img_shape[0],i * img_shape[1]: (i + 1) * img_shape[1],:] = digit
                                info='x=%.1f,y=%.1f,z=%.1f,t=%.1f,d=%.1f,c=%.1f,b=%.1f,a=%.1f'%(xi, yi,zi,ti,di,ci,bi,ai) #文字内容
                                p_center=(int(i * img_shape[1]+img_shape[1]*0/5),int(j * img_shape[0]+img_shape[0]*5/6)) #文字位置
                                cv2.putText(figure,info,p_center,cv2.FONT_HERSHEY_PLAIN,1,(1,1,1),1) #color=(1,1,1)的原因：保存文件时要乘以255。《OpenCV 4.5计算机视觉开发实战（基于Python）》P81
                        figure_all=np.concatenate((figure_all,figure),axis=0)
img = image.array_to_img(figure_all * 255., scale=False)
img.save(os.path.join(png_folder,h5_fileName+'_value=%d、%d'%(-max_value, max_value)+ '_global_search_images.png'))


endTime=time.time()
print('How many seconds:',(endTime-startTime))  