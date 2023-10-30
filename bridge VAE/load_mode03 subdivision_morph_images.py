#代码复制自《生成式深度学习》“03_02_autoencoder_analysis.ipynb”
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import norm
from tensorflow.keras.models import load_model
from keras.preprocessing import image
import cv2

#可变参数
latent_dim = 8
kl_loss_coefficient=1 
n = 40 #每行图片数量
information=[[0,4,0.32,0.68],#任意epochs均可。在morph_images函数所成图上，读取其图中i、j、factor值，细分渐变图像。依次为：起点桥型标签、终点桥型标签、需要细分的factor前后值
             [0,5,0.32,0.68],#共发现5种新桥型
             [0,7,0.32,0.68],
             [1,4,0.32,0.68],
             [1,5,0.32,0.68],
             [1,7,0.32,0.68],
             [4,6,0.32,0.68],
             [4,7,0.32,0.68],
             [5,6,0.32,0.68],
             [5,7,0.32,0.68]]

#固定参数
epochs=10
h5_fileName='1nd Edition - %dD 02_epochs=%d_kl_loss_coefficient=%.4f'%(latent_dim,epochs,kl_loss_coefficient)
png_folder='.\\VAE - %dD_img\\'%(latent_dim)
img_shape = (128, 512, 1) #对于1920*1080电影截屏图片，第一个1920是列（图片水平向像素数量）、第二个1080是行（图片竖向像素数量）。所以输入神经网络时，input_shape=(720 * 1980,)

#加载模型，P255
decoder = load_model(os.path.join(h5_fileName + '_decoder.h5'))  # 从文件*.h5 中载入模型
decoder.summary()

def morph_images(i,j,current_mean_POS,current_mean_NEG,factors): #绘制两个桥型之间的渐变图像。自编代码，原代码不好。
    #movement= np.linalg.norm(np.array(current_mean_NEG)-np.array(current_mean_POS))#求向量的模，这里求两个桥型之间的空间距离。np.linalg.norm()用于求范数，linalg本意为linear(线性) + algebra(代数)，norm则表示范数，默认二范数（距离）。
    #print('两个桥型之间的空间距离=',movement)
    for k,factor in enumerate(factors): #显示执行算术之后的图像
        #print('factor=',factor)
        changed_z_point = np.array(current_mean_POS) * (1-factor) + np.array(current_mean_NEG)  * factor #执行算术之后的空间坐标
        #print('changed_z_point=',changed_z_point)
        changed_image = decoder.predict(np.array([changed_z_point])) #解码
        img = changed_image[0]
        figure[counter * img_shape[0]: (counter + 1) * img_shape[0],k * img_shape[1]: (k + 1) * img_shape[1],:] = img
        info='i=%d,j=%d,factor=%.2f'%(i, j,factor) #文字内容
        p_center=(int(k * img_shape[1]+img_shape[1]*1/3),int(counter * img_shape[0]+img_shape[0]*5/6)) #文字位置
        cv2.putText(figure,info,p_center,cv2.FONT_HERSHEY_PLAIN,1,(1,1,1),1) #color=(1,1,1)的原因：保存文件时要乘以255。《OpenCV 4.5计算机视觉开发实战（基于Python）》P81

central_coordinate=np.loadtxt(png_folder +h5_fileName+ '_central_coordinate.txt')

#在morph_images函数成图的基础上，读取其图中i、j、factor值，细分渐变图像
figure = np.zeros((img_shape[0]*len(information), img_shape[1] * n,3))
for counter,data in enumerate(information): 
    i=data[0]
    j=data[1]
    factors = np.linspace(data[2], data[3], n) 
    morph_images(i,j,central_coordinate[i],central_coordinate[j],factors)
img = image.array_to_img(figure * 255., scale=False)
img.save(os.path.join(png_folder,h5_fileName+ '_subdivision_morph_images.png'))



