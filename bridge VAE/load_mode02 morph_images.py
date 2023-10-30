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

#固定参数
epochs=10
h5_fileName='1nd Edition - %dD 02_epochs=%d_kl_loss_coefficient=%.4f'%(latent_dim,epochs,kl_loss_coefficient)
png_folder='.\\VAE - %dD_img\\'%(latent_dim)
img_shape = (128, 512, 1) #对于1920*1080电影截屏图片，第一个1920是列（图片水平向像素数量）、第二个1080是行（图片竖向像素数量）。所以输入神经网络时，input_shape=(720 * 1980,)
n = 20 #每行图片数量

#加载模型，P255
#vae = load_model('vae.h5', custom_objects={'CustomVariationalLayer': CustomVariationalLayer})  # 从文件*.h5 中载入模型
#vae = load_model('vae.h5')  # 这个模型含自定义的层layer或者损失函数loss，加载总是出错，上网查询了，搞不定
#vae.summary()
#encoder = load_model(os.path.join(h5_fileName + '_encoder.h5'))  # 从文件*.h5 中载入模型
#encoder.summary()
decoder = load_model(os.path.join(h5_fileName + '_decoder.h5'))  # 从文件*.h5 中载入模型
decoder.summary()

def get_central_coordinate(): #计算单个桥型的所有样本的中心坐标。自编代码，原代码不好。
    central_coordinate=list() #中心坐标列表

    #加载"load_mode spatial_distribution_maps -2D.py"生成的数据集标签、编码坐标数据
    labels_all=np.loadtxt(png_folder +h5_fileName+ '_labels_all.txt',dtype=int)
    #print('labels_all=',labels_all) #labels_all= [7 2 7 ... 2 4 6]
    z_points=np.loadtxt(png_folder +h5_fileName+ '_z_points.txt')
    #print('z_points=',z_points) #z_points= [[ 0.22860715  1.68702447] [-0.36188382 -0.2625753 ]  ... [ 1.89099419  0.03854401] [ 0.72789901  1.38185728]]
    
    for i in range(8): #8种桥型的标签
        #print("i=",i)
        current_z= z_points[labels_all==i]#根据变量labels_all值，划分数据集z_points
        #print('current_z.shape=',current_z.shape) #current_z.shape= (1200, 2)，即样本集中有1200个样本满足变量labels_all==i要求
        current_sum = np.sum(current_z, axis = 0) #样本维度值的和，axis = 0让样本相加、维度数目不变，最终shape=(2,)
        current_mean= current_sum / len(current_z) #维度均值，平均空间坐标
        #print('current_mean=',current_mean)#current_mean= [-1.3793593   1.10789724]
        central_coordinate.append(list(current_mean))

    return central_coordinate   
central_coordinate=get_central_coordinate()
np.savetxt(png_folder +h5_fileName+ '_central_coordinate.txt', central_coordinate) #python自身的写文件方式，大型矩阵会出现省略号，导致txt文件中数据不全。np.savetxt()存数据到本地，np.loadtxt()从本地文件读取
#print('central_coordinate=',central_coordinate)#central_coordinate= [[-1.3793592993666728, 1.107897238265723], [0.8759388037274282, -1.434982194850842], [-1.1248372309406598, -0.51925582960248], [-0.6153662838724753, -1.5113180011014145], [1.2610451300938925, 0.12959922909659022], [1.443844306493799, 0.05666283542678381], [0.707865101446708, 1.0083138615017135], [-0.0479549014964141, 1.4491035128136476]]
#print('central_coordinate[6]=',central_coordinate[6])

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

counter = 0 #计数
figure = np.zeros((img_shape[0]*28, img_shape[1] * n,3)) #因为要添加文字，故RGB格式.28行*n列。八种桥型中，任意选两种，combinations(range(8), 2)=8*7/2=28种选择。
factors = np.linspace(0, 1, n)  #中间过渡的比例因子。np.linspace(0, 1, 11)=[0.  0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1. ]
from itertools import combinations
for i,j in combinations(range(8), 2): #八种桥型中，任意选两种，combinations(range(8), 2)=8*7/2=28种选择。
    morph_images(i,j,central_coordinate[i],central_coordinate[j],factors)
    counter+=1
img = image.array_to_img(figure * 255., scale=False)
img.save(os.path.join(png_folder,h5_fileName+ '_morph_images.png'))




