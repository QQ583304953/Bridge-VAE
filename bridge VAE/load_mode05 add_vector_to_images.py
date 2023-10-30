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
file_name='Arch Bottom_bear0064.png' #需要添加微笑向量的原始图片
i=6 #微笑向量的起点桥型标签
j=7 #微笑向量的终点桥型标签

#固定参数
latent_dim = 8
kl_loss_coefficient=1
epochs=10
h5_fileName='1nd Edition - %dD 02_epochs=%d_kl_loss_coefficient=%.4f'%(latent_dim,epochs,kl_loss_coefficient)
png_folder='.\\VAE - %dD_img\\'%(latent_dim)
img_shape = (128, 512, 1) #对于1920*1080电影截屏图片，第一个1920是列（图片水平向像素数量）、第二个1080是行（图片竖向像素数量）。所以输入神经网络时，input_shape=(720 * 1980,)

#加载模型，P255
#vae = load_model('vae.h5', custom_objects={'CustomVariationalLayer': CustomVariationalLayer})  # 从文件*.h5 中载入模型
#vae = load_model('vae.h5')  # 这个模型含自定义的层layer或者损失函数loss，加载总是出错，上网查询了，搞不定
#vae.summary()
encoder = load_model(os.path.join(h5_fileName + '_encoder.h5'))  # 从文件*.h5 中载入模型
encoder.summary()
decoder = load_model(os.path.join(h5_fileName + '_decoder.h5'))  # 从文件*.h5 中载入模型
decoder.summary()
central_coordinate=np.loadtxt(png_folder +h5_fileName+ '_central_coordinate.txt')

def read_image_file(file_name): #我自编函数。读取图片文件
    img = image.load_img(file_name,color_mode='grayscale') #默认RGB格式
    x = image.img_to_array(img)/255 #必须除以255。x.shape= (128, 512, 1)
    return x
my_image=read_image_file(png_folder+ file_name )
n=5#图片总数量
figure = np.zeros((img_shape[0]*2, img_shape[1]*n,3))#第一行放生成图片；第二行是生成原始图片
figure[img_shape[0]: 2*img_shape[0],int(n/2)*img_shape[1]: (int(n/2)+1)*img_shape[1],:] = my_image
info='original_image' #文字内容
p_center=(int(int(n/2) * img_shape[1]+img_shape[1]*2/5),int(img_shape[0]*(1+4.4/6))) #文字位置
cv2.putText(figure,info,p_center,cv2.FONT_HERSHEY_PLAIN,1,(1,1,1),1)

#隐空间的算术（先求微笑向量，再将某图片加上微笑向量的倍数，生成微笑图片）。示例见《生成式深度学习》P105图3-21。代码复制自《生成式深度学习》“03_02_autoencoder_analysis.ipynb”
def add_vector_to_images(my_image,current_mean_POS,current_mean_NEG):
    my_image=np.expand_dims(my_image, axis=0) # 其形状由(150, 150, 1)变为(1, 150, 150, 1)
    z_points = encoder.predict(my_image)[0] #编码器。encoder.predict()[0]是批量的坐标值，encoder.predict()[1]是批量的方差
    #print('z_points=',z_points)
    vector=np.array(current_mean_NEG)-np.array(current_mean_POS) #微笑向量
    #print('vector=',vector)
    factors = np.linspace(-2, 2, n)  #中间过渡的比例因子。np.linspace(-4, 4, 9)=[-4. -3. -2. -1.  0.  1.  2.  3.  4.]
    for i,factor in enumerate(factors): 
        changed_z_point = z_points+vector*factor #执行算术之后的空间坐标
        #print('changed_z_point=',changed_z_point)#changed_z_point= [[ 0.34436779 -7.7868424  -3.26988994 -1.10186805 -0.64099886  9.03514959   2.38252223  4.85790545]]
        changed_image = decoder.predict(changed_z_point) #解码
        img = changed_image[0]
        figure[0: img_shape[0],i*img_shape[1]: (i+1)*img_shape[1],:] = img
        info='x=%.1f,y=%.1f,z=%.1f,t=%.1f,d=%.1f,c=%.1f,b=%.1f,a=%.1f'%(changed_z_point[0][0],changed_z_point[0][1],changed_z_point[0][2],changed_z_point[0][3],changed_z_point[0][4],changed_z_point[0][5],changed_z_point[0][6],changed_z_point[0][7]) #文字内容
        info1='factor=%d'%(factor) #文字内容
        p_center=(int(i * img_shape[1]+img_shape[1]*1/10),int(img_shape[0]*5/6)) #文字位置
        p_center1=(int(i * img_shape[1]+img_shape[1]*3/7),int(img_shape[0]*4.4/6)) #文字位置
        cv2.putText(figure,info,p_center,cv2.FONT_HERSHEY_PLAIN,0.9,(1,1,1),1) #color=(1,1,1)的原因：保存文件时要乘以255。《OpenCV 4.5计算机视觉开发实战（基于Python）》P81
        cv2.putText(figure,info1,p_center1,cv2.FONT_HERSHEY_PLAIN,1,(1,1,1),1)
add_vector_to_images(my_image,central_coordinate[i],central_coordinate[j])
img = image.array_to_img(figure * 255., scale=False)
img.save(os.path.join(png_folder, file_name+'_add_%dto%d_vector_to_images.png'%(i,j)))








