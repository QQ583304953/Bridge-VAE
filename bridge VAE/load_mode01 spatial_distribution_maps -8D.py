#必须python3.8 GPU运行，而python3.9 CPU死机
#代码复制自《生成式深度学习》“03_02_autoencoder_analysis.ipynb”
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import norm
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

#可变参数
latent_dim = 8
kl_loss_coefficient=1 

#固定参数
epochs=10
png_folder='.\\VAE - %dD_img\\'%(latent_dim)
img_shape = (128, 512, 1) 
train_dir='.\\png 512x128_GRAY' 
h5_fileName='1nd Edition - %dD 02_epochs=%d_kl_loss_coefficient=%.4f'%(latent_dim,epochs,kl_loss_coefficient)
batch_size = 128 #取256则出错

#统计文件夹中的文件总个数（含所有的子目录下的）
def count_files_in_folder(folder_path): 
    count = 0
    for _, _, files in os.walk(folder_path):
        count += len(files)
    return count
file_count = count_files_in_folder(train_dir)
steps_per_epoch=file_count//batch_size


#加载模型，P255
#vae = load_model('vae.h5', custom_objects={'CustomVariationalLayer': CustomVariationalLayer})  # 从文件*.h5 中载入模型
#vae = load_model('vae.h5')  # 这个模型含自定义的层layer或者损失函数loss，加载总是出错，上网查询了，搞不定
#vae.summary()
encoder = load_model(os.path.join(h5_fileName + '_encoder.h5'))  # 从文件*.h5 中载入模型
encoder.summary()
decoder = load_model(os.path.join(h5_fileName + '_decoder.h5'))  # 从文件*.h5 中载入模型
decoder.summary()

#加载数据集，P108
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
                                                    train_dir, #各个子目录自动打乱的，某个子目录中文件也是打乱的
                                                    target_size=(img_shape[0], img_shape[1]),
                                                    color_mode="grayscale",
                                                    batch_size=batch_size,
                                                    class_mode="sparse" #返回1D的整数标签
                                                    )

labels_all=list()
for i in range(steps_per_epoch):
    for data_batch, labels_batch in train_generator:
        labels_all.extend(list(labels_batch))
        break
#print(train_generator.class_indices) #{'Arch Bottom_bear': 0, 'Arch Top_bear': 1, 'Beam Three_span': 2, 'Beam V_type': 3, 'Cable Fan_shaped': 4, 'Cable Harp_shaped': 5, 'Suspension Diagonal_sling': 6, 'Suspension Vertical_sling': 7}
np.savetxt(png_folder +h5_fileName+ '_labels_all.txt', labels_all) #python自身的写文件方式，大型矩阵会出现省略号，导致txt文件中数据不全。np.savetxt()存数据到本地，np.loadtxt()从本地文件读取

#训练集全部样本隐空间的坐标分布图，颜色区分标签。代码复制自《生成式深度学习》“03_02_autoencoder_analysis.ipynb”
z_points_all = encoder.predict_generator(train_generator, steps = steps_per_epoch, verbose = 1)#predict_generator是keras的功能，这里steps = 20表示读取行为共20次。verbose日志显示：0 静默；1 为输出进度条记录， 默认为 1。
#keras有专门的encoder.predict_generator方法，见chapter03-P102-VAE.py
#print('z_points_all=',z_points_all) #z_points_all[0]是批量的坐标值，z_points_all[1]是批量的方差
#P254代码清单8-24。
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),mean=0., stddev=1.)
    return z_mean + K.exp(0.5*z_log_var) * epsilon #第一版遗漏了0.5系数，见第二版P353
z_points =sampling(z_points_all) #采用扰动后的坐标，空间分布图好看，否则太难看了
np.savetxt(png_folder +h5_fileName+ '_z_points.txt', z_points) #python自身的写文件方式，大型矩阵会出现省略号，导致txt文件中数据不全。np.savetxt()存数据到本地，np.loadtxt()从本地文件读取

def drawFigures(normMat,datingLabels,map_name): #绘图。将数据集图形化，更加直观地理解数据。与计算无关。
    fig=plt.figure(figsize=(20, 20)) #建立绘图对象
    fig.subplots_adjust(hspace=0.6, wspace=0.4)
    counter = 1 #计数
    from itertools import combinations
    for i,j in combinations(range(8), 2): #八种桥型中，任意选两种，combinations(range(8), 2)=8*7/2=28种选择。
        ax=fig.add_subplot(4,7,counter) #二维图
        ax.scatter(normMat[:,i],normMat[:,j],cmap='rainbow',c=datingLabels, alpha=0.5, s=1)
        plt.xlabel("X%d"%i)
        plt.ylabel("X%d"%j)
        plt.xticks(np.arange(-3, 3.1, step=1))
        plt.yticks(np.arange(-3, 3.1, step=1))
        plt.gca().set_aspect(1)  #坐标比例关系为等比例
        counter+=1
    plt.savefig(os.path.join(png_folder +h5_fileName+ map_name+ '_2D.png'), dpi=300)

    fig=plt.figure() #建立绘图对象
    ax=fig.add_subplot(111) #二维图
    scatter=ax.scatter(normMat[:,1],normMat[:,7],cmap='rainbow',c=datingLabels, alpha=0.5, s=1)#空间分布图：透明度设置0.5，能够反映中心密集、边缘稀疏的特性。
    plt.xlabel("X1")
    plt.ylabel("X7")
    plt.xticks(np.arange(-3, 3.1, step=1))
    plt.yticks(np.arange(-3, 3.1, step=1))
    plt.gca().set_aspect(1)  #坐标比例关系为等比例
    #plt.legend(handles=scatter.legend_elements()[0],labels=['Arch Bottom_bear;0', 'Arch Top_bear;1', 'Beam Three_span;2', 'Beam V_type;3', 'Cable Fan_shaped;4', 'Cable Harp_shaped;5', 'Suspension Diagonal_sling;6', 'Suspension Vertical_sling;7'],title="classes_name", loc='best', fontsize=8)#显示图例（标签名是手工输入的，不支持中文）
    plt.savefig(os.path.join(png_folder +h5_fileName+ map_name+ '_2D_classic.png'), dpi=300)

drawFigures(z_points,labels_all,map_name='_spatial_distribution_maps')#绘图。

#隐空间中的各个维度的点分布，它们是否符合正态分布？示例见《生成式深度学习》P103图3-19。代码复制自《生成式深度学习》“03_02_autoencoder_analysis.ipynb”
x = np.linspace(-3, 3, 100)
fig = plt.figure(figsize=(20, 20))
fig.subplots_adjust(hspace=0.6, wspace=0.4)
for i in range(latent_dim): #latent_dim维度
    ax = fig.add_subplot(5, int(latent_dim/5+1), i+1)
    ax.hist(np.array(z_points)[:,i], density=True, bins = 30)#Matplotlib-hist-直方图。density：是否将直方图的频数转换成频率。bins 整数：分箱数目，横坐标的划分数目，这里bins=50，即横坐标有50个数值。
    # i表示维度，z_points.shape= (200, 2)[200是样本数目，2是维度数目]，即z_test[:,i]含义“第i维度的全部样本值”
    #ax.axis('off') #正态分布函数，x=0时，概率密度=0.399，所以图中竖坐标最上刻度是0.4，见“隐空间中的前50个维度的点分布.png”
    ax.text(0.5, -0.35, str(i), fontsize=10, ha='center', transform=ax.transAxes)#前两个参数：注释的坐标x、y位置（标量）。transform=ax.transAxes 表示坐标是相对于坐标区边界框给出的，其中 (0, 0) 是坐标区的左下角， (1, 1) 是右上角的坐标区。   
    ax.plot(x,norm.pdf(x))#norm.pdf是绘制正态分布函数。这行代码含义是：绘制横坐标为x的正态分布函数图，此图这里功能是与直方图对比用
#plt.show()
plt.savefig(os.path.join(png_folder +h5_fileName+ '_hist_maps.png'), dpi=300)



#训练集全部样本隐空间的方差分布图，颜色区分标签。
#z_points =z_points_all[1] #z_points_all[0]是批量的坐标值，z_points_all[1]是批量的方差
#drawFigures(z_points,labels_all,map_name='_variance_maps')#绘图。
