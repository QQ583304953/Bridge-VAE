#这个版本速度比第二版快10%。第二版优点：损失函数细节输出
#CPU计算死机，必须GPU
import time
startTime=time.time()

#《python深度学习（第1版）》P253代码清单8-23。
import tensorflow
tensorflow.compat.v1.disable_v2_behavior() #兼容tf1版本的操作，屏蔽tf2，否则出现错误TypeError: Tensors are unhashable...
import tensorflow.keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
import numpy as np
import os

#可变参数
latent_dim = 8
kl_loss_coefficient=1

#固定参数
epochs=10 #取10最佳
img_shape = (128, 512, 1) #对于1920*1080电影截屏图片，第一个1920是列（图片水平向像素数量）、第二个1080是行（图片竖向像素数量）。所以输入神经网络时，input_shape=(720 * 1980,)
train_dir='.\\png 512x128_GRAY' 
save_fileName='1nd Edition - %dD 02_epochs=%d_kl_loss_coefficient=%.4f'%(latent_dim,epochs,kl_loss_coefficient) 
batch_size = 16 #取32则出错.增加批量规范化、Dropout[必须减少batch_size数值，否则出错]


#统计文件夹中的文件总个数（含所有的子目录下的）
def count_files_in_folder(folder_path): 
    count = 0
    for _, _, files in os.walk(folder_path):
        count += len(files)
    return count
file_count = count_files_in_folder(train_dir)
steps_per_epoch=file_count//batch_size


#神经网络
input_img = tensorflow.keras.Input(shape=img_shape)
x = layers.Conv2D(64, 3, strides=2, padding="same")(input_img)
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)
x = layers.Dropout(0.5)(x)
x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)
x = layers.Dropout(0.5)(x)
x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)
x = layers.Dropout(0.5)(x)
x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)
x = layers.Dropout(0.5)(x)
x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)
x = layers.Dropout(0.5)(x)
shape_before_flattening = K.int_shape(x)
x = layers.Flatten()(x)
z_mean = layers.Dense(latent_dim)(x)
z_log_var = layers.Dense(latent_dim)(x)
encoder = Model(input_img, [z_mean,z_log_var])
encoder.summary()


#P254代码清单8-24。
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),mean=0., stddev=1.)
    return z_mean + K.exp(0.5*z_log_var) * epsilon #第一版遗漏了0.5系数，见第二版P353
z = layers.Lambda(sampling)([z_mean, z_log_var]) #对于自定义的层、损失函数，模型训练、保存均一切正常，但是模型加载load_model('*.h5'就会出错，我搞不定
#encoder = Model(input_img, z)
#encoder.summary()


#P254代码清单8-25。
decoder_input = layers.Input(K.int_shape(z)[1:])
x = layers.Dense(np.prod(shape_before_flattening[1:]),activation='relu')(decoder_input)
x = layers.Reshape(shape_before_flattening[1:])(x)
x = layers.Conv2DTranspose(128, 3, strides=2, padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)
x = layers.Dropout(0.5)(x)
x = layers.Conv2DTranspose(128, 3, strides=2, padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)
x = layers.Dropout(0.5)(x)
x = layers.Conv2DTranspose(128, 3, strides=2, padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)
x = layers.Dropout(0.5)(x)
x = layers.Conv2DTranspose(64, 3, strides=2, padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)
x = layers.Dropout(0.5)(x)
x = layers.Conv2DTranspose(1, 3,padding='same',activation='sigmoid',strides=(2, 2))(x)
decoder = Model(decoder_input, x)
decoder.summary()
z_decoded = decoder(z)


#P254代码清单8-26。
class CustomVariationalLayer(tensorflow.keras.layers.Layer):
    def vae_loss(self, x, z_decoded):
        x = K.flatten(x) #K.flatten不保留批量。layers.Flatten()保留批量
        z_decoded = K.flatten(z_decoded)
        xent_loss = tensorflow.keras.metrics.binary_crossentropy(x, z_decoded) #metrics.binary_crossentropy与losses.binary_crossentropy结果一模一样.与第2版原理相同，但是具体数值是完全不同的
        kl_loss = -kl_loss_coefficient * 5e-4 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)) #删除axis=-1。axis=-1保留批量，删除它则不保留。因为xent_loss不保留批量，故这里应该不要它。这个不影响最终结果
        #xent_loss=K.print_tensor(xent_loss, message='    xent_loss = ')#可以输出
        #kl_loss=K.print_tensor(kl_loss, message='    kl_loss = ')#可以输出
        return K.mean(xent_loss + kl_loss)
    def call(self, inputs):
        x = inputs[0]
        z_decoded = inputs[1]
        loss = self.vae_loss(x, z_decoded)
        self.add_loss(loss)
        return x
y = CustomVariationalLayer()([input_img, z_decoded])


#P255代码清单8-27。
vae = Model(input_img, y)
vae.compile(optimizer='rmsprop', loss=None)
vae.summary()


#加载数据集，P108
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
                                                    train_dir, #各个子目录自动打乱的，某个子目录中文件也是打乱的
                                                    target_size=(img_shape[0], img_shape[1]),
                                                    color_mode="grayscale",
                                                    batch_size=batch_size,
                                                    class_mode=None
                                                    )

#for data_batch in train_generator: #查看图片载入后，是否值0～1之间，是否乱序加载
#    print('data batch shape:', data_batch.shape)
#    #import matplotlib.pyplot as plt
#    #plt.imshow(data_batch[0], cmap=plt.cm.binary)
#    #plt.show()
#    #plt.imshow(data_batch[1], cmap=plt.cm.binary)
#    #plt.show()
#    #plt.imshow(data_batch[2], cmap=plt.cm.binary)
#    #plt.show()
#    #fileWrite01=open('out.txt','w')
#    #fileWrite01.write(str(data_batch[0][64]))
#    #fileWrite01.write(str(data_batch[0][96]))
#    #fileWrite01.write(str(data_batch[0][128]))
#    #fileWrite01.write(str(data_batch[0][160]))
#    #fileWrite01.write(str(data_batch[0][192]))
#    #fileWrite01.close
#    break

#因为模型复杂，无法使用回调函数每几轮存储h5文件
history=vae.fit_generator(train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs)
#vae.save('vae_1nd Edition04.h5')
decoder.save(os.path.join(save_fileName + '_decoder.h5'))  #可以输出
encoder.save(os.path.join(save_fileName + '_encoder.h5')) #可以输出

with open(os.path.join(save_fileName + '.txt'), 'w') as file_object:#有了这个，可以随时绘制训练图像
    file_object.write(str(history.history))

endTime=time.time()
print('How many seconds:',(endTime-startTime))  #epochs=10时，How many seconds: 241.50461506843567


import matplotlib.pyplot as plt  #P57代码清单3-9
history_dict = history.history
total_loss_values = history_dict['loss']
epochs = range(1, len(total_loss_values) + 1)
plt.plot(epochs, total_loss_values, 'bo', label='total_loss_values',markersize=2)
plt.title('total_loss,How many seconds:%d'%(endTime-startTime))
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(save_fileName + '.png'), dpi=300)
plt.show()

