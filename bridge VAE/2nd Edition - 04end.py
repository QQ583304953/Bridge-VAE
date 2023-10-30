import time
startTime=time.time()

#《python深度学习（第2版）》P352
from tensorflow import keras
from tensorflow.keras import layers
import os
import numpy as np

#可变参数
latent_dim = 8
kl_loss_coefficient=10间

#固定参数
epochs=50
img_shape = (128, 512, 1) #对于1920*1080电影截屏图片，第一个1920是列（图片水平向像素数量）、第二个1080是行（图片竖向像素数量）。所以输入神经网络时，input_shape=(720 * 1980,)
train_dir='.\\png 512x128_GRAY' 
save_fileName='2nd Edition - %dD 04_epochs=%d_kl_loss_coefficient=%.4f'%(latent_dim,epochs,kl_loss_coefficient) 
batch_size = 32 #取64则出错.增加批量规范化、Dropout[必须减少batch_size数值，否则出错]


#统计文件夹中的文件总个数（含所有的子目录下的）
def count_files_in_folder(folder_path): 
    count = 0
    for _, _, files in os.walk(folder_path):
        count += len(files)
    return count
file_count = count_files_in_folder(train_dir)
steps_per_epoch=file_count//batch_size


#神经网络
encoder_inputs = keras.Input(shape=img_shape)
x = layers.Conv2D(64, 3, strides=2, padding="same")(encoder_inputs)
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
x = layers.Flatten()(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var], name="encoder")
encoder.summary()

#P353
import tensorflow as tf
class Sampler(layers.Layer):
    def call(self, z_mean, z_log_var):
        batch_size = tf.shape(z_mean)[0]
        z_size = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch_size, z_size))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

#P354
latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(4 * 16 * 128, activation="relu")(latent_inputs)
x = layers.Reshape((4, 16, 128))(x)
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
decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", strides=2, padding="same")(x)#此为最后一层，故sigmoid激活
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()

#P354
class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.sampler = Sampler()
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    def call(self, data, training=False): #增加call代码原因见“带有数据生成器的 keras VAE故障解决.pdf”
        # your custom code when you call the model
        # or just pass, you don't need this method
        # for training
        pass

    @property
    def metrics(self):
        return [self.total_loss_tracker,
                self.reconstruction_loss_tracker,
                self.kl_loss_tracker]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var = self.encoder(data)
            z = self.sampler(z_mean, z_log_var)
            reconstruction = decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), #metrics.binary_crossentropy与losses.binary_crossentropy结果一模一样
                    axis=(1, 2)
                )
            )
            kl_loss = -kl_loss_coefficient * 0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)) #第一版是0.0005，第二版是0.5。这个参数需要优化。经过测试，KL loss系数随着batch_size同比例增加才行
            total_loss = reconstruction_loss + tf.reduce_mean(kl_loss)
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "total_loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


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

vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam(), run_eagerly=True) #Adam()的学习率取0.01效果不好，默认最佳
history=vae.fit_generator(train_generator,#因为模型复杂，无法使用回调函数每几轮存储h5文件
        steps_per_epoch=steps_per_epoch,
        epochs=epochs)

#vae.save('vae_2nd Edition01.h5') #出错
decoder.save(os.path.join(save_fileName + '_decoder.h5'))  #可以输出
encoder.save(os.path.join(save_fileName + '_encoder.h5')) #可以输出

with open(os.path.join(save_fileName + '.txt'), 'w') as file_object:#有了这个，可以随时绘制训练图像
    file_object.write(str(history.history))

endTime=time.time()
print('How many seconds:',(endTime-startTime))  
#epochs=50、batch_size=64时，How many seconds:6367，total_loss: 4855.4087 - reconstruction_loss: 4847.2026 - kl_loss: 8.2086

import matplotlib.pyplot as plt  #P57代码清单3-9
history_dict = history.history
total_loss_values = history_dict['total_loss']
reconstruction_loss_values = history_dict['reconstruction_loss']
kl_loss_values = history_dict['kl_loss']
epochs = range(1, len(total_loss_values) + 1)
plt.plot(epochs, total_loss_values, 'bo', label='total_loss_values',markersize=2)
plt.plot(epochs, reconstruction_loss_values, 'b', label='reconstruction_loss_values')
plt.plot(epochs, kl_loss_values, 'b:', label='kl_loss_values')
plt.title('total_loss , reconstruction_loss and kl_loss,How many seconds:%d'%(endTime-startTime))
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(save_fileName + '.png'), dpi=300)
plt.show()




