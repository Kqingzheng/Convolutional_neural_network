from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras import models
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
import matplotlib.pyplot as plt
from tensorboard.plugins.hparams import keras
from tensorflow import optimizers


def model():
    model = models.Sequential()
    # 第一层卷积网络，使用96个卷积核，大小为11x11步长为4， 要求输入的图片为227x227， 3个通道，不加边，激活函数使用relu
    model.add(Conv2D(96, (11, 11), strides=(4, 4), input_shape=(227, 227, 3),  activation='relu',
                     kernel_initializer='uniform'))
    # 池化层
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    # 第二层加边使用256个5x5的卷积核，加边，激活函数为relu
    model.add(Conv2D(256, (5, 5), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    # 使用池化层，步长为2
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    # 第三层卷积，大小为3x3的卷积核使用384个
    model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    # 第四层卷积,同第三层
    model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    # 第五层卷积使用的卷积核为256个，其他同上
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    return model


#  训练数据增强
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   #  表示图像随机旋转的角度范围
                                   rotation_range=40,
                                   #  图像在水平方向上平移的范围
                                   width_shift_range=0.2,
                                   #  图像在垂直方向上平移的范围
                                   height_shift_range=0.2,
                                   #  随机错切变换的角度
                                   shear_range=0.2,
                                   #  图像随机缩放的范围
                                   zoom_range=0.2,
                                   #  随机将一半图像水平翻转
                                   horizontal_flip=True)
#  验证数据不能增强
validation_datagen = ImageDataGenerator(rescale=1. / 255)

train_dir = r"./cats_and_dogs_small/train/"
validation_dir = r"./cats_and_dogs_small/validation/"

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    #  将所有图像的大小调整为227*227
                                                    target_size=(227, 227),
                                                    #  批量大小
                                                    batch_size=16,
                                                    class_mode='binary')
validation_generator = validation_datagen.flow_from_directory(validation_dir,
                                                              target_size=(227, 227),
                                                              batch_size=16,
                                                              class_mode='binary')



#  初始化模型
model = model()
#  用于配置训练模型（优化器、目标函数、模型评估标准）

model.compile(optimizer=optimizers.Adam(lr=1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#  查看各个层的信息
model.summary()
#  回调函数，在每个训练期之后保存模型
model_checkpoint = ModelCheckpoint('model_AlexNet_adam.hdf5',  # 保存模型的路径
                                   monitor='loss',  # 被监测的数据
                                   verbose=1,  # 日志显示模式:0=>安静模式,1=>进度条,2=>每轮一行
                                   save_best_only=True)  # 若为True,最佳模型就不会被覆盖
#  用history接收返回值用于画loss/acc曲线
history = model.fit(train_generator,
                              steps_per_epoch=313,
                              epochs=200,
                              callbacks=[model_checkpoint],
                              validation_data=validation_generator,
                              validation_steps=50)

# print(history.history)
#
# print(history.epoch)
#
# print(history.history['val_loss'])
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()