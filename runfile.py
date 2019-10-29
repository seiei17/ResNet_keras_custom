# runfile
import keras
import keras.backend as K
import tensorflow
import math
import os

from data_generator import Cifar10Gen
from ResNetModel import resnet

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

data_name = 'cifar10'
path = '../database/{}/'.format(data_name)
num_classes = 10

batch_size = 256
steps = math.ceil(10000 / batch_size)
epochs = 50

weight_decay = 0.0001
momentum = 0.9
lr = 0.1

gen = Cifar10Gen(path, batch_size)
model = resnet((224, 224, 3,), num_classes, weight_decay)

optimizer = keras.optimizers.Adam(lr)
callback = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1,
                                             patience=5, verbose=1)

model.compile(optimizer=optimizer, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

model.fit_generator(gen.train_generator(),
                    steps_per_epoch=5*steps,
                    epochs=epochs,
                    verbose=1,
                    callbacks=[callback],
                    validation_data=gen.valid_generator(),
                    validation_steps=steps)
keras.models.save_model(model, './checkpoint/ResNet_{}_{}.h5'.format(data_name, epochs))
