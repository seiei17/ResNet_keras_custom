# data preprocessing files


import numpy as np
import pickle
import os
import cv2
from keras.utils import to_categorical
import math


class Cifar10Gen(object):
    # using cifar 10 dataset
    # have 5 batch, which form is 'data_batch_i'
    # batches.meta consisting of class name

    def __init__(self, path, batch_size=256):
        self.path = path
        self.num = 0
        self.batch_size = batch_size

    def load_file(self, filename):
        with open(filename, 'rb') as file:
            data = pickle.load(file, encoding='latin1')
        return data

    def train_generator(self):
        while True:
            for i in range(5):
                data = self.load_file(os.path.join(self.path,
                                                   'data_batch_{}'.format(i + 1)))
                labels = data['labels']
                images = data['data']
                steps = math.ceil(10000 / self.batch_size)
                for i in range(steps):
                    if i == steps - 1:
                        label = labels[i * self.batch_size:]
                        image = images[i * self.batch_size:][:]
                    else:
                        label = labels[i * self.batch_size: (i + 1) * self.batch_size]
                        image = images[i * self.batch_size: (i + 1) * self.batch_size][:]
                    image = image.reshape(len(label), 32, 32, 3)
                    mean = np.mean(image, axis=0)

                    new_image = np.zeros((len(label), 224, 224, 3))
                    for j in range(len(label)):
                        image[j, :, :, :] = image[j, :, :, :] - mean
                        new_image[j, :, :, :] = cv2.resize(src=image[j, :, :, :],
                                                           dsize=(224, 224),
                                                           dst=new_image[j, :, :, :],
                                                           interpolation=cv2.INTER_LINEAR)
                    new_image = np.divide(new_image, 128)
                    label = to_categorical(label, num_classes=10)
                    yield new_image, label

    def valid_generator(self):
        while True:
            data = self.load_file(os.path.join(self.path, 'test_batch'))
            labels = data['labels']
            images = data['data']
            steps = math.ceil(10000 / self.batch_size)
            for i in range(steps):
                if i == steps - 1:
                    label = labels[i * self.batch_size:]
                    image = images[i * self.batch_size:][:]
                else:
                    label = labels[i * self.batch_size: (i + 1) * self.batch_size]
                    image = images[i * self.batch_size: (i + 1) * self.batch_size][:]
                image = image.reshape(len(label), 32, 32, 3)
                mean = np.mean(image, axis=0)

                new_image = np.zeros((len(label), 224, 224, 3))
                for j in range(len(label)):
                    image[j, :, :, :] = image[j, :, :, :] - mean
                    new_image[j, :, :, :] = cv2.resize(src=image[j, :, :, :],
                                                       dsize=(224, 224),
                                                       dst=new_image[j, :, :, :],
                                                       interpolation=cv2.INTER_LINEAR)
                new_image = np.divide(new_image, 128)
                label = to_categorical(label, num_classes=10)
                yield new_image, label

    def valid_data(self):
        data = self.load_file(os.path.join(self.path, 'test_batch'))
        labels = data['labels']
        images = data['data']
        images = images.reshape((len(labels), 32, 32, 3))
        images = np.divide(images, 255)
        new_images = np.zeros((len(labels), 224, 224, 3))
        for j in range(len(labels)):
            new_images[j] = cv2.resize(images[j], (224, 224))
        labels = to_categorical(labels, 10)
        return new_images, labels


class Cifar100Gen(object):
    # using cifar 100 dataset
    # have 5 batch, which form is 'data_batch_i'
    # batches.meta consisting of class name

    def __init__(self, path, batch_size=256):
        self.path = path
        self.num = 0
        self.batch_size = batch_size

    def load_file(self, filename):
        with open(filename, 'rb') as file:
            data = pickle.load(file, encoding='latin1')
        return data

    def train_generator(self):
        while True:
            data = self.load_file(os.path.join(self.path, 'train'))
            labels = data['fine_labels']
            images = data['data']
            steps = math.ceil(50000 / self.batch_size)
            for i in range(steps):
                if i == steps - 1:
                    label = labels[i * self.batch_size:]
                    image = images[i * self.batch_size:][:]
                else:
                    label = labels[i * self.batch_size: (i + 1) * self.batch_size]
                    image = images[i * self.batch_size: (i + 1) * self.batch_size][:]
                image = image.reshape(len(label), 32, 32, 3)
                image = np.divide(image, 255)
                new_image = np.zeros((len(label), 224, 224, 3))
                for j in range(len(label)):
                    new_image[j, :, :, :] = cv2.resize(src=image[j, :, :, :],
                                                       dsize=(224, 224),
                                                       dst=new_image[j, :, :, :],
                                                       interpolation=cv2.INTER_LINEAR)
                label = to_categorical(label, num_classes=100)
                yield new_image, label

    def valid_generator(self):
        while True:
            data = self.load_file(os.path.join(self.path, 'test'))
            labels = data['fine_labels']
            images = data['data']
            steps = math.ceil(10000 / self.batch_size)
            for i in range(steps):
                if i == steps - 1:
                    label = labels[i * self.batch_size:]
                    image = images[i * self.batch_size:][:]
                else:
                    label = labels[i * self.batch_size: (i + 1) * self.batch_size]
                    image = images[i * self.batch_size: (i + 1) * self.batch_size][:]
                image = image.reshape(len(label), 32, 32, 3)
                image = np.divide(image, 255)
                new_image = np.zeros((len(label), 224, 224, 3))
                for j in range(len(label)):
                    new_image[j, :, :, :] = cv2.resize(src=image[j, :, :, :],
                                                       dsize=(224, 224),
                                                       dst=new_image[j, :, :, :],
                                                       interpolation=cv2.INTER_LINEAR)
                label = to_categorical(label, num_classes=100)
                yield new_image, label

    def valid_data(self):
        data = self.load_file(os.path.join(self.path, 'test_batch'))
        labels = data['labels']
        images = data['data']
        images = images.reshape((len(labels), 32, 32, 3))
        images = np.divide(images, 255)
        new_images = np.zeros((len(labels), 224, 224, 3))
        for j in range(len(labels)):
            new_images[j] = cv2.resize(images[j], (224, 224))
        labels = to_categorical(labels, 100)
        return new_images, labels
