import os

import tensorflow as tf
import tensorflow.keras as K
import tensorflow.keras.layers as L
from matplotlib import pyplot as plt

from layers import PConv2D


class PConvUnet():
    def __init__(self, vgg_path="", image_size=512, batch_size=4, lr=2e-4, m=64):
        self.image_size = image_size
        self.batch_size = batch_size
        self.lr = lr
        self.ema = 0.999
        self.optimizer = tf.optimizers.Adam(lr)
        self.losses = {}

        self.pconv_unet = self.pconv_unet(m=m)
        self.pconv_unet.build(input_shape=(None, image_size, image_size, 3))

        self.vgg = self.build_vgg(vgg_path=vgg_path)
        if self.vgg:
            self.vgg.build(input_shape=(None, image_size, image_size, 3))

    def pconv_unet(self, m=64):
        kernel = 3
        stride = 2

        c = [1, 2, 4, 8, 16, 16, 16, 16, 16, 16, 16, 8, 4, 2, 1]
        filters = [i * m for i in c] + [3]

        l_in = L.Input(shape=(self.image_size, self.image_size,
                       3), name="gen_input_image")
        m_in = L.Input(shape=(self.image_size, self.image_size,
                       3), name="gen_input_mask")

        ls, ms = [], []

        l, m = PConv2D(filters[0], 7, stride,
                       activation='relu', padding='same')([l_in, m_in])
        ls.append(l)
        ms.append(m)

        for i in range(7):
            if i < 2:
                k = 5
            else:
                k = kernel
            l, m = PConv2D(filters[i + 1], k, stride,
                           activation='relu', padding='same')([l, m])
            l = L.BatchNormalization()(l)
            ls.append(l)
            ms.append(m)

        ms = ms[::-1]
        ls = ls[::-1]

        for i in range(7):
            l = L.UpSampling2D(size=2, interpolation='nearest')(l)
            l = L.Concatenate()([l, ls[i + 1]])
            m = L.UpSampling2D(size=2, interpolation='nearest')(m)
            m = L.Concatenate()([m, ms[i + 1]])
            l, m = PConv2D(filters[i + 8], kernel, padding='same')([l, m])
            l = L.LeakyReLU(alpha=0.2)(l)
            l = L.BatchNormalization()(l)

        l = L.UpSampling2D(size=2, interpolation='nearest')(l)
        l = L.Concatenate()([l, l_in])
        m = L.UpSampling2D(size=2, interpolation='nearest')(m)
        m = L.Concatenate()([m, m_in])
        l, m = PConv2D(filters[15], kernel,
                       padding='same', activation='relu')([l, m])
        l = L.Conv2D(filters[15], kernel_size=1, strides=1,
                     activation='sigmoid', name='output_image')(l)

        return K.Model(inputs=[l_in, m_in], outputs=l, name="pconv_unet")

    def build_vgg(self, vgg_path=""):
        if not vgg_path:
            return None
        mean = [0.485, 0.456, 0.406]
        stdev = [0.229, 0.224, 0.225]

        inputs = L.Input(shape=(self.image_size, self.image_size, 3))
        processed = L.Lambda(lambda x: (x - mean) / stdev)(inputs)

        vgg = K.applications.VGG16(
            weights=None, include_top=False, input_tensor=processed)
        vgg.load_weights(vgg_path, by_name=True)

        vgg.outputs = [vgg.layers[i].output for i in [4, 7, 11]]
        model = K.Model(inputs=inputs, outputs=vgg.outputs)
        model.trainable = False
        model.compile(loss='mse', optimizer='adam')

        return model

    def gen_loss(self, vgg_output_real, vgg_output_gen, vgg_output_comp, target,
                 generated, mask, weights=[1, 6, 0.05, 120.0, 120.0, 0.1]):

        comp = target * mask + (1 - mask) * generated

        loss = 0
        d = {}

        l = tf.reduce_mean(
            tf.abs(target * mask - generated * mask)) * weights[0]
        d['valid'] = l
        loss += l

        l = tf.reduce_mean(tf.abs(target * (1 - mask) -
                           generated * (1 - mask))) * weights[1]
        d['hole'] = l
        loss += l

        for p in range(len(vgg_output_real)):
            l = tf.reduce_mean(tf.math.abs(
                vgg_output_real[p] - vgg_output_gen[p])) * weights[2]
            l += tf.reduce_mean(tf.math.abs(
                vgg_output_comp[p] - vgg_output_gen[p])) * weights[2]
            d['perceprual_' + str(p)] = l
            loss += l

        for p in range(len(vgg_output_real)):
            b, w, h, c = vgg_output_real[p].shape
            r = tf.reshape(vgg_output_real[p], [b, w * h, c])
            f = tf.reshape(vgg_output_gen[p], [b, w * h, c])
            k = tf.reshape(vgg_output_comp[p], [b, w * h, c])

            r = tf.keras.backend.batch_dot(r, r, axes=[1, 1])
            f = tf.keras.backend.batch_dot(f, f, axes=[1, 1])
            k = tf.keras.backend.batch_dot(k, k, axes=[1, 1])

            l = tf.reduce_sum(tf.math.abs(r - f) / c**3 / h / w) * weights[3]
            d['style_gen_' + str(p)] = l
            loss += l
            l = tf.reduce_sum(tf.math.abs(r - k) / c**3 / h / w) * weights[4]
            d['style_comp_' + str(p)] = l
            loss += l

        kernel = K.backend.ones(shape=(3, 3, mask.shape[3], mask.shape[3]))
        dilated_mask = K.backend.conv2d(
            1 - mask, kernel, data_format='channels_last', padding='same')
        dilated_mask = K.backend.cast(
            K.backend.greater(dilated_mask, 0), 'float32')
        TV = dilated_mask * comp

        l = tf.reduce_mean(
            tf.abs(TV[:, 1:, :, :] - TV[:, :-1, :, :])) * weights[5]
        l += tf.reduce_mean(tf.abs(TV[:, :, 1:, :] -
                            TV[:, :, :-1, :])) * weights[5]

        d['tv'] = l
        loss += l

        d['total'] = loss

        return loss, d

    def load_model(self, fname):
        self.pconv_unet.load_weights(fname)

    def save_model(self, fname):
        self.pconv_unet.save_weights(fname)
