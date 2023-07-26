import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras.layers import InputSpec


class PConv2D(tf.keras.layers.Conv2D):
    def __init__(self, *args, **kwargs):
        super(PConv2D, self).__init__(*args, **kwargs)
        self.input_spec = [InputSpec(ndim=4), InputSpec(ndim=4)]

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1

        self.input_dim = input_shape[0][channel_axis]

        self.kernel_shape = self.kernel_size + (self.input_dim, self.filters)

        self.kernel = self.add_weight(shape=self.kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='image_kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint,
                                      trainable=True)

        self.window_size = self.kernel_size[0] * self.kernel_size[1]

        if self.use_bias:
            self.bias = self.add_weight(shape=self.filters,
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint,
                                        trainable=True)
        else:
            self.bias = None

    def call(self, inputs):
        mask_output = K.backend.conv2d(
            inputs[1], tf.ones(self.kernel_shape),
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate
        )

        img_output = K.backend.conv2d(
            inputs[0] * inputs[1], self.kernel,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate
        )

        mask_ratio = self.window_size / (mask_output + 1e-8)
        mask_output = K.backend.clip(mask_output, 0.0, 1.0)
        mask_ratio *= mask_output

        img_output = img_output * mask_ratio

        if self.use_bias:
            img_output = K.backend.bias_add(
                img_output, self.bias, data_format=self.data_format)

        if self.activation is not None:
            img_output = self.activation(img_output)

        return [img_output, mask_output]
