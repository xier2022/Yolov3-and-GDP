import math

import tensorflow as tf

# arc face loss calculation
class ArcFace(tf.keras.layers.Layer):

    def __init__(self, n_classes=10, s=32.0, m=0.50, regularizer=None, **kwargs):
        super(ArcFace, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.regularizer = tf.keras.regularizers.get(regularizer)

    def build(self, input_shape):
        super(ArcFace, self).build(input_shape[0])
        self.W = self.add_weight(name='W',
                                shape=(input_shape[0][-1], self.n_classes),
                                initializer='glorot_uniform',
                                trainable=True,
                                regularizer=self.regularizer)

    def call(self, inputs, **kwargs):
        x, y = inputs
        c = tf.shape(x)[-1]
        # normalize feature
        x = tf.math.l2_normalize(x, axis=1)
        # normalize weights
        W = tf.math.l2_normalize(self.W, axis=0)
        # dot product
        logits = x @ W
        # add margin
        # clip logits to prevent zero division when backward
        theta = tf.acos(tf.clip_by_value(logits, -1.0 + tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon()))
        target_logits = tf.cos(theta + self.m)
        # sin = tf.sqrt(1 - logits**2)
        # cos_m = tf.cos(logits)
        # sin_m = tf.sin(logits)
        # target_logits = logits * cos_m - sin * sin_m
        #
        logits = logits * (1 - y) + target_logits * y
        # feature re-scale
        logits *= self.s
        out = tf.nn.softmax(logits)

        return out

    def compute_output_shape(self, input_shape):
        return (None, self.n_classes)

    def get_config(self):
        config = {"n_classes": self.n_classes,
                  "s": self.s,
                  "m":self.m
                  }
        base_config = super(ArcFace, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ArcFace_v2(tf.keras.layers.Layer):
    '''
    Arguments:
        inputs: the input embedding vectors
        n_classes: number of classes
        s: scaler value (default as 64)
        m: the margin value (default as 0.5)
    Returns:
        the final calculated outputs
    '''

    def __init__(self, n_classes, s=32., m=0.5, **kwargs):
        self.init = tf.keras.initializers.get('glorot_uniform')  # Xavier uniform intializer
        self.n_classes = n_classes
        self.s = s
        self.m = m
        super(ArcFace_v2, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape[0]) == 2 and len(input_shape[1]) == 2
        self.W = self.add_weight(name='{}_W'.format(self.name),
                                 shape=(input_shape[0][-1], self.n_classes),
                                 initializer=self.init)
        super(ArcFace_v2, self).build(input_shape[0])

    def call(self, inputs, **kwargs):
        cos_m = math.cos(self.m)
        sin_m = math.sin(self.m)
        mm = sin_m * self.m
        threshold = math.cos(math.pi - self.m)

        # features
        #X = inputs[0]
        # 1-D or one-hot label works as mask
        #Y_mask = inputs[1]
        X, Y_mask = inputs
        # If Y_mask is not in one-hot form, transfer it to one-hot form.
        #if tf.shape(Y_mask)[-1] == 1:
        #    Y_mask = tf.cast(Y_mask, tf.int32)
        #    Y_mask = tf.reshape(tf.one_hot(Y_mask, self.n_classes), (-1, self.n_classes))

        X_normed = tf.math.l2_normalize(X, axis=1)  # L2 Normalized X
        W = tf.math.l2_normalize(self.W, axis=0)  # L2 Normalized Weights

        # cos(theta + m)
        cos_theta = tf.keras.backend.dot(X_normed, W)  # æÿ’Û≥À∑®
        cos_theta2 = tf.square(cos_theta)
        sin_theta2 = 1. - cos_theta2
        sin_theta = tf.sqrt(sin_theta2 + tf.keras.backend.epsilon())
        cos_tm = self.s * ((cos_theta * cos_m) - (sin_theta * sin_m))

        # This condition controls the theta + m should in range [0, pi]
        #   0 <= theta + m < = pi
        #   -m <= theta <= pi - m
        cond_v = cos_theta - threshold
        cond = tf.cast(tf.nn.relu(cond_v), dtype=tf.bool)
        keep_val = self.s * (cos_theta - mm)
        cos_tm_temp = tf.where(cond, cos_tm, keep_val)

        # mask by label
        # Y_mask =+ K.epsilon()
        inv_mask = 1. - Y_mask
        s_cos_theta = self.s * cos_theta

        output = tf.nn.softmax((s_cos_theta * inv_mask) + (cos_tm_temp * Y_mask))

        return output

    def get_config(self):
        config = {"n_classes": self.n_classes,
                  "s": self.s,
                  "m":self.m
                  }
        base_config = super(ArcFace_v2, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.n_classes
