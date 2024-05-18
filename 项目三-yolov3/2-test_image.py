import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


cls_num = 10572


def preprocess(img):
    img = (img.astype('float32') - 127.5) / 128.0
    img = np.expand_dims(img, axis=0)
    return img


# load image
img_yzy = preprocess(plt.imread("data/face_dataset/yzy/yzy.jpg"))
img_lm = preprocess(plt.imread("data/face_dataset/lm/lm.jpg"))
img_zt = preprocess(plt.imread("data/face_dataset/zt/zt.jpg"))

img_test = preprocess(plt.imread("data/test0.jpg"))


if __name__ == '__main__':

    # feed forward
    model = tf.keras.models.load_model("model_data/mobilefacenet_model.h5")

    embedding_yzy = model.predict(img_yzy)
    embedding_lm = model.predict(img_lm)
    embedding_zt = model.predict(img_zt)

    embedding_test = model.predict(img_test)

    # test result
    embedding_yzy = embedding_yzy / np.expand_dims(np.sqrt(np.sum(np.power(embedding_yzy, 2), 1)), 1)
    embedding_lm = embedding_lm / np.expand_dims(np.sqrt(np.sum(np.power(embedding_lm, 2), 1)), 1)
    embedding_zt = embedding_zt / np.expand_dims(np.sqrt(np.sum(np.power(embedding_zt, 2), 1)), 1)
    embedding_test = embedding_test / np.expand_dims(np.sqrt(np.sum(np.power(embedding_test, 2), 1)), 1)

    # get result
    print(np.sum(np.multiply(embedding_yzy, embedding_test), 1))
    print(np.sum(np.multiply(embedding_lm, embedding_test), 1))
    print(np.sum(np.multiply(embedding_zt, embedding_test), 1))

    # # save database
    # db = np.concatenate((embedding_yzy, embedding_lm, embedding_steve), axis=0)
    # print(db.shape)
    # np.save("pretrained_model/db", db)
