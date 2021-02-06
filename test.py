import os
import numpy as np
import pandas as pd
import tensorflow as tf
from train import MyModel, Model2

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


def load_test_data_name(filefolder):
    data = np.load(os.path.abspath(filefolder + '/names_onehots.npy'), allow_pickle=True).item()
    onehots = data['onehots']
    name = data['names']
    return onehots, name


def test1(test_data):
    # model
    tf.reset_default_graph()  #
    model = MyModel()
    input_place_holder = tf.placeholder(tf.float32, [None] + list(test_data.shape[1:]), name='input')
    input_place_holder_reshaped = tf.reshape(input_place_holder, [-1] + list(test_data.shape[1:]) + [1])
    output, output_with_sm = model(input_place_holder_reshaped)

    BatchSize = 128

    # Predict on the test set
    data_size = test_data.shape[0]
    with tf.Session() as sess:
        # restore model
        saver = tf.train.Saver()
        saver.restore(sess, os.path.abspath('weights/add_1/model'))

        # predict
        prediction = []
        for i in range(0, data_size, BatchSize):
            test_output = sess.run(output, {input_place_holder: test_data[i:i + BatchSize]})
            test_output_with_sm = sess.run(output_with_sm, {input_place_holder: test_data[i:i + BatchSize]})
            pred = test_output_with_sm[:, 1]
            prediction.extend(list(pred))

    sess.close()
    return prediction


def test2(test_data):
    # data
    test_path = "../test/"
    test_data, test_name = load_test_data_name(test_path)
    name = test_name

    # model
    tf.reset_default_graph()  #
    model = Model2()
    input_place_holder = tf.placeholder(tf.float32, [None] + list(test_data.shape[1:]), name='input')
    input_place_holder_reshaped = tf.reshape(input_place_holder, [-1] + list(test_data.shape[1:]) + [1])
    output, output_with_sm = model(input_place_holder_reshaped)

    BatchSize = 128

    # Predict on the test set
    data_size = test_data.shape[0]
    with tf.Session() as sess:
        # restore model
        saver = tf.train.Saver()
        saver.restore(sess, os.path.abspath('weights/add_2/model'))

        # predict
        prediction = []
        for i in range(0, data_size, BatchSize):
            test_output = sess.run(output, {input_place_holder: test_data[i:i + BatchSize]})
            test_output_with_sm = sess.run(output_with_sm, {input_place_holder: test_data[i:i + BatchSize]})
            pred = test_output_with_sm[:, 1]
            prediction.extend(list(pred))

    sess.close()
    return prediction


# data
test_path = "../test/"
test_data, test_name = load_test_data_name(test_path)
name = test_name

pred1 = test1(test_data)
print('model 1 predicted.')
pred2 = test2(test_data)
print('model 2 predicted.')


prediction = []
for i in range(len(pred1)):
    prediction.append((pred1[i] + pred2[i]) / 2)

# write into file
f = open('output_518021911194.txt', 'w')
f.write('Chemical,Label\n')
for i, v in enumerate(prediction):
    f.write(name[i] + ',%f\n' % v)
f.close()

print('predict over')
