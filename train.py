import os
import numpy as np
import pandas as pd
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


def load_data(filefolder):
    data = np.load(os.path.abspath(filefolder + '/names_onehots.npy'), allow_pickle=True).item()
    data = data['onehots']
    label = pd.read_csv(os.path.abspath(filefolder + '/names_labels.txt'), sep=',')
    label = label['Label'].values
    return data, label


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1_layer = tf.keras.layers.Conv2D(64, 5, 1, 'same', activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.1))
        self.pool1_layer = tf.keras.layers.MaxPool2D(2, 2)

        self.conv2_layer = tf.keras.layers.Conv2D(128, 3, (1, 2), 'same', activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.1))
        self.pool2_layer = tf.keras.layers.MaxPool2D(2, 2)

        self.conv3_layer = tf.keras.layers.Conv2D(256, 3, (1, 2), 'same', activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.1))
        self.pool3_layer = tf.keras.layers.MaxPool2D(2, 2)

        # flat
        self.flat = tf.keras.layers.Flatten()
        self.FCN1 = tf.keras.layers.Dense(32, activation=tf.nn.relu)
        self.FCN2 = tf.keras.layers.Dense(2)
        # softmax

    def call(self, inputs):
        x = self.conv1_layer(inputs)
        x = self.pool1_layer(x)

        x = self.conv2_layer(x)
        x = self.pool2_layer(x)

        x = self.conv3_layer(x)
        x = self.pool3_layer(x)

        x = self.flat(x)
        x = self.FCN1(x)
        output = self.FCN2(x)
        output_with_sm = tf.nn.softmax(output)
        return output, output_with_sm


class Model2(tf.keras.Model):
    def __init__(self):
        super(Model2, self).__init__()
        self.conv1_layer = tf.keras.layers.Conv2D(32, 5, 1, 'same', activation=tf.nn.relu)
        self.pool1_layer = tf.keras.layers.MaxPool2D(2, 2)
        self.conv2_layer = tf.keras.layers.Conv2D(32, 3, (1, 2), 'same', activation=tf.nn.relu)
        self.pool2_layer = tf.keras.layers.MaxPool2D(2, 2)
        # flat
        self.FCN = tf.keras.layers.Dense(2)
        # softmax

    def call(self, inputs):
        x = self.conv1_layer(inputs)
        x = self.pool1_layer(x)
        x = self.conv2_layer(x)
        x = self.pool2_layer(x)
        flat = tf.reshape(x, [-1, 18 * 50 * 32])
        output = self.FCN(flat)
        output_with_sm = tf.nn.softmax(output)
        return output, output_with_sm


def train1():
    # parameters
    LR = 0.0001
    BatchSize = 128
    EPOCH = 30

    train_data_path = "../train/"
    validation_data_path = "../validation/"
    # data
    train_x, train_y = load_data(train_data_path)
    valid_x, valid_y = load_data(validation_data_path)

    # model & input and output of model
    tf.reset_default_graph()
    model = MyModel()

    onehots_shape = list(train_x.shape[1:])
    input_place_holder = tf.placeholder(tf.float32, [None] + onehots_shape, name='input')
    input_place_holder_reshaped = tf.reshape(input_place_holder, [-1] + onehots_shape + [1])
    label_place_holder = tf.placeholder(tf.int32, [None], name='label')
    label_place_holder_2d = tf.one_hot(label_place_holder, 2)
    output, output_with_sm = model(input_place_holder_reshaped)
    model.summary()  # show model's structure

    # loss
    bce = tf.keras.losses.BinaryCrossentropy()  # compute cost
    loss = bce(label_place_holder_2d, output_with_sm)

    # Optimizer
    train_op = tf.train.AdamOptimizer(LR).minimize(loss)

    # auc
    prediction_place_holder = tf.placeholder(tf.float64, [None], name='pred')
    auc, update_op = tf.metrics.auc(labels=label_place_holder, predictions=prediction_place_holder)

    # run
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init_op)

        saver = tf.train.Saver()

        train_size = train_x.shape[0]
        best_val_auc = 0
        for epoch in range(EPOCH):
            for i in range(0, train_size, BatchSize):
                b_x, b_y = train_x[i:i + BatchSize], train_y[i:i + BatchSize]
                _, loss_ = sess.run([train_op, loss], {'input:0': b_x, 'label:0': b_y})

                # print("Epoch {}: [{}/{}], training set loss: {:.4}".format(epoch, i, train_size, loss_))

            if epoch % 1 == 0:
                val_prediction = sess.run(output_with_sm, {'input:0': valid_x})
                val_prediction = val_prediction[:, 1]
                auc_value = sess.run(update_op,
                                     feed_dict={prediction_place_holder: val_prediction, label_place_holder: valid_y})

                if auc_value > best_val_auc:
                    saver.save(sess, 'weights/add_1/model')
                    best_val_auc = auc_value
                print("epoch:", epoch, "auc_value", auc_value, "best", best_val_auc)


def train2():
    # parameters
    LR = 0.001
    BatchSize = 128
    EPOCH = 30

    train_data_path = "../train/"
    validation_data_path = "../validation/"
    # data
    train_x, train_y = load_data(train_data_path)
    valid_x, valid_y = load_data(validation_data_path)

    # model & input and output of model
    tf.reset_default_graph()
    model = Model2()

    onehots_shape = list(train_x.shape[1:])
    input_place_holder = tf.placeholder(tf.float32, [None] + onehots_shape, name='input')
    input_place_holder_reshaped = tf.reshape(input_place_holder, [-1] + onehots_shape + [1])
    label_place_holder = tf.placeholder(tf.int32, [None], name='label')
    label_place_holder_2d = tf.one_hot(label_place_holder, 2)
    output, output_with_sm = model(input_place_holder_reshaped)
    model.summary()  # show model's structure

    # loss
    bce = tf.keras.losses.BinaryCrossentropy()  # compute cost
    loss = bce(label_place_holder_2d, output_with_sm)

    # Optimizer
    train_op = tf.train.AdamOptimizer(LR).minimize(loss)

    # auc
    prediction_place_holder = tf.placeholder(tf.float64, [None], name='pred')
    auc, update_op = tf.metrics.auc(labels=label_place_holder, predictions=prediction_place_holder)

    # run
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init_op)

        saver = tf.train.Saver()

        train_size = train_x.shape[0]
        best_val_auc = 0
        for epoch in range(EPOCH):
            for i in range(0, train_size, BatchSize):
                b_x, b_y = train_x[i:i + BatchSize], train_y[i:i + BatchSize]
                _, loss_ = sess.run([train_op, loss], {'input:0': b_x, 'label:0': b_y})

                # print("Epoch {}: [{}/{}], training set loss: {:.4}".format(epoch, i, train_size, loss_))

            if epoch % 1 == 0:
                val_prediction = sess.run(output_with_sm, {'input:0': valid_x})
                val_prediction = val_prediction[:, 1]
                auc_value = sess.run(update_op,
                                     feed_dict={prediction_place_holder: val_prediction, label_place_holder: valid_y})

                if auc_value > best_val_auc:
                    saver.save(sess, 'weights/add_2/model')
                    best_val_auc = auc_value
                print("epoch:", epoch, "auc_value", auc_value, "best", best_val_auc)

if __name__=="__main__":
    train1()
    train2()
