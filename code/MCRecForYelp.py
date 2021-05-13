import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

import numpy as np
import keras
from keras import backend as K
from keras import initializers
from keras.models import Sequential, Model, load_model, save_model
from keras.layers import Dense, Lambda, Activation, LSTM, Reshape, Conv1D, GlobalMaxPooling1D, Dropout
from keras.layers import Embedding, Input, Dense, merge, Reshape, Merge, Flatten, concatenate, RepeatVector, multiply
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adagrad, Adam, SGD, RMSprop, Nadam
from keras.regularizers import l2
from YelpDataset import Dataset
from evaluate import evaluate_model
from time import time
import multiprocessing as mp
import sys
import math
import argparse
import scipy.sparse as sp
import gc


def parse_args():
    parser = argparse.ArgumentParser(description="Run MCRec.")
    parser.add_argument('--dataset', nargs='?', default='yelp',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--learner', nargs='?', default='adam',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--latent_dim', type=int, default='128',
                        help="Embedding size for user and item embedding")
    parser.add_argument('--latent_layer_dim', nargs='?', default='[512, 256, 128, 64]',
                        help="Embedding size for each layer")
    parser.add_argument('--num_neg', type=int, default=1,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--K', type=int, default=3,
                        help='Number of topK in experiments.')
    parser.add_argument('--metapath', type=str, default='all',
                        help='Specify meta-paths. "+" or "-" means only use or remove. "all" means use all meta-paths' )

    return parser.parse_args()


def slice(x, index):
    return x[:, index, :, :]


def slice_2(x, index):
    return x[:, index, :]


def path_attention(user_latent, item_latent, path_latent, latent_size, att_size, path_attention_layer_1,
                   path_attention_layer_2, path_name):
    # user_latent (batch_size, latent_size)
    # item_latent (batch_size, latent_size)
    # path_latent (batch_size, path_num, mp_latent_size)
    latent_size = user_latent.shape[1].value
    path_num, path_latent_size = path_latent.shape[1].value, path_latent.shape[2].value

    path = Lambda(slice_2, output_shape=(path_latent_size,), arguments={'index': 0})(path_latent)
    inputs = concatenate([user_latent, item_latent, path])
    output = (path_attention_layer_1(inputs))
    output = (path_attention_layer_2(output))
    for i in range(1, path_num):
        path = Lambda(slice_2, output_shape=(path_latent_size,), arguments={'index': i})(path_latent)
        inputs = concatenate([user_latent, item_latent, path])
        tmp_output = (path_attention_layer_1(inputs))
        tmp_output = (path_attention_layer_2(tmp_output))
        output = concatenate([output, tmp_output])

    atten = Lambda(lambda x: K.softmax(x), name='%s_attention_softmax' % path_name)(output)
    output = Lambda(lambda x: K.sum(x[0] * K.expand_dims(x[1], -1), 1))([path_latent, atten])
    return output


def get_ubcib_embedding(ubcib_input, path_num, timestamps, length, user_latent, item_latent, path_attention_layer_1,
                         path_attention_layer_2):
    conv_ubcib = Conv1D(filters=128,
                         kernel_size=4,
                         activation='relu',
                         kernel_regularizer=l2(0.0),
                         kernel_initializer='glorot_uniform',
                         padding='valid',
                         strides=1,
                         name='ubcib_conv')

    path_input = Lambda(slice, output_shape=(timestamps, length), arguments={'index': 0})(ubcib_input)
    output = conv_ubcib(path_input)
    output = GlobalMaxPooling1D()(output)
    output = Dropout(0.5)(output)

    for i in range(1, path_num):
        path_input = Lambda(slice, output_shape=(timestamps, length), arguments={'index': i})(ubcib_input)
        tmp_output = GlobalMaxPooling1D()(conv_ubcib(path_input))
        tmp_output = Dropout(0.5)(tmp_output)
        output = concatenate([output, tmp_output])

    output = Reshape((path_num, 128))(output)
    # output = path_attention(user_latent, item_latent, output, 128, 64, path_attention_layer_1, path_attention_layer_2, 'ubcib')
    output = GlobalMaxPooling1D()(output)
    return output


def get_ubcab_embedding(ubcab_input, path_num, timestamps, length, user_latent, item_latent, path_attention_layer_1,
                       path_attention_layer_2):
    conv_ubcab = Conv1D(filters=128,
                       kernel_size=4,
                       activation='relu',
                       kernel_regularizer=l2(0.0),
                       kernel_initializer='glorot_uniform',
                       padding='valid',
                       strides=1,
                       name='ubcab_conv')

    path_input = Lambda(slice, output_shape=(timestamps, length), arguments={'index': 0})(ubcab_input)
    output = GlobalMaxPooling1D()(conv_ubcab(path_input))
    output = Dropout(0.5)(output)

    for i in range(1, path_num):
        path_input = Lambda(slice, output_shape=(timestamps, length), arguments={'index': i})(ubcab_input)
        tmp_output = GlobalMaxPooling1D()(conv_ubcab(path_input))
        tmp_output = Dropout(0.5)(tmp_output)
        output = concatenate([output, tmp_output])

    output = Reshape((path_num, 128))(output)
    # output = path_attention(user_latent, item_latent, output, 128, 64, path_attention_layer_1, path_attention_layer_2, 'ubcab')
    output = GlobalMaxPooling1D()(output)
    return output


def get_ubub_embedding(ubub_input, path_num, timestamps, length, user_latent, item_latent, path_attention_layer_1,
                       path_attention_layer_2):
    conv_ubub = Conv1D(filters=128,
                       kernel_size=4,
                       activation='relu',
                       kernel_regularizer=l2(0.0),
                       kernel_initializer='glorot_uniform',
                       padding='valid',
                       strides=1,
                       name='ubub_conv')

    path_input = Lambda(slice, output_shape=(timestamps, length), arguments={'index': 0})(ubub_input)
    output = GlobalMaxPooling1D()(conv_ubub(path_input))
    output = Dropout(0.5)(output)

    for i in range(1, path_num):
        path_input = Lambda(slice, output_shape=(timestamps, length), arguments={'index': i})(ubub_input)
        tmp_output = GlobalMaxPooling1D()(conv_ubub(path_input))
        tmp_output = Dropout(0.5)(tmp_output)
        output = concatenate([output, tmp_output])

    output = Reshape((path_num, 128))(output)
    # output = path_attention(user_latent, item_latent, output, 128, 64, path_attention_layer_1, path_attention_layer_2, 'ubub')
    output = GlobalMaxPooling1D()(output)
    return output


def get_uub_embedding(ubub_input, path_num, timestamps, length, user_latent, item_latent, path_attention_layer_1,
                       path_attention_layer_2):
    conv_ubub = Conv1D(filters=128,
                       kernel_size=3,
                       activation='relu',
                       kernel_regularizer=l2(0.0),
                       kernel_initializer='glorot_uniform',
                       padding='valid',
                       strides=1,
                       name='uub_conv')

    path_input = Lambda(slice, output_shape=(timestamps, length), arguments={'index': 0})(ubub_input)
    output = GlobalMaxPooling1D()(conv_ubub(path_input))
    output = Dropout(0.5)(output)

    for i in range(1, path_num):
        path_input = Lambda(slice, output_shape=(timestamps, length), arguments={'index': i})(ubub_input)
        tmp_output = GlobalMaxPooling1D()(conv_ubub(path_input))
        tmp_output = Dropout(0.5)(tmp_output)
        output = concatenate([output, tmp_output])


    output = Reshape((path_num, 128))(output)
    # output = path_attention(user_latent, item_latent, output, 128, 64, path_attention_layer_1, path_attention_layer_2, 'uub')
    output = GlobalMaxPooling1D()(output)
    return output


def metapath_attention(user_latent, item_latent, metapath_latent, latent_size, att_size):
    # user_latent (batch_size, latent_size)
    # item_latent (batch_size, latent_size)
    # metapath_latent (batch_size, path_num, mp_latent_size)
    # print user_latent.shape
    latent_size = user_latent.shape[1].value
    path_num, mp_latent_size = metapath_latent.shape[1].value, metapath_latent.shape[2].value
    dense_layer_1 = Dense(att_size,
                          activation='relu',
                          kernel_initializer='glorot_normal',
                          kernel_regularizer=l2(0.001),
                          name='metapath_attention_layer_1')

    dense_layer_2 = Dense(1,
                          activation='relu',
                          kernel_initializer='glorot_normal',
                          kernel_regularizer=l2(0.001),
                          name='metapath_attention_layer_2')

    metapath = Lambda(slice_2, output_shape=(mp_latent_size,), arguments={'index': 0})(metapath_latent)
    inputs = concatenate([user_latent, item_latent, metapath])
    output = (dense_layer_1(inputs))
    output = (dense_layer_2(output))
    for i in range(1, path_num):
        metapath = Lambda(slice_2, output_shape=(mp_latent_size,), arguments={'index': i})(metapath_latent)
        inputs = concatenate([user_latent, item_latent, metapath])
        tmp_output = (dense_layer_1(inputs))
        tmp_output = (dense_layer_2(tmp_output))
        output = concatenate([output, tmp_output])

    atten = Lambda(lambda x: K.softmax(x), name='metapath_attention_softmax')(output)
    output = Lambda(lambda x: K.sum(x[0] * K.expand_dims(x[1], -1), 1))([metapath_latent, atten])
    return output


def user_attention(user_latent, path_output):
    latent_size = user_latent.shape[1].value

    inputs = concatenate([user_latent, path_output])
    # inputs = user_latent
    output = Dense(latent_size,
                   activation='relu',
                   kernel_initializer='glorot_normal',
                   kernel_regularizer=l2(0.001),
                   name='user_attention_layer')(inputs)
    atten = Lambda(lambda x: K.softmax(x), name='user_attention_softmax')(output)
    output = multiply([user_latent, atten])
    return output


def item_attention(item_latent, path_output):
    latent_size = item_latent.shape[1].value

    inputs = concatenate([item_latent, path_output])
    # inputs = item_latent
    output = Dense(latent_size,
                   activation='relu',
                   kernel_initializer='glorot_normal',
                   kernel_regularizer=l2(0.001),
                   name='item_attention_layer')(inputs)
    atten = Lambda(lambda x: K.softmax(x), name='item_attention_softmax')(output)
    output = multiply([item_latent, atten])
    return output


def get_model(usize, isize, path_nums, timestamps, length, layers=[20, 10], reg_layers=[0, 0], latent_dim=40,
              reg_latent=0, metapath='all'):
    user_input = Input(shape=(1,), dtype='int32', name='user_input', sparse=False)
    item_input = Input(shape=(1,), dtype='int32', name='item_input', sparse=False)
    ubcab_input = Input(shape=(path_nums[0], timestamps[0], length,), dtype='float32', name='ubcab_input')
    ubub_input = Input(shape=(path_nums[1], timestamps[1], length,), dtype='float32', name='ubub_input')
    ubcib_input = Input(shape=(path_nums[2], timestamps[2], length,), dtype='float32', name='ubcib_input')
    uub_input = Input(shape=(path_nums[3], timestamps[3], length,), dtype='float32', name='uub_input')
    Embedding_User_Feedback = Embedding(input_dim=usize,
                                        output_dim=latent_dim,
                                        input_length=1,
                                        embeddings_initializer='glorot_normal',
                                        name='user_feedback_embedding')

    Embedding_Item_Feedback = Embedding(input_dim=isize,
                                        output_dim=latent_dim,
                                        input_length=1,
                                        embeddings_initializer='glorot_normal',
                                        name='item_feedback_embedding')
    user_latent = Reshape((latent_dim,))(Flatten()(Embedding_User_Feedback(user_input)))
    item_latent = Reshape((latent_dim,))(Flatten()(Embedding_Item_Feedback(item_input)))

    path_attention_layer_1 = Dense(128,
                                   activation='relu',
                                   kernel_regularizer=l2(0.001),
                                   kernel_initializer='glorot_normal',
                                   name='path_attention_layer_1')

    path_attention_layer_2 = Dense(1,
                                   activation='relu',
                                   kernel_regularizer=l2(0.001),
                                   kernel_initializer='glorot_normal',
                                   name='path_attention_layer_2')

    ubcab_latent = get_ubcab_embedding(ubcab_input, path_nums[0], timestamps[0], length, user_latent, item_latent,
                                     path_attention_layer_1, path_attention_layer_2)
    ubub_latent = get_ubub_embedding(ubub_input, path_nums[1], timestamps[1], length, user_latent, item_latent,
                                     path_attention_layer_1, path_attention_layer_2)
    ubcib_latent = get_ubcib_embedding(ubcib_input, path_nums[2], timestamps[2], length, user_latent, item_latent,
                                         path_attention_layer_1, path_attention_layer_2)
    uub_latent = get_uub_embedding(uub_input, path_nums[3], timestamps[3], length, user_latent, item_latent,
                                     path_attention_layer_1, path_attention_layer_2)

    path_output = concatenate([
        ubcab_latent,
        ubub_latent,
        ubcib_latent,
        uub_latent
    ])
    path_output = Reshape((4, 128))(path_output)

    if metapath == 'all':
        path_output = concatenate([
            ubcab_latent,
            ubub_latent,
            ubcib_latent,
            uub_latent
        ])
        path_output = Reshape((4, 128))(path_output)
    elif metapath[0] == '+':
        if metapath[1:] == 'uub':
            path_output = uub_latent
        elif metapath[1:] == 'ubcib':
            path_output = ubcib_latent
        elif metapath[1:] == 'ubub':
            path_output = ubub_latent
        elif metapath[1:] == 'ubcab':
            path_output = ubcab_latent
        path_output = Reshape((1, 128))(path_output)
    else:
        if metapath[1:] == 'uub':
            path_output = concatenate([
                ubcab_latent,
                ubub_latent,
                ubcib_latent
            ])
        elif metapath[1:] == 'ubcib':
            path_output = concatenate([
                ubcab_latent,
                ubub_latent,
                uub_latent
            ])
        elif metapath[1:] == 'ubub':
            path_output = concatenate([
                ubcab_latent,
                ubcib_latent,
                uub_latent
            ])
        elif metapath[1:] == 'ubcab':
            path_output = concatenate([
                ubub_latent,
                ubcib_latent,
                uub_latent
            ])
        path_output = Reshape((3, 128))(path_output)

    path_output = metapath_attention(user_latent, item_latent, path_output, latent_dim, 128)

    user_atten = user_attention(user_latent, path_output)
    item_atten = item_attention(item_latent, path_output)

    output = concatenate([user_atten
                             , path_output
                             , item_atten])
    for idx in range(0, len(layers)):
        layer = Dense(layers[idx],
                      kernel_regularizer=l2(0.001),
                      kernel_initializer='glorot_normal',
                      activation='relu',
                      name='item_layer%d' % idx)
        output = layer(output)

    # user_output = concatenate([user_atten, path_output])
    # for idx in xrange(0, len(layers)):
    #    layer = Dense(layers[idx],
    #                  kernel_regularizer = l2(0.001),
    #                  kernel_initializer = 'glorot_normal',
    #                  activation = 'relu',
    #                  name = 'user_layer%d' % idx)
    #    user_output = layer(user_output)

    # item_output = concatenate([path_output, item_atten])
    # for idx in xrange(0, len(layers)):
    #    layer = Dense(layers[idx],
    #                  kernel_regularizer = l2(0.001),
    #                  kernel_initializer = 'glorot_normal',
    #                  activation = 'relu',
    #                  name = 'item_layer%d' % idx)
    # item_output = layer(item_output)

    # output = concatenate([user_output, item_output])

    print('output.shape = ', output.shape)
    prediction_layer = Dense(1,
                             activation='sigmoid',
                             kernel_initializer='lecun_normal',
                             name='prediction')

    prediction = prediction_layer(output)
    model = Model(inputs=[user_input, item_input,
                          ubcab_input,
                          ubub_input, ubcib_input,
                          uub_input
                          ],
                  outputs=[prediction])

    return model


def get_train_instances(user_feature, item_feature, type_feature,
                        path_ubcab,
                        path_ubub, path_ubcib,
                        path_uub,
                        path_nums, timestamps,
                        train_list, num_negatives, batch_size, shuffle=True):
    num_batches_per_epoch = int((len(train_list) - 1) / batch_size) + 1

    def data_generator():
        data_size = len(train_list)
        while True:
            if shuffle == True:
                np.random.shuffle(train_list)

            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                k = 0
                _user_input = np.zeros((batch_size * (num_negatives + 1),))
                _item_input = np.zeros((batch_size * (num_negatives + 1),))
                _ubcab_input = np.zeros((batch_size * (num_negatives + 1), path_nums[0], timestamps[0], 64))
                _ubub_input = np.zeros((batch_size * (num_negatives + 1), path_nums[1], timestamps[1], 64))
                _ubcib_input = np.zeros((batch_size * (num_negatives + 1), path_nums[2], timestamps[2], 64))
                _uub_input = np.zeros((batch_size * (num_negatives + 1), path_nums[3], timestamps[3], 64))
                _labels = np.zeros(batch_size * (num_negatives + 1))

                for u, i in train_list[start_index: end_index]:

                    _user_input[k] = u
                    _item_input[k] = i

                    if (u, i) in path_ubcab:
                        for p_i in range(len(path_ubcab[(u, i)])):
                            for p_j in range(len(path_ubcab[(u, i)][p_i])):
                                type_id = path_ubcab[(u, i)][p_i][p_j][0]
                                index = path_ubcab[(u, i)][p_i][p_j][1]
                                if type_id == 1:
                                    _ubcab_input[k][p_i][p_j] = user_feature[index]
                                elif type_id == 2:
                                    _ubcab_input[k][p_i][p_j] = item_feature[index]

                    if (u, i) in path_ubub:
                        for p_i in range(len(path_ubub[(u, i)])):
                            for p_j in range(len(path_ubub[(u, i)][p_i])):
                                type_id = path_ubub[(u, i)][p_i][p_j][0]
                                index = path_ubub[(u, i)][p_i][p_j][1]
                                if type_id == 1:
                                    _ubub_input[k][p_i][p_j] = user_feature[index]
                                elif type_id == 2:
                                    _ubub_input[k][p_i][p_j] = item_feature[index]

                    if (u, i) in path_ubcib:
                        for p_i in range(len(path_ubcib[(u, i)])):
                            for p_j in range(len(path_ubcib[(u, i)][p_i])):
                                type_id = path_ubcib[(u, i)][p_i][p_j][0]
                                index = path_ubcib[(u, i)][p_i][p_j][1]
                                if type_id == 1:
                                    _ubcib_input[k][p_i][p_j] = user_feature[index]
                                elif type_id == 2:
                                    _ubcib_input[k][p_i][p_j] = item_feature[index]

                    if (u, i) in path_uub:
                        for p_i in range(len(path_uub[(u, i)])):
                            for p_j in range(len(path_uub[(u, i)][p_i])):
                                type_id = path_uub[(u, i)][p_i][p_j][0]
                                index = path_uub[(u, i)][p_i][p_j][1]
                                if type_id == 1:
                                    _uub_input[k][p_i][p_j] = user_feature[index]
                                elif type_id == 2:
                                    _uub_input[k][p_i][p_j] = item_feature[index]
                    _labels[k] = 1.0
                    k += 1
                    # negative instances
                    for t in range(num_negatives):
                        j = np.random.randint(1, num_items - 1)
                        while j in user_item_map[u]:
                            j = np.random.randint(1, num_items - 1)

                        _user_input[k] = u
                        _item_input[k] = j

                        if (u, j) in path_ubcab:
                            for p_i in range(len(path_ubcab[(u, j)])):
                                for p_j in range(len(path_ubcab[(u, j)][p_i])):
                                    type_id = path_ubcab[(u, j)][p_i][p_j][0]
                                    index = path_ubcab[(u, j)][p_i][p_j][1]
                                    if type_id == 1:
                                        _ubcab_input[k][p_i][p_j] = user_feature[index]
                                    elif type_id == 2:
                                        _ubcab_input[k][p_i][p_j] = item_feature[index]

                        if (u, j) in path_ubub:
                            for p_i in range(len(path_ubub[(u, j)])):
                                for p_j in range(len(path_ubub[(u, j)][p_i])):
                                    type_id = path_ubub[(u, j)][p_i][p_j][0]
                                    index = path_ubub[(u, j)][p_i][p_j][1]
                                    if type_id == 1:
                                        _ubub_input[k][p_i][p_j] = user_feature[index]
                                    elif type_id == 2:
                                        _ubub_input[k][p_i][p_j] = item_feature[index]
                        if (u, j) in path_ubcib:
                            for p_i in range(len(path_ubcib[(u, j)])):
                                for p_j in range(len(path_ubcib[(u, j)][p_i])):
                                    type_id = path_ubcib[(u, j)][p_i][p_j][0]
                                    index = path_ubcib[(u, j)][p_i][p_j][1]
                                    if type_id == 1:
                                        _ubcib_input[k][p_i][p_j] = user_feature[index]
                                    elif type_id == 2:
                                        _ubcib_input[k][p_i][p_j] = item_feature[index]

                        if (u, j) in path_uub:
                            for p_i in range(len(path_uub[(u, j)])):
                                for p_j in range(len(path_uub[(u, j)][p_i])):
                                    type_id = path_uub[(u, j)][p_i][p_j][0];
                                    index = path_uub[(u, j)][p_i][p_j][1]
                                    if type_id == 1:
                                        _uub_input[k][p_i][p_j] = user_feature[index]
                                    elif type_id == 2:
                                        _uub_input[k][p_i][p_j] = item_feature[index]
                        _labels[k] = 0.0
                        k += 1
                yield ([_user_input, _item_input,
                        _ubcab_input,
                        _ubub_input, _ubcib_input,
                        _uub_input
                        ], _labels)

    return num_batches_per_epoch, data_generator()


if __name__ == '__main__':

    args = parse_args()

    dataset = args.dataset
    latent_dim = args.latent_dim
    layers = eval(args.latent_layer_dim)
    learning_rate = args.lr
    epochs = args.epochs
    batch_size = args.batch_size
    num_negatives = args.num_neg
    learner = args.learner
    verbose = args.verbose
    topK = args.K
    metapath = args.metapath

    # print ("layers : ",layers, "  type : ", type(layers))

    out = 0
    reg_latent = 0
    reg_layes = [0, 0, 0, 0]
    evaluation_threads = 1

    # dataset = 'ml-100k'
    # latent_dim = 128
    # reg_latent = 0
    # layers = [512, 256, 128, 64]
    # reg_layes = [0 ,0, 0, 0]
    # learning_rate = 0.001
    # epochs = 30
    # batch_size = 256
    # num_negatives = 4
    # learner = 'adam'
    # verbose = 1
    # out = 0

    print('num_negatives = ', num_negatives)

    t1 = time()
    dataset = Dataset('../data/yelp/' + dataset)
    trainMatrix, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    train = dataset.train
    user_item_map = dataset.user_item_map
    item_user_map = dataset.item_user_map
    path_ubcab = dataset.path_ubcab
    path_ubub = dataset.path_ubub
    path_ubcib = dataset.path_ubcib
    path_uub = dataset.path_uub
    user_feature, item_feature, type_feature = dataset.user_feature, dataset.item_feature, dataset.type_feature
    num_users, num_items = trainMatrix.shape[0], trainMatrix.shape[1]
    path_nums = [dataset.ubcab_path_num, dataset.ubub_path_num, dataset.ubcib_path_num, dataset.uub_path_num]
    timestamps = [dataset.ubcab_timestamp, dataset.ubub_timestamp, dataset.ubcib_timestamp, dataset.uub_timestamp]
    length = dataset.fea_size

    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d" % (
        time() - t1, num_users, num_items, len(train), len(testRatings)))
    print('path nums = ', path_nums)
    print('timestamps = ', timestamps)

    model = get_model(num_users, num_items,
                      path_nums, timestamps,
                      length, layers, reg_layes, latent_dim, reg_latent, metapath)

    model.compile(optimizer=Adam(lr=learning_rate, decay=1e-4),
                  loss='binary_crossentropy')
    # model.compile(optimizer = Nadam(),
    #              loss = 'binary_crossentropy')

    # Check Init performance
    t1 = time()
    # (ps, rs, ndcgs) = evaluate_model(model, user_feature, item_feature, type_feature, num_users, num_items,
    #                                  path_ubcab,
    #                                  path_ubub,
    #                                  path_ubcib,
    #                                  path_uub,
    #                                  path_nums, timestamps, length, testRatings,
    #                                  testNegatives, topK, evaluation_threads)
    # p, r, ndcg = np.array(ps).mean(), np.array(rs).mean(), np.array(ndcgs).mean()
    # print('Init: Precision = %.4f, Recall = %.4f, NDCG = %.4f [%.1f]' % (p, r, ndcg, time() - t1))

    best_p = -1
    p_list, r_list, ndcg_list = [], [], []
    print('Begin training....')

    for epoch in range(epochs):
        t1 = time()

        # Generate training instance
        train_steps, train_batches = get_train_instances(user_feature, item_feature, type_feature,
                                                         path_ubcab,
                                                         path_ubub,
                                                         path_ubcib,
                                                         path_uub,
                                                         path_nums, timestamps, train,
                                                         num_negatives, batch_size, True)
        t = time()
        print('[%.1f s] epoch %d train_steps %d' % (t - t1, epoch, train_steps))
        # Training
        hist = model.fit_generator(train_batches,
                                   train_steps,
                                   epochs=1,
                                   verbose=0)
        print('training time %.1f s' % (time() - t))

        t2 = time()
        if epoch % verbose == 0:
            (ps, rs, ndcgs) = evaluate_model(model, user_feature, item_feature, type_feature, num_users, num_items,
                                             path_ubcab,
                                             path_ubub, path_ubcib,
                                             path_uub,
                                             path_nums, timestamps,
                                             length, testRatings, testNegatives, topK, evaluation_threads)
            p, r, ndcg, loss = np.array(ps).mean(), np.array(rs).mean(), np.array(ndcgs).mean(), hist.history['loss'][0]
            print('Validation: Iteration %d [%.1f s]: Precision = %.4f, Recall = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]'
                  % (epoch, t2 - t1, p, r, ndcg, loss, time() - t2))

            # if p > best_p:
            #    best_p = p
            #    attention_layer_model = Model(inputs=model.input,
            #                          outputs = [model.get_layer('user_input').output, model.get_layer('item_input').output, model.get_layer('metapath_attention_softmax').output])
            #    [user_input_output, item_input_output, metapath_attention_output] = attention_layer_model.predict_generator(train_batches, train_steps)
            #    with open('../data/ml-100k.attention_2', 'w') as outfile:
            #        num = user_input_output.shape[0]
            #        for i in range(num):
            #            outfile.write(str(user_input_output[i]) + ',' + str(item_input_output[i]))
            #            for j in range(metapath_attention_output.shape[1]):
            #                outfile.write(' ' + str(metapath_attention_output[i][j]))
            #            outfile.write('\n')
            #    print 'write succeccfully...'
            p_list.append(p)
            r_list.append(r)
            ndcg_list.append(ndcg)
    print("End. Precision = %.4f, Recall = %.4f, NDCG = %.4f. " % (max(p_list), max(r_list), max(ndcg_list)))
