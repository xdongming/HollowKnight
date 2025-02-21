import copy

import tensorflow as tf
from tensorflow.keras.models import load_model, Model as mod
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.layers import add, Input, Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization, \
    Activation, GlobalAveragePooling2D, Conv3D, MaxPooling3D, GlobalAveragePooling3D, Reshape, Lambda

import time
import os


class BasicBlock(layers.Layer):
    def __init__(self, filter_num, name, stride=1, **kwargs):
        super(BasicBlock, self).__init__(**kwargs)
        self.filter_num = filter_num
        self.stride = stride
        self.layers = []
        self.conv1 = layers.Conv2D(filter_num, 3, strides=stride, padding='same', name=name + '_1')
        # self.bn1=layers.BatchNormalization()
        self.relu = layers.Activation('relu')

        self.conv2 = layers.Conv2D(filter_num, 3, strides=1, padding='same', name=name + '_2')
        # self.bn2 = layers.BatchNormalization()
        self.layers.append(self.conv1)
        self.layers.append(self.conv2)
        # self.layers.append(self.bn1)
        # self.layers.append(self.bn2)
        if stride != 1:
            self.downsample = models.Sequential()
            self.downsample.add(layers.Conv2D(filter_num, 1, strides=stride))
            self.layers.append(self.downsample)
        else:
            self.downsample = lambda x: x

    def get_layer(self, index):
        return self.layers[index]

    def get_layers(self):
        return self.layers

    def call(self, input, training=None):
        out = self.conv1(input)
        # out=self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        # out=self.bn2(out)

        identity = self.downsample(input)
        output = layers.add([out, identity])  # ***
        output = tf.nn.relu(output)
        return output

    def get_config(self):
        config = {
            'filter_num':
                self.filter_num,
            'stride':
                self.stride
        }

        base_config = super(BasicBlock, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Model:
    def __init__(self, input_shape, act_dim):
        self.act_dim = act_dim
        self.input_shape = input_shape
        self._build_model()
        self.act_loss = []
        self.move_loss = []

    def load_model(self):

        # self.shared_model = load_model("./model/shared_model.h5", custom_objects={'BasicBlock': BasicBlock})
        if os.path.exists("./model/act_part.h5"):
            print("load action model")
            self.act_model = models.Sequential()
            self.private_act_model = load_model("./model/act_part.h5", custom_objects={'BasicBlock': BasicBlock})
            self.target_act_model  = load_model("./model/act_part.h5", custom_objects={'BasicBlock': BasicBlock})
            # self.act_model.add(self.shared_model)
            self.act_model.add(self.private_act_model)

        if os.path.exists("./model/move_part.h5"):
            print("load move model")
            self.move_model = models.Sequential()
            self.private_move_model = load_model("./model/move_part.h5", custom_objects={'BasicBlock': BasicBlock})
            self.target_move_model = load_model("./model/move_part.h5", custom_objects={'BasicBlock': BasicBlock})
            # self.move_model.add(self.shared_model)
            self.move_model.add(self.private_move_model)

    def save_mode(self):
        print("save model")
        self.private_act_model.save("./model/act_part.h5")
        self.private_move_model.save("./model/move_part.h5")

    def build_resblock(self, filter_num, blocks, name="Resnet", stride=1):
        res_blocks = models.Sequential()
        # may down sample
        res_blocks.add(BasicBlock(filter_num, name + '_1', stride))
        # just down sample one time
        for pre in range(1, blocks):
            res_blocks.add(BasicBlock(filter_num, name + '_2', stride=1))
        return res_blocks

    # use two groups of net, one for action, one for move
    def _build_model(self):

        # ------------------ build evaluate_net ------------------

        # self.shared_model = models.Sequential()
        # self.private_act_model = models.Sequential()
        # self.private_move_model = models.Sequential()

        # shared part
        # pre-process block
        # self.shared_model.add(Conv2D(64, (2,3,3),strides=(1,2,2), input_shape=self.input_shape, name='conv1'))
        # # self.shared_model.add(BatchNormalization(name='b1'))
        # self.shared_model.add(Activation('relu'))
        # self.shared_model.add(MaxPooling3D(pool_size=(2,2,2), strides=1, padding="VALID", name='p1'))

        # # resnet blocks
        # self.shared_model.add(self.build_resblock(64, 2, name='Resnet_1'))
        # self.shared_model.add(self.build_resblock(80, 2, name='Resnet_2', stride=2))
        # self.shared_model.add(self.build_resblock(128, 2, name='Resnet_3', stride=2))

        # output layer for action model
        inputs = Input(shape=self.input_shape)
        x = Conv3D(32, (2, 3, 3), strides=(1, 2, 2), input_shape=self.input_shape, name='conv1', activation='relu')(
            inputs)
        x = Conv3D(48, (2, 3, 3), strides=(1, 1, 1), input_shape=self.input_shape, name='conv2', activation='relu')(x)
        x = Conv3D(64, (2, 3, 3), strides=(1, 1, 1), input_shape=self.input_shape, name='conv3', activation='relu')(x)
        x = Lambda(lambda x: tf.reduce_sum(x, 1))(x)
        x = self.build_resblock(64, 2, name='Resnet_1')(x)
        x = self.build_resblock(96, 2, name='Resnet_2', stride=2)(x)
        x = self.build_resblock(128, 2, name='Resnet_3', stride=2)(x)
        x = self.build_resblock(256, 2, name='Resnet_4', stride=2)(x)
        x = GlobalAveragePooling2D(name='pooling')(x)
        v_func = Dense(1, name='v_func')(x)
        a_func = Dense(self.act_dim, name='a_func')(x)
        a_func = add([a_func, -tf.reduce_mean(a_func, keepdims=True)])
        outputs = add([v_func, a_func])
        self.private_act_model = mod(inputs=inputs, outputs=outputs)

        # self.private_act_model.add(Reshape((1, -1)))
        # self.private_act_model.add(CuDNNLSTM(32))
        # self.private_act_model.add(Dense(self.act_dim, name="d1"))        # action model
        self.private_act_model.summary()
        self.act_model = models.Sequential()
        # self.act_model.add(self.shared_model)
        self.act_model.add(self.private_act_model)

        #target_act_model
        input = Input(shape=self.input_shape)
        y = Conv3D(32, (2, 3, 3), strides=(1, 2, 2), input_shape=self.input_shape, name='conv1', activation='relu')(
            input)
        y = Conv3D(48, (2, 3, 3), strides=(1, 1, 1), input_shape=self.input_shape, name='conv2', activation='relu')(y)
        y = Conv3D(64, (2, 3, 3), strides=(1, 1, 1), input_shape=self.input_shape, name='conv3', activation='relu')(y)
        y = Lambda(lambda x: tf.reduce_sum(x, 1))(y)
        y = self.build_resblock(64, 2, name='Resnet_1')(y)
        y = self.build_resblock(96, 2, name='Resnet_2', stride=2)(y)
        y = self.build_resblock(128, 2, name='Resnet_3', stride=2)(y)
        y = self.build_resblock(256, 2, name='Resnet_4', stride=2)(y)
        y = GlobalAveragePooling2D(name='pooling')(y)
        v = Dense(1, name='v_func')(y)
        a = Dense(self.act_dim, name='a_func')(y)
        a = add([a, -tf.reduce_mean(a, keepdims=True)])
        output = add([v, a])
        self.target_act_model = mod(inputs=input, outputs=output)

        # output layer for move model
        _inputs = Input(shape=self.input_shape)
        _x = Conv3D(32, (2, 3, 3), strides=(1, 2, 2), input_shape=self.input_shape, name='conv1', activation='relu')(
            _inputs)
        _x = Conv3D(48, (2, 3, 3), strides=(1, 1, 1), input_shape=self.input_shape, name='conv2', activation='relu')(_x)
        _x = Conv3D(64, (2, 3, 3), strides=(1, 1, 1), input_shape=self.input_shape, name='conv3', activation='relu')(_x)
        _x = Lambda(lambda x: tf.reduce_sum(x, 1))(_x)
        _x = self.build_resblock(64, 2, name='Resnet_1')(_x)
        _x = self.build_resblock(96, 2, name='Resnet_2', stride=2)(_x)
        _x = self.build_resblock(128, 2, name='Resnet_3', stride=2)(_x)
        _x = self.build_resblock(256, 2, name='Resnet_4', stride=2)(_x)
        _x = GlobalAveragePooling2D(name='pooling')(_x)
        _v_func = Dense(1, name='v_func')(_x)
        _a_func = Dense(self.act_dim, name='a_func')(_x)
        _a_func = add([_a_func, -tf.reduce_mean(_a_func, keepdims=True)])
        _outputs = add([_v_func, _a_func])
        self.private_move_model = mod(inputs=_inputs, outputs=_outputs)

        # movement model
        self.move_model = models.Sequential()
        # self.move_model.add(self.shared_model)
        self.move_model.add(self.private_move_model)

        # target_move_model
        _input = Input(shape=self.input_shape)
        _y = Conv3D(32, (2, 3, 3), strides=(1, 2, 2), input_shape=self.input_shape, name='conv1', activation='relu')(
            _input)
        _y = Conv3D(48, (2, 3, 3), strides=(1, 1, 1), input_shape=self.input_shape, name='conv2', activation='relu')(_y)
        _y = Conv3D(64, (2, 3, 3), strides=(1, 1, 1), input_shape=self.input_shape, name='conv3', activation='relu')(_y)
        _y = Lambda(lambda x: tf.reduce_sum(x, 1))(_y)
        _y = self.build_resblock(64, 2, name='Resnet_1')(_y)
        _y = self.build_resblock(96, 2, name='Resnet_2', stride=2)(_y)
        _y = self.build_resblock(128, 2, name='Resnet_3', stride=2)(_y)
        _y = self.build_resblock(256, 2, name='Resnet_4', stride=2)(_y)
        _y = GlobalAveragePooling2D(name='pooling')(_y)
        _v = Dense(1, name='v_func')(_y)
        _a = Dense(self.act_dim, name='a_func')(_y)
        _a = add([_a, -tf.reduce_mean(_a, keepdims=True)])
        _output = add([_v, _a])
        self.target_act_model = mod(inputs=_input, outputs=_output)

    #     # ------------------ build target_model ------------------
    #    # shared part

    #     self.shared_target_model = models.Sequential()
    #     # pre-process block
    #     self.shared_target_model.add(Conv3D(64, (2,3,3),strides=(1,2,2), input_shape=self.input_shape, name='conv1'))
    #     self.shared_target_model.add(BatchNormalization(name='b1'))
    #     self.shared_target_model.add(Activation('relu'))
    #     self.shared_target_model.add(MaxPooling3D(pool_size=(2,2,2), strides=1, padding="VALID", name='p1'))

    #     # resnet blocks
    #     self.shared_target_model.add(self.build_resblock(64, 2, name='Resnet_1'))
    #     self.shared_target_model.add(self.build_resblock(80, 2, name='Resnet_2', stride=2))
    #     self.shared_target_model.add(self.build_resblock(128, 2, name='Resnet_3', stride=2))

    #     # output layer for action model
    #     self.private_act_target_model = models.Sequential()
    #     self.private_act_target_model.add(self.build_resblock(200, 2, name='Resnet_4', stride=2))
    #     self.private_act_target_model.add(GlobalAveragePooling3D())
    #     # self.private_act_target_model.add(Reshape((1, -1)))
    #     # self.private_act_target_model.add(CuDNNLSTM(32))
    #     self.private_act_target_model.add(Dense(self.act_dim, name="d1", kernel_regularizer=regularizers.L2(0.001)))

    #     # action model
    #     self.act_target_model = models.Sequential()
    #     self.act_target_model.add(self.shared_target_model)
    #     self.act_target_model.add(self.private_act_target_model)

    #     # output layer for move model
    #     self.private_move_target_model = models.Sequential()
    #     self.private_move_target_model.add(self.build_resblock(200, 2, name='Resnet_4', stride=2))
    #     self.private_move_target_model.add(GlobalAveragePooling3D())
    #     # self.private_move_target_model.add(Reshape((1, -1)))
    #     # self.private_move_target_model.add(CuDNNLSTM(32))
    #     self.private_move_target_model.add(Dense(4, name="d1", kernel_regularizer=regularizers.L2(0.001)))

    #     # action model
    #     self.move_target_model = models.Sequential()
    #     self.move_target_model.add(self.shared_target_model)
    #     self.move_target_model.add(self.private_move_target_model)

    def predict(self, input):

        input = tf.expand_dims(input, axis=0)
        # shard_output = self.shared_model.predict(input)
        pred_move = self.private_move_model(input)
        pred_act = self.private_act_model(input)
        return pred_move, pred_act
