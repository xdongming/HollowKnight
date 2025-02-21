import tensorflow as tf
import numpy as np


class DQN:
    def __init__(self, model, gamma=0.9, learnging_rate=0.0001):
        self.model = model
        self.act_dim = model.act_dim
        self.act_model = model.act_model
        self.move_model = model.move_model
        self.gamma = gamma
        self.lr = learnging_rate
        # --------------------------训练模型--------------------------- # 
        self.act_model.optimizer = tf.optimizers.Adam(learning_rate=self.lr)
        self.act_model.loss_func = tf.losses.MeanSquaredError()

        self.move_model.optimizer = tf.optimizers.Adam(learning_rate=self.lr)
        self.move_model.loss_func = tf.losses.MeanSquaredError()
        # self.act_model.train_loss = tf.metrics.Mean(name="train_loss")
        # ------------------------------------------------------------ #
        self.act_global_step = 0
        self.move_global_step = 0
        self.update_target_steps = 100  # 每隔200个training steps再把model的参数复制到target_model中

    # train functions for act model
    def act_predict(self, obs):
        """ 使用self.act_model的value网络来获取 [Q(s,a1),Q(s,a2),...]
        """
        return self.act_model.predict(obs)

    def act_train_step(self, action, features, next_features, labels):
        """ 训练步骤
        """
        with tf.GradientTape() as tape:
            # 计算 Q(s,a) 与 target_Q的均方差，得到loss
            predictions = self.act_model(features, training=True)
            next_pred = self.act_model(next_features, training=True)
            best_a = []
            for i in range(next_pred.shape[0]):
                best_a.append(np.argmax(next_pred[i, :]))
            target_q = []
            target_pred = self.model.target_act_model(next_features, training=False)
            for i in range(next_pred.shape[0]):
                target_q.append(target_pred[i, best_a[i]])
            target_q = tf.convert_to_tensor(target_q)
            enum_action = list(enumerate(action))
            pred_action_value = tf.gather_nd(predictions, indices=enum_action)
            loss = self.act_model.loss_func(labels + self.gamma * target_q, pred_action_value)
        gradients = tape.gradient(loss, self.act_model.trainable_variables)
        self.act_model.optimizer.apply_gradients(zip(gradients, self.act_model.trainable_variables))
        self.model.act_loss.append(loss)
        return np.mean(target_q), loss
        # self.act_model.train_loss.update_state(loss)

    def act_train_model(self, action, features, next_features, labels, epochs=1):
        """ 训练模型
        """
        q_sum = 0
        loss_sum = 0
        for epoch in tf.range(1, epochs + 1):
            q, loss = self.act_train_step(action, features, next_features, labels)
            q_sum = q_sum + q
            loss_sum = loss_sum + loss
        return q_sum/epochs, loss_sum/epochs
    def act_learn(self, obs, action, reward, next_obs, terminal):
        """ 使用DQN算法更新self.act_model的value网络
        """
        # print('learning')
        # 每隔200个training steps同步一次model和target_model的参数
        # if self.act_global_step % self.update_target_steps == 0:
        #     self.act_replace_target()

        # 从target_model中获取 max Q' 的值，用于计算target_Q

        q, loss = self.act_train_model(action, obs, next_obs, reward, epochs=1)
        self.act_global_step += 1
        return q, loss
        # print('finish')

    def act_replace_target(self):  # ???
        '''预测模型权重更新到target模型权重'''
        for i, l in enumerate(
                self.act_target_model.get_layer(index=1).get_layer(index=0).get_layer(index=0).get_layers()):
            l.set_weights(self.act_model.get_layer(index=1).get_layer(index=0).get_layer(index=0).get_layer(
                index=i).get_weights())
        for i, l in enumerate(
                self.act_target_model.get_layer(index=1).get_layer(index=0).get_layer(index=1).get_layers()):
            l.set_weights(self.act_model.get_layer(index=1).get_layer(index=0).get_layer(index=1).get_layer(
                index=i).get_weights())

        # for i, l in enumerate(self.act_target_model.get_layer(index=1).get_layer(index=1).get_layer(index=0).get_layers()):
        #     l.set_weights(self.act_model.get_layer(index=1).get_layer(index=1).get_layer(index=0).get_layer(index=i).get_weights())
        # for i, l in enumerate(self.act_target_model.get_layer(index=1).get_layer(index=1).get_layer(index=1).get_layers()):
        #     l.set_weights(self.act_model.get_layer(index=1).get_layer(index=1).get_layer(index=1).get_layer(index=i).get_weights())

        # for i, l in enumerate(self.act_target_model.get_layer(index=1).get_layer(index=2).get_layer(index=0).get_layers()):
        #     l.set_weights(self.act_model.get_layer(index=1).get_layer(index=2).get_layer(index=0).get_layer(index=i).get_weights())
        # for i, l in enumerate(self.act_target_model.get_layer(index=1).get_layer(index=2).get_layer(index=1).get_layers()):
        #     l.set_weights(self.act_model.get_layer(index=1).get_layer(index=2).get_layer(index=1).get_layer(index=i).get_weights())

        self.act_target_model.get_layer(index=1).get_layer(index=2).set_weights(
            self.act_model.get_layer(index=1).get_layer(index=2).get_weights())

        # self.act_target_model.get_layer(index=1).get_layer(index=6).set_weights(self.act_model.get_layer(index=1).get_layer(index=6).get_weights())

    # train functions for move_model

    def move_predict(self, obs):
        """ 使用self.move_model的value网络来获取 [Q(s,a1),Q(s,a2),...]
        """
        return self.move_model.predict(obs)

    def move_train_step(self, action, features, next_features, labels):
        """ 训练步骤
        """
        with tf.GradientTape() as tape:
            # 计算 Q(s,a) 与 target_Q的均方差，得到loss
            predictions = self.move_model(features, training=True)
            next_pred = self.move_model(next_features, training=True)
            best_a = []
            for i in range(next_pred.shape[0]):
                best_a.append(np.argmax(next_pred[i, :]))
            target_q = []
            target_pred = self.model.target_move_model(next_features, training=False)
            for i in range(next_pred.shape[0]):
                target_q.append(target_pred[i, best_a[i]])
            target_q = tf.convert_to_tensor(target_q)
            enum_action = list(enumerate(action))
            pred_action_value = tf.gather_nd(predictions, indices=enum_action)
            loss = self.move_model.loss_func(labels + self.gamma * target_q, pred_action_value)
        gradients = tape.gradient(loss, self.move_model.trainable_variables)
        self.move_model.optimizer.apply_gradients(zip(gradients, self.move_model.trainable_variables))
        self.model.move_loss.append(loss)
        return np.mean(target_q), loss
        # self.move_plot_loss()
        # print("Move loss: ", loss)
        # self.move_model.train_loss.update_state(loss)

    def move_train_model(self, action, features, next_features, labels, epochs=1):
        """ 训练模型
        """
        q_sum = 0
        loss_sum = 0
        for epoch in tf.range(1, epochs + 1):
            q, loss = self.move_train_step(action, features, next_features, labels)
            q_sum = q_sum + q
            loss_sum = loss_sum +loss
        return q_sum/epochs, loss_sum/epochs

    def move_learn(self, obs, action, reward, next_obs, terminal):
        """ 使用DQN算法更新self.move_model的value网络
        """
        q, loss = self.move_train_model(action, obs, next_obs, reward, epochs=1)
        self.move_global_step += 1
        return q, loss

    def move_replace_target(self):
        '''预测模型权重更新到target模型权重'''

        for i, l in enumerate(
                self.move_target_model.get_layer(index=1).get_layer(index=0).get_layer(index=0).get_layers()):
            l.set_weights(self.move_model.get_layer(index=1).get_layer(index=0).get_layer(index=0).get_layer(
                index=i).get_weights())
        for i, l in enumerate(
                self.move_target_model.get_layer(index=1).get_layer(index=0).get_layer(index=1).get_layers()):
            l.set_weights(self.move_model.get_layer(index=1).get_layer(index=0).get_layer(index=1).get_layer(
                index=i).get_weights())

        # for i, l in enumerate(self.move_target_model.get_layer(index=1).get_layer(index=1).get_layer(index=0).get_layers()):
        #     l.set_weights(self.move_model.get_layer(index=1).get_layer(index=1).get_layer(index=0).get_layer(index=i).get_weights())
        # for i, l in enumerate(self.move_target_model.get_layer(index=1).get_layer(index=1).get_layer(index=1).get_layers()):
        #     l.set_weights(self.move_model.get_layer(index=1).get_layer(index=1).get_layer(index=1).get_layer(index=i).get_weights())

        # for i, l in enumerate(self.move_target_model.get_layer(index=1).get_layer(index=2).get_layer(index=0).get_layers()):
        #     l.set_weights(self.move_model.get_layer(index=1).get_layer(index=2).get_layer(index=0).get_layer(index=i).get_weights())
        # for i, l in enumerate(self.move_target_model.get_layer(index=1).get_layer(index=2).get_layer(index=1).get_layers()):
        #     l.set_weights(self.move_model.get_layer(index=1).get_layer(index=2).get_layer(index=1).get_layer(index=i).get_weights())

        self.move_target_model.get_layer(index=1).get_layer(index=2).set_weights(
            self.move_model.get_layer(index=1).get_layer(index=2).get_weights())

        # self.move_target_model.get_layer(index=1).get_layer(index=6).set_weights(self.move_model.get_layer(index=1).get_layer(index=6).get_weights())

    def replace_target(self):
        # print("replace target")

        # copy conv3d_1
        self.model.shared_target_model.get_layer(index=0).set_weights(
            self.model.shared_model.get_layer(index=0).get_weights())
        # copy batchnormalization_1
        self.model.shared_target_model.get_layer(index=1).set_weights(
            self.model.shared_model.get_layer(index=1).get_weights())

        # copy shard_resnet block
        for i, l in enumerate(self.model.shared_target_model.get_layer(index=4).get_layer(index=0).get_layers()):
            l.set_weights(
                self.model.shared_model.get_layer(index=4).get_layer(index=0).get_layer(index=i).get_weights())
        for i, l in enumerate(self.model.shared_target_model.get_layer(index=4).get_layer(index=1).get_layers()):
            l.set_weights(
                self.model.shared_model.get_layer(index=4).get_layer(index=1).get_layer(index=i).get_weights())

        for i, l in enumerate(self.model.shared_target_model.get_layer(index=5).get_layer(index=0).get_layers()):
            l.set_weights(
                self.model.shared_model.get_layer(index=5).get_layer(index=0).get_layer(index=i).get_weights())
        for i, l in enumerate(self.model.shared_target_model.get_layer(index=5).get_layer(index=1).get_layers()):
            l.set_weights(
                self.model.shared_model.get_layer(index=5).get_layer(index=1).get_layer(index=i).get_weights())

        self.move_replace_target()
        self.act_replace_target()
