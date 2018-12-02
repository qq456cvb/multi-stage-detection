#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: DQNModel.py


import abc
import tensorflow as tf
from tensorflow.contrib import slim
import tensorpack
from tensorpack import ModelDesc
from tensorpack.utils import logger
from tensorpack.tfutils import (
    varreplace, summary, get_current_tower_context, optimizer, gradproc)
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
from vgg16 import *
from tensorflow.contrib.slim.nets import resnet_utils
from env import Env
import config


class Model(ModelDesc):
    learning_rate = 1e-6

    def __init__(self, state_shape, method, gamma):
        self.state_shape = state_shape
        self.method = method
        self.num_actions = len(Env.action_space)
        self.gamma = gamma

    def inputs(self):
        # Use a combined state for efficiency.
        # The first h channels are the current state, and the last h channels are the next state.
        return [tf.placeholder(tf.float32,
                               (None, 2, *self.state_shape),
                               'joint_state'),
                tf.placeholder(tf.int64, (None,), 'action'),
                tf.placeholder(tf.float32, (None,), 'reward'),
                tf.placeholder(tf.bool, (None,), 'isOver'),
                tf.placeholder(tf.float32, (None, 2, config.HISTORY_LEN * self.num_actions), 'joint_history')]

    @abc.abstractmethod
    def _get_DQN_prediction(self, state, history):
        pass

    @auto_reuse_variable_scope
    def get_DQN_prediction(self, state, history):
        return self._get_DQN_prediction(state, history)

    # joint state: B * 2 * 224 * 224 * 3
    # dynamic action range
    def build_graph(self, joint_state, action, reward, isOver, joint_history):
        state = tf.identity(joint_state[:, 0, ...], name='state')
        history = tf.identity(joint_history[:, 0, ...], name='history')
        next_state = tf.identity(joint_state[:, 1, ...], name='next_state')
        next_history = tf.identity(joint_history[:, 1, ...], name='next_history')

        self.predict_value = self.get_DQN_prediction(state, history)
        if not get_current_tower_context().is_training:
            return

        action_onehot = tf.one_hot(action, self.num_actions, 1.0, 0.0)

        pred_action_value = tf.reduce_sum(self.predict_value * action_onehot, 1)  # N,
        # max_pred_reward = tf.reduce_mean(pred_action_value, name='predict_reward')
        max_pred_reward = tf.reduce_mean(tf.reduce_max(
           self.predict_value, 1), name='predict_reward')
        summary.add_moving_summary(max_pred_reward)

        with tf.variable_scope('target'), varreplace.freeze_variables(skip_collection=True):
            targetQ_predict_value = self.get_DQN_prediction(next_state, next_history)    # NxA

        if self.method != 'Double':
            # DQN
            best_v = tf.reduce_max(targetQ_predict_value, 1)    # N,
        else:
            # Double-DQN
            next_predict_value = self.get_DQN_prediction(next_state, next_history)
            self.greedy_choice = tf.argmax(next_predict_value, 1)   # N,
            predict_onehot = tf.one_hot(self.greedy_choice, self.num_actions, 1.0, 0.0)
            best_v = tf.reduce_sum(targetQ_predict_value * predict_onehot, 1)

        target = reward + (1.0 - tf.cast(isOver, tf.float32)) * self.gamma * tf.stop_gradient(best_v)
        average_target = tf.reduce_mean(target, name='average_target')
        # average_target = tf.Print(average_target, [], name='average_target')
        cost = tf.losses.mean_squared_error(
            target, pred_action_value, reduction=tf.losses.Reduction.MEAN)

        summary.add_moving_summary(cost)
        summary.add_moving_summary(average_target)
        return cost

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=Model.learning_rate, trainable=False)
        # opt = tf.train.AdamOptimizer(lr, epsilon=1e-3)
        opt = tf.train.AdamOptimizer(lr)
        return optimizer.apply_grad_processors(
            opt, [gradproc.MapGradient(lambda grad: tf.clip_by_average_norm(grad, 0.5)),
                  gradproc.SummaryGradient()])

    @staticmethod
    def update_target_param():
        vars = tf.global_variables()
        ops = []
        G = tf.get_default_graph()
        for v in vars:
            target_name = v.op.name
            if target_name.startswith('target'):
                new_name = target_name.replace('target/', '')
                logger.info("Target Network Update: {} <- {}".format(target_name, new_name))
                ops.append(v.assign(G.get_tensor_by_name(new_name + ':0')))
        return tf.group(*ops, name='update_target_network')