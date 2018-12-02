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
        self.num_actions = (len(Env.action_space), len(Env.action_space_refine))
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
                tf.placeholder(tf.float32, (None, 2, config.HISTORY_LEN * self.num_actions[0]), 'joint_history'),
                tf.placeholder(tf.float32,
                               (None, 2, *self.state_shape),
                               'joint_state_refine'),
                tf.placeholder(tf.int64, (None,), 'action_refine'),
                tf.placeholder(tf.float32, (None,), 'reward_refine'),
                tf.placeholder(tf.bool, (None,), 'isOver_refine'),
                tf.placeholder(tf.float32, (None, 2, config.HISTORY_LEN * self.num_actions[1]), 'joint_history_refine')
                ]

    @abc.abstractmethod
    def _get_DQN_prediction(self, state, history, num_actions):
        pass

    @auto_reuse_variable_scope
    def get_DQN_prediction(self, state, history, num_actions):
        return self._get_DQN_prediction(state, history, num_actions)

    @auto_reuse_variable_scope
    def get_features(self, state):
        # BGR mean
        mean = tf.constant([103.939, 123.68, 116.779], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
        image = state - mean
        with tf.variable_scope('vgg16'):
            features = vgg_conv(image)
        return features

    # joint state: B * 2 * 224 * 224 * 3
    # dynamic action range
    def build_graph(self, joint_state, action, reward, isOver, joint_history,
                    joint_state_refine, action_refine, reward_refine, isOver_refine, joint_history_refine):
        # vgg are shared among stage-1 and stage-2
        state = self.get_features(tf.identity(joint_state[:, 0, ...], name='state'))
        history = tf.identity(joint_history[:, 0, ...], name='history')
        next_state = self.get_features(tf.identity(joint_state[:, 1, ...], name='next_state'))
        next_history = tf.identity(joint_history[:, 1, ...], name='next_history')

        state_refine = self.get_features(tf.identity(joint_state_refine[:, 0, ...], name='state_refine'))
        history_refine = tf.identity(joint_history_refine[:, 0, ...], name='history_refine')
        next_state_refine = self.get_features(tf.identity(joint_state_refine[:, 1, ...], name='next_state_refine'))
        next_history_refine = tf.identity(joint_history_refine[:, 1, ...], name='next_history_refine')

        total_cost = []
        for i, data in enumerate([[state, history, next_state, next_history, action, reward, isOver],
                                  [state_refine, history_refine, next_state_refine, next_history_refine, action_refine, reward_refine, isOver_refine]]):
            with tf.variable_scope('stage%d' % (i + 1)):
                st, hist, next_st, next_hist, act, rw, over = data

                predict_value = self.get_DQN_prediction(st, hist, self.num_actions[i])
                if not get_current_tower_context().is_training:
                    if i == 0:
                        continue
                    elif i == 1:
                        return

                action_onehot = tf.one_hot(act, self.num_actions[i], 1.0, 0.0)

                pred_action_value = tf.reduce_sum(predict_value * action_onehot, 1)  # N,
                # max_pred_reward = tf.reduce_mean(pred_action_value, name='predict_reward')
                max_pred_reward = tf.reduce_mean(tf.reduce_max(
                   predict_value, 1), name='predict_reward')
                summary.add_moving_summary(max_pred_reward)

                with tf.variable_scope('target'), varreplace.freeze_variables(skip_collection=True):
                    targetQ_predict_value = self.get_DQN_prediction(next_st, next_hist, self.num_actions[i])    # NxA

                if self.method != 'Double':
                    # DQN
                    best_v = tf.reduce_max(targetQ_predict_value, 1)    # N,
                else:
                    # Double-DQN
                    next_predict_value = self.get_DQN_prediction(next_st, next_hist, self.num_actions[i])
                    greedy_choice = tf.argmax(next_predict_value, 1)   # N,
                    predict_onehot = tf.one_hot(greedy_choice, self.num_actions[i], 1.0, 0.0)
                    best_v = tf.reduce_sum(targetQ_predict_value * predict_onehot, 1)

                target = rw + (1.0 - tf.cast(over, tf.float32)) * self.gamma * tf.stop_gradient(best_v)
                average_target = tf.reduce_mean(target, name='average_target')
                # average_target = tf.Print(average_target, [], name='average_target')
                cost = tf.losses.mean_squared_error(
                    target, pred_action_value, reduction=tf.losses.Reduction.MEAN)
                total_cost.append(cost)

                summary.add_moving_summary(cost)
                summary.add_moving_summary(average_target)
        return tf.add_n(total_cost)

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