#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: DQN.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import os
import argparse
import cv2
import tensorflow as tf
import config
import numpy as np
import random as rd
from tensorflow.contrib import slim
from config import *
from evaluator import Evaluator, play_one_episode
from functools import lru_cache, partial

from tensorpack import *
import sys
import datetime
import os

if os.name == 'nt':
    sys.path.insert(0, '../../build/Release')
else:
    sys.path.insert(0, '../../build.linux')
sys.path.insert(0, '../..')
from DQNModel import Model as DQNModel
from expreplay import ExpReplay
from env import Env
# from evaluator import Evaluator
# import tensorflow.contrib.slim as slim
from tensorpack.utils.gpu import get_nr_gpu
from datasets.factory import get_imdb
from vgg16 import *


def res_fc_block(inputs, units, stack=3):
    residual = inputs
    for i in range(stack):
        residual = FullyConnected('fc%d' % i, residual, units, activation=tf.nn.relu)
    x = inputs
    if inputs.shape[1].value != units:
        x = FullyConnected('fc', x, units, activation=tf.nn.relu)
    return tf.contrib.layers.layer_norm(residual + x, scale=False)


BATCH_SIZE = 16
STATE_SHAPE = (224, 224, 3)
UPDATE_FREQ = 1

GAMMA = 0.9

MEMORY_SIZE = 1e3
INIT_MEMORY_SIZE = MEMORY_SIZE // 5
STEPS_PER_EPOCH = 479 * MAX_STEP // UPDATE_FREQ
EVAL_EPISODE = 150

ROM_FILE = None
METHOD = None


@lru_cache(maxsize=32)
def get_trainset():
    train_db = get_imdb('voc_2012_trainval')
    train_db._image_index = train_db._load_image_set_index('aeroplane')
    train_bboxes = []
    train_imgs = []
    single_cnt = 0
    for idx in train_db.image_index:
        if train_db._load_pascal_annotation(idx)['boxes'].shape[0] == 1:
            single_cnt += 1
            # BGR
            img = cv2.imread(train_db.image_path_from_index(idx))
            train_imgs.append(img)
            width, height = img.shape[1], img.shape[0]
            bbox = train_db._load_pascal_annotation(idx)['boxes'][0]
            # YXYX normalized
            train_bboxes.append([bbox[1] / height, bbox[0] / width, bbox[3] / height, bbox[2] / width])
    # print(single_cnt)
    return train_imgs, train_bboxes


@lru_cache(maxsize=32)
def get_testset():
    test_db = get_imdb('voc_2007_test')
    test_db._image_index = test_db._load_image_set_index('aeroplane')
    test_bboxes = []
    test_imgs = []
    single_cnt = 0
    for idx in test_db.image_index:
        if test_db._load_pascal_annotation(idx)['boxes'].shape[0] == 1:
            single_cnt += 1
            # BGR
            img = cv2.imread(test_db.image_path_from_index(idx))
            test_imgs.append(img)
            width, height = img.shape[1], img.shape[0]
            bbox = test_db._load_pascal_annotation(idx)['boxes'][0]
            # YXYX normalized
            test_bboxes.append([bbox[1] / height, bbox[0] / width, bbox[3] / height, bbox[2] / width])
    return test_imgs, test_bboxes


def get_player(test=False):
    if test:
        return Env(*get_testset())
    else:
        return Env(*get_trainset())


class Model(DQNModel):
    def __init__(self):
        super(Model, self).__init__(STATE_SHAPE, METHOD, GAMMA)

    def _get_DQN_prediction(self, features, history, num_actions):
        # fc
        with tf.variable_scope('fc'):
            fc1 = slim.dropout(slim.fully_connected(tf.concat([slim.flatten(features), history], axis=1), 1024),
                               keep_prob=0.8, is_training=get_current_tower_context().is_training)
            fc2 = slim.dropout(slim.fully_connected(fc1, 1024), keep_prob=0.8,
                               is_training=get_current_tower_context().is_training)

        if self.method != 'Dueling':
            Q = FullyConnected('fct', fc2, num_actions)
        else:
            # Dueling DQN
            V = FullyConnected('fctV', fc2, 1)
            As = FullyConnected('fctA', fc2, num_actions)
            Q = tf.add(As, V - tf.reduce_mean(As, 1, keep_dims=True))
        return tf.identity(Q, name='Qvalue')


def get_config():
    expreplay = ExpReplay(
        predictor_io_names=(['state', 'history'], ['stage1/Qvalue']),
        predictor_refine_io_names=(['state_refine', 'history_refine'], ['stage2/Qvalue']),
        env=get_player(test=False),
        state_shape=STATE_SHAPE,
        batch_size=BATCH_SIZE,
        memory_size=MEMORY_SIZE,
        init_memory_size=INIT_MEMORY_SIZE,
        init_exploration=1.,
        update_frequency=UPDATE_FREQ
    )

    # ds = FakeData([(2, 2, *STATE_SHAPE), [2], [2], [2], [2]], dtype=['float32', 'int64', 'float32', 'bool', 'bool'])
    # ds = PrefetchData(ds, nr_prefetch=6, nr_proc=2)
    return AutoResumeTrainConfig(
        data=QueueInput(expreplay),
        model=Model(),
        callbacks=[
            Evaluator(EVAL_EPISODE, ['state', 'history'], ['stage1/Qvalue'],
                      ['state_refine', 'history_refine'], ['stage2/Qvalue'], partial(get_player, True)),
            ModelSaver(),
            PeriodicTrigger(
                RunOp(DQNModel.update_target_param, verbose=True),
                every_k_steps=STEPS_PER_EPOCH // 10),  # update target network every 10k steps
            expreplay,
            # ScheduledHyperParamSetter('learning_rate',
            #                           [(60, 4e-4), (100, 2e-4)]),
            ScheduledHyperParamSetter(
                ObjAttrParam(expreplay, 'exploration'),
                [(0, 1.), (9, 0.1)],  # 1->0.1 in the first million steps
                interp='linear'),
            HumanHyperParamSetter('learning_rate'),
        ],
        # session_init=SaverRestore("/home/neil/PycharmProjects/RPN-RL-master/save/resnet_v2_50.ckpt"),
        steps_per_epoch=STEPS_PER_EPOCH,
        max_epoch=1000,
    )


# TODO: difference between the paper and ours:
# epoch steps slightly varies
# double DQN
# train VGG16 or use pre-trained network?
# train and test # of images not the same
# initial memory size: stage-1 1000, stage-2 5000
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--task', help='task to perform',
                        choices=['play', 'eval', 'train'], default='train')
    parser.add_argument('--algo', help='algorithm',
                        choices=['DQN', 'Double', 'Dueling'], default='Double')
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    METHOD = args.algo

    nr_gpu = get_nr_gpu()
    train_tower = list(range(nr_gpu))
    if args.task != 'train':
        assert args.load is not None
        pred = OfflinePredictor(PredictConfig(
            model=Model(),
            session_init=get_model_loader(args.load),
            input_names=['state', 'history'],
            output_names=['Qvalue']))
        for i in range(1000):
            play_one_episode(get_player(test=True), pred, True)
    else:
        logger.set_logger_dir(
            os.path.join('train_log', 'RPN-RL'))
        cf = get_config()
        if args.load:
            cf.session_init = get_model_loader(args.load)
        trainer = SimpleTrainer() if nr_gpu == 1 else AsyncMultiGPUTrainer(train_tower)
        launch_train_with_config(cf, trainer)
