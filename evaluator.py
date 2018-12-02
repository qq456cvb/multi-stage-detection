import random
import time
import multiprocessing
from tqdm import tqdm
from six.moves import queue

from tensorpack.utils.concurrency import StoppableThread, ShareSessionThread
from tensorpack.callbacks import Callback
from tensorpack.utils import logger
from tensorpack.utils.stats import StatCounter
from tensorpack.utils.utils import get_tqdm_kwargs

import config
import sys
import os
import cv2
import numpy as np
import tensorflow as tf


def play_one_episode(env, func, num_actions, verbose=False):

    def pad_state(state):
        # since out net uses max operation, we just dup the last row and keep the result same
        newstates = []
        for s in state:
            assert s.shape[0] <= num_actions[1]
            s = np.concatenate([s, np.repeat(s[-1:, :], num_actions[1] - s.shape[0], axis=0)], axis=0)
            newstates.append(s)
        newstates = np.stack(newstates, axis=0)
        if len(state) < num_actions[0]:
            state = np.concatenate([newstates, np.repeat(newstates[-1:, :, :], num_actions[0] - newstates.shape[0], axis=0)], axis=0)
        else:
            state = newstates
        return state

    def pad_action_space(available_actions):
        # print(available_actions)
        for i in range(len(available_actions)):
            available_actions[i] += [available_actions[i][-1]] * (num_actions[1] - len(available_actions[i]))
        if len(available_actions) < num_actions[0]:
            available_actions.extend([available_actions[-1]] * (num_actions[0] - len(available_actions)))

    def subsample_combs_masks(combs, masks, num_sample):
        if masks is not None:
            assert len(combs) == masks.shape[0]
        idx = np.random.permutation(len(combs))[:num_sample]
        return [combs[i] for i in idx], (masks[idx] if masks is not None else None)

    def get_state_and_action_space(is_comb, cand_state=None, cand_actions=None, action=None):
        if is_comb:
            available_actions = []
            state = []
            # first diagonal
            for i in range(config.NUM_SPLITS_DIAGONAL):
                # XY
                split_pt = [1. / (config.NUM_SPLITS_DIAGONAL + 1) * (i + 1)] * 2
                focus = [[0, 0, *split_pt],
                         [0, split_pt[0], split_pt[1], 1.],
                         [split_pt[1], 0, 1., split_pt[0]],
                         [split_pt[1], split_pt[0], 1., 1.],
                         'trigger']
                available_actions.append(focus)
                # average pooling
                # focus_features = np.array(
                #     [np.mean(env.get_focus(focus[j]), axis=(0, 1)) for j in range(len(focus))])
                focus_features = np.array([np.reshape(env.get_focus(focus[j]), [-1]) for j in range(len(focus))])
                state.append(focus_features)
            # second diagonal
            for i in range(config.NUM_SPLITS_DIAGONAL):
                # XY
                split_pt = [1. - 1. / (config.NUM_SPLITS_DIAGONAL + 1) * (i + 1),
                            1. / (config.NUM_SPLITS_DIAGONAL + 1) * (i + 1)]
                focus = [[0, 0, *split_pt],
                         [0, split_pt[0], split_pt[1], 1.],
                         [split_pt[1], 0, 1., split_pt[0]],
                         [split_pt[1], split_pt[0], 1., 1.],
                         'trigger']
                available_actions.append(focus)
                # average pooling
                # focus_features = np.array(
                #     [np.mean(env.get_focus(focus[j]), axis=(0, 1)) for j in range(len(focus))])
                focus_features = np.array([np.reshape(env.get_focus(focus[j]), [-1]) for j in range(len(focus))])
                state.append(focus_features)
            pad_action_space(available_actions)
            state = pad_state(state)
            assert state.shape[0] == num_actions[0] and state.shape[1] == num_actions[1]
        else:
            available_actions = cand_actions[action]
            state = cand_state[action:action+1, :, :]
            state = np.repeat(state, num_actions[0], axis=0)
            assert state.shape[0] == num_actions[0] and state.shape[1] == num_actions[1]
        return state, available_actions

    env.reset()

    r = 0
    done = False
    while not done:
        # first hierarchy
        state, available_actions = get_state_and_action_space(True, env)
        q_values = func([state[None, :, :, :], np.array([True])])[0][0]
        action = np.argmax(q_values)
        # action = np.random.randint(q_values.size)
        # clamp action to valid range
        action = min(action, num_actions[0] - 1)

        # second hierarchy
        state, available_actions = get_state_and_action_space(False, state, available_actions, action)
        q_values = func([state[None, :, :, :], np.array([False])])[0][0]
        action = np.argmax(q_values)
        # action = np.random.randint(q_values.size)
        # clamp action to valid range
        action = min(action, num_actions[1] - 1)

        # intention
        intention = available_actions[action]
        reward, done = env.step(intention)
        if verbose:
            print(intention)
            print(reward)
            img = env.image.copy()

            def enlarge(bbox):
                w = bbox[3] - bbox[1]
                h = bbox[2] - bbox[0]
                return [np.clip(bbox[0] - 0.1 * h, 0, 1), np.clip(bbox[1] - 0.1 * w, 0, 1),
                        np.clip(bbox[2] + 0.1 * h, 0, 1), np.clip(bbox[3] + 0.1 * w, 0, 1)]

            # bbox = enlarge(env.target_bbox)
            # cv2.rectangle(img, (int(bbox[1] * img.shape[1]), int(bbox[0] * img.shape[0])),
            #               (int(bbox[3] * img.shape[1]), int(bbox[2] * img.shape[0])), (0, 0, 255),
            #               2)
            cv2.rectangle(img, (int(env.target_bbox[1] * img.shape[1]), int(env.target_bbox[0] * img.shape[0])),
                          (int(env.target_bbox[3] * img.shape[1]), int(env.target_bbox[2] * img.shape[0])), (0, 255, 0), 2)
            cv2.rectangle(img, (int(env.crt_bbox[1] * img.shape[1]), int(env.crt_bbox[0] * img.shape[0])),
                          (int(env.crt_bbox[3] * img.shape[1]), int(env.crt_bbox[2] * img.shape[0])), (255, 0, 0), 2)
            cv2.imshow('image', img)
            cv2.waitKey(0)
        r += reward
    return r


def eval_with_funcs(predictors, nr_eval, get_player_fn, num_actions, verbose=False):
    """
    Args:
        predictors ([PredictorBase])
    """

    class Worker(StoppableThread, ShareSessionThread):
        def __init__(self, func, queue):
            super(Worker, self).__init__()
            self._func = func
            self.q = queue

        def func(self, *args, **kwargs):
            if self.stopped():
                raise RuntimeError("stopped!")
            return self._func(*args, **kwargs)

        def run(self):
            with self.default_sess():
                player = get_player_fn()
                while not self.stopped():
                    try:
                        val = play_one_episode(player, self.func, num_actions)
                    except RuntimeError:
                        return
                    self.queue_put_stoppable(self.q, val)

    q = queue.Queue()
    threads = [Worker(f, q) for f in predictors]

    for k in threads:
        k.start()
        time.sleep(0.1)  # avoid simulator bugs
    stat = StatCounter()

    def fetch():
        val = q.get()
        stat.feed(val)
        if verbose:
            logger.info("reward %f" % val)

    for _ in tqdm(range(nr_eval), **get_tqdm_kwargs()):
        fetch()
    logger.info("Waiting for all the workers to finish the last run...")
    for k in threads:
        k.stop()
    for k in threads:
        k.join()
    while q.qsize():
        fetch()
    farmer_win_rate = stat.average
    return farmer_win_rate


class Evaluator(Callback):
    def __init__(self, nr_eval, input_names, output_names, num_actions, get_player_fn):
        self.eval_episode = nr_eval
        self.input_names = input_names
        self.output_names = output_names
        self.get_player_fn = get_player_fn
        self.num_actions = num_actions

    def _setup_graph(self):
        # self.lord_win_rate = tf.get_variable('lord_win_rate', shape=[], initializer=tf.constant_initializer(0.),
        #                trainable=False)
        nr_proc = min(multiprocessing.cpu_count() // 2, 20)
        self.pred_funcs = [self.trainer.get_predictor(
            self.input_names, self.output_names)] * nr_proc

    def _before_train(self):
        t = time.time()
        r = eval_with_funcs(
            self.pred_funcs, self.eval_episode, self.get_player_fn, self.num_actions, verbose=False)
        t = time.time() - t
        logger.info("evaluated reward: {}".format(r))
        # self.lord_win_rate.load(1 - farmer_win_rate)
        # if t > 10 * 60:  # eval takes too long
        #     self.eval_episode = int(self.eval_episode * 0.94)

    def _trigger_epoch(self):
        t = time.time()
        r = eval_with_funcs(
            self.pred_funcs, self.eval_episode, self.get_player_fn, self.num_actions, verbose=False)
        t = time.time() - t
        if t > 10 * 60:  # eval takes too long
            self.eval_episode = int(self.eval_episode * 0.94)
        self.trainer.monitors.put_scalar('evaluated reward', r)