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


def play_one_episode(env, func, func_refine, verbose=False):
    env.reset()

    r = 0
    done = False
    while not done:
        # first hierarchy
        state, history = env.focus_image, env.history
        q_values = func(state[None, ...], history.reshape(1, -1))[0][0]
        action = np.argmax(q_values)

        if env.action_space[action] != 'trigger':
            env.step(env.action_space[action])
            refine_stop = False
            while not refine_stop:
                state_refine, history_refine = env.focus_image, env.history_refine
                q_values = func_refine(state_refine[None, ...], history_refine.reshape(1, -1))[0][0]
                act_refine = np.argmax(q_values)
                reward_refine, refine_stop = env.step_refine(env.action_space_refine[act_refine])
            reward, done = env.step_post()
        else:
            reward, done = env.step(env.action_space[action])
        if verbose:
            print(env.action_space[action])
            print(reward)
            img = env.image.copy()

            cv2.rectangle(img, (int(env.target_bbox[1] * img.shape[1]), int(env.target_bbox[0] * img.shape[0])),
                          (int(env.target_bbox[3] * img.shape[1]), int(env.target_bbox[2] * img.shape[0])), (0, 255, 0), 2)
            cv2.rectangle(img, (int(env.crt_bbox[1] * img.shape[1]), int(env.crt_bbox[0] * img.shape[0])),
                          (int(env.crt_bbox[3] * img.shape[1]), int(env.crt_bbox[2] * img.shape[0])), (255, 0, 0), 2)
            cv2.imshow('image', img)
            cv2.waitKey(0)
        r += reward
    return r, float(env.iou > 0.5)


def eval_with_funcs(predictors, nr_eval, get_player_fn, verbose=False):
    """
    Args:
        predictors ([PredictorBase])
    """

    class Worker(StoppableThread, ShareSessionThread):
        def __init__(self, func, func_refine, queue):
            super(Worker, self).__init__()
            self._func = func
            self._func_refine = func_refine
            self.q = queue

        def func(self, *args, **kwargs):
            if self.stopped():
                raise RuntimeError("stopped!")
            return self._func(*args, **kwargs)

        def func_refine(self, *args, **kwargs):
            if self.stopped():
                raise RuntimeError("stopped!")
            return self._func_refine(*args, **kwargs)

        def run(self):
            with self.default_sess():
                player = get_player_fn()
                while not self.stopped():
                    try:
                        r, tp = play_one_episode(player, self.func, self.func_refine, verbose)
                    except RuntimeError:
                        return
                    self.queue_put_stoppable(self.q, (r, tp))

    q = queue.Queue()
    threads = [Worker(f[0], f[1], q) for f in predictors]

    for k in threads:
        k.start()
        time.sleep(0.1)  # avoid simulator bugs
    stat_r = StatCounter()
    stat_tp = StatCounter()

    def fetch():
        r, tp = q.get()
        stat_r.feed(r)
        stat_tp.feed(tp)
        if verbose:
            logger.info("reward %f" % r)
            logger.info("true positive %f" % tp)

    for _ in tqdm(range(nr_eval), **get_tqdm_kwargs()):
        fetch()
    logger.info("Waiting for all the workers to finish the last run...")
    for k in threads:
        k.stop()
    for k in threads:
        k.join()
    while q.qsize():
        fetch()
    return stat_r.average, stat_tp.average


class Evaluator(Callback):
    def __init__(self, nr_eval, input_names, output_names,
                 input_names_refine, output_names_refine, get_player_fn):
        self.eval_episode = nr_eval
        self.input_names = input_names
        self.output_names = output_names
        self.input_names_refine = input_names_refine
        self.output_names_refine = output_names_refine
        self.get_player_fn = get_player_fn

    def _setup_graph(self):
        nr_proc = min(multiprocessing.cpu_count() // 2, 20)
        self.pred_funcs = [(self.trainer.get_predictor(
            self.input_names, self.output_names), self.trainer.get_predictor(
            self.input_names_refine, self.output_names_refine))] * nr_proc

    def _before_train(self):
        t = time.time()
        r, tp = eval_with_funcs(
            self.pred_funcs, self.eval_episode, self.get_player_fn, verbose=False)
        t = time.time() - t
        logger.info("evaluated reward: {}, evaluated tp: {}".format(r, tp))

    def _trigger_epoch(self):
        t = time.time()
        r, tp = eval_with_funcs(
            self.pred_funcs, self.eval_episode, self.get_player_fn, verbose=False)
        t = time.time() - t
        if t > 10 * 60:  # eval takes too long
            self.eval_episode = int(self.eval_episode * 0.94)
        self.trainer.monitors.put_scalar('evaluated reward', r)
        self.trainer.monitors.put_scalar('evaluated tp', tp)