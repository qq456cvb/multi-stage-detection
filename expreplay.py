#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: expreplay.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>
# Adapted by: Neil You for Fight the Lord

import numpy as np
import config
from collections import deque, namedtuple
import threading
from six.moves import queue, range

from tensorpack.dataflow import DataFlow, RNGDataFlow
from tensorpack.utils import logger
from tensorpack.utils.utils import get_tqdm, get_rng
from tensorpack.utils.stats import StatCounter
from tensorpack.utils.concurrency import LoopThread, ShareSessionThread
from tensorpack.callbacks.base import Callback
from env import Env
import sys
import os

__all__ = ['ExpReplay']

Experience = namedtuple('Experience',
                        ['joint_state', 'action', 'reward', 'isOver'])


class ReplayMemory(object):
    def __init__(self, max_size, state_shape):
        self.max_size = int(max_size)
        self.state_shape = state_shape

        self.state = np.zeros((self.max_size,) + state_shape, dtype='float32')
        self.action = np.zeros((self.max_size,), dtype='int32')
        self.reward = np.zeros((self.max_size,), dtype='float32')
        self.isOver = np.zeros((self.max_size,), dtype='bool')

        self._curr_size = 0
        self._curr_pos = 0

    def append(self, exp):
        """
        Args:
            exp (Experience):
        """
        if self._curr_size < self.max_size:
            self._assign(self._curr_pos, exp)
            self._curr_pos = (self._curr_pos + 1) % self.max_size
            self._curr_size += 1
        else:
            self._assign(self._curr_pos, exp)
            self._curr_pos = (self._curr_pos + 1) % self.max_size

    def sample(self, idx):
        """ return a tuple of (s,r,a,o),
            where s is of shape STATE_SIZE + (2,)"""
        idx = (self._curr_pos + idx) % self._curr_size
        action = self.action[idx]
        reward = self.reward[idx]
        isOver = self.isOver[idx]
        if idx + 2 <= self._curr_size:
            state = self.state[idx:idx+2]
        else:
            end = idx + 2 - self._curr_size
            state = self._slice(self.state, idx, end)
        return state, action, reward, isOver

    def _slice(self, arr, start, end):
        s1 = arr[start:]
        s2 = arr[:end]
        return np.concatenate((s1, s2), axis=0)

    def __len__(self):
        return self._curr_size

    def _assign(self, pos, exp):
        self.state[pos] = exp.joint_state
        self.action[pos] = exp.action
        self.reward[pos] = exp.reward
        self.isOver[pos] = exp.isOver


class ExpReplay(RNGDataFlow, Callback):
    """
    Implement experience replay in the paper
    `Human-level control through deep reinforcement learning
    <http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html>`_.
    This implementation provides the interface as a :class:`DataFlow`.
    This DataFlow is __not__ fork-safe (thus doesn't support multiprocess prefetching).
    This implementation assumes that state is
    batch-able, and the network takes batched inputs.
    """

    def __init__(self,
                 predictor_io_names,
                 env,
                 state_shape,
                 batch_size,
                 memory_size, init_memory_size,
                 init_exploration,
                 update_frequency):
        """
        Args:
            predictor_io_names (tuple of list of str): input/output names to
                predict Q value from state.
            player (RLEnvironment): the player.
            history_len (int): length of history frames to concat. Zero-filled
                initial frames.
            update_frequency (int): number of new transitions to add to memory
                after sampling a batch of transitions for training.
        """
        init_memory_size = int(init_memory_size)

        items = locals().items()
        for k, v in items:
            if k != 'self':
                setattr(self, k, v)
        self.exploration = init_exploration
        self.env = env
        logger.info("Number of Legal actions: {}, {}".format(*self.num_actions))

        self.rng = get_rng(self)
        # print('RNG------------------------------------------', self.rng.randint(10))
        self._init_memory_flag = threading.Event()  # tell if memory has been initialized

        # a queue to receive notifications to populate memory
        self._populate_job_queue = queue.Queue(maxsize=5)

        self.mem = ReplayMemory(memory_size, state_shape)
        self.env.reset()
        self._current_ob = self.env.focus_image
        # stage 1 ar actions
        self._action_space = self.env.action_space
        self.num_actions = len(self._action_space)
        self._player_scores = StatCounter()
        self._current_game_score = StatCounter()
        self.state_shape = state_shape

    def get_simulator_thread(self):
        # spawn a separate thread to run policy
        def populate_job_func():
            self._populate_job_queue.get()
            for _ in range(self.update_frequency):
                self._populate_exp()
        th = ShareSessionThread(LoopThread(populate_job_func, pausable=False))
        th.name = "SimulatorThread"
        return th

    def _init_memory(self):
        logger.info("Populating replay memory with epsilon={} ...".format(self.exploration))

        with get_tqdm(total=self.init_memory_size) as pbar:
            while len(self.mem) < self.init_memory_size:
                self._populate_exp()
                pbar.update()
        self._init_memory_flag.set()

    def _populate_exp(self):
        """ populate a transition by epsilon-greedy"""
        old_s = self._current_ob
        # forced termination
        if self.env.crt_iou > 0.5:
            act = self._action_space.index('trigger')
        else:
            if self.rng.rand() <= self.exploration:
                act = self.rng.choice(range(self.num_actions))
            else:
                q_values = self.predictor(old_s[None, ...])[0][0]
                act = np.argmax(q_values)
                # clamp action to valid range
                act = min(act, self.num_actions - 1)

        reward, isOver = self.env.step(self._action_space[act])
        self._current_game_score.feed(reward)

        if isOver:
            # print('lord wins' if reward > 0 else 'farmer wins')
            # print(self._current_game_score.sum)
            self._player_scores.feed(self._current_game_score.sum)
            self.env.reset()
            self._current_game_score.reset()
        self._current_ob = self.env.focus_image
        self.mem.append(Experience(old_s, act, reward, isOver))

    def debug(self, cnt=100000):
        with get_tqdm(total=cnt) as pbar:
            for i in range(cnt):
                self.mem.append(Experience(np.zeros([self.num_actions, self.num_actions, 256]), 0, 0, False, True if i % 2 == 0 else False))
                # self._current_ob, self._action_space = self.get_state_and_action_spaces(None)
                pbar.update()

    def get_data(self):
        # wait for memory to be initialized
        self._init_memory_flag.wait()

        while True:
            idx = self.rng.randint(
                self._populate_job_queue.maxsize * self.update_frequency,
                len(self.mem) - 1,
                size=self.batch_size)
            batch_exp = [self.mem.sample(i) for i in idx]

            yield self._process_batch(batch_exp)
            self._populate_job_queue.put(1)

    def _process_batch(self, batch_exp):
        state = np.asarray([e[0] for e in batch_exp], dtype='float32')
        # print('state',np.array(state).shape)
        action = np.asarray([e[1] for e in batch_exp], dtype='int32')
        # print('action',action)
        reward = np.asarray([e[2] for e in batch_exp], dtype='float32')
        # print('reward',reward)
        isOver = np.asarray([e[3] for e in batch_exp], dtype='bool')
        # print('isOver',isOver)
        return [state, action, reward, isOver]

    def _setup_graph(self):
        self.predictor = self.trainer.get_predictor(*self.predictor_io_names)

    def _before_train(self):
        self._init_memory()
        self._simulator_th = self.get_simulator_thread()
        self._simulator_th.start()

    def _trigger(self):
        v = self._player_scores
        try:
            mean, max = v.average, v.max
            self.trainer.monitors.put_scalar('expreplay/mean_score', mean)
            self.trainer.monitors.put_scalar('expreplay/max_score', max)
        except Exception:
            logger.exception("Cannot log training scores.")
        v.reset()


if __name__ == '__main__':
    def predictor(x):
        return [np.random.random([1, 4])]
    env = Env(None, [0.5, 0.5, 0.6, 0.6])
    E = ExpReplay(
        predictor_io_names=(['state', 'comb_mask'], ['Qvalue']),
        env=env,
        state_shape=(config.NUM_SPLITS_DIAGONAL * 2, 4, 7 * 7 * 2048),
        num_actions=[config.NUM_SPLITS_DIAGONAL * 2, 4],
        batch_size=16,
        memory_size=1e2,
        init_memory_size=1e2,
        init_exploration=0.,
        update_frequency=4
    )
    E.predictor = predictor
    E._init_memory()