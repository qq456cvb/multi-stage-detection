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
                        ['joint_state', 'action', 'reward', 'isOver', 'history'])


class ReplayMemory(object):
    def __init__(self, max_size, state_shape):
        self.max_size = int(max_size)
        self.state_shape = state_shape

        self.state = np.zeros((self.max_size,) + state_shape, dtype='float32')
        self.action = np.zeros((self.max_size,), dtype='int32')
        self.reward = np.zeros((self.max_size,), dtype='float32')
        self.isOver = np.zeros((self.max_size,), dtype='bool')
        self.history = np.zeros((self.max_size, config.HISTORY_LEN * len(Env.action_space)), dtype='float32')

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
            history = self.history[idx:idx+2]
        else:
            end = idx + 2 - self._curr_size
            state = self._slice(self.state, idx, end)
            history = self._slice(self.history, idx, end)
        return state, action, reward, isOver, history

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
        self.history[pos] = exp.history


class ReplayMemoryRefine(ReplayMemory):
    def __init__(self, max_size, state_shape):
        super().__init__(max_size, state_shape)
        self.history = np.zeros((self.max_size, config.HISTORY_LEN * len(Env.action_space_refine)), dtype='float32')


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
                 predictor_refine_io_names,
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

        self.rng = get_rng(self)
        # print('RNG------------------------------------------', self.rng.randint(10))
        self._init_memory_flag = threading.Event()  # tell if memory has been initialized

        # a queue to receive notifications to populate memory
        self._populate_job_queue = queue.Queue(maxsize=5)

        self.mem = ReplayMemory(memory_size, state_shape)
        self.mem_refine = ReplayMemoryRefine(memory_size, state_shape)
        self.env.reset()
        self._current_ob, self._current_history = self.env.focus_image, self.env.history
        # stage 1 ar actions
        self._action_space = self.env.action_space
        # stage 2 actions
        self._action_space_refine = self.env.action_space_refine
        logger.info("Number of Legal actions: stage-1-ar {}, stage-2 {}".format(len(self._action_space), len(self._action_space_refine)))
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
        old_s, old_history = self._current_ob, self._current_history
        # forced termination
        if self.env.iou > 0.5:
            act = -1
        else:
            if self.rng.rand() <= self.exploration:
                act = self.rng.choice(range(len(self._action_space)))
            else:
                q_values = self.predictor(old_s[None, ...], old_history.reshape(1, -1))[0][0]
                act = np.argmax(q_values)

        # stage 2
        if self._action_space[act] != 'trigger':
            self.env.step(self._action_space[act])
            refine_stop = False
            while not refine_stop:
                state_refine, history_refine = self.env.focus_image, self.env.history_refine
                if self.rng.rand() <= self.exploration:
                    act_refine = self.rng.choice(range(len(self._action_space_refine)))
                else:
                    q_values = self.predictor_refine(state_refine[None, ...], history_refine.reshape(1, -1))[0][0]
                    act_refine = np.argmax(q_values)
                reward_refine, refine_stop = self.env.step_refine(self._action_space_refine[act_refine])
                self.mem_refine.append(Experience(state_refine, act_refine, reward_refine, refine_stop, history_refine.reshape(-1)))
            reward, isOver = self.env.step_post()
        else:
            reward, isOver = self.env.step(self._action_space[act])
        self._current_game_score.feed(reward)

        if isOver:
            # print('lord wins' if reward > 0 else 'farmer wins')
            # print(self._current_game_score.sum)
            self._player_scores.feed(self._current_game_score.sum)
            self.env.reset()
            self._current_game_score.reset()
        self._current_ob, self._current_history = self.env.focus_image, self.env.history
        self.mem.append(Experience(old_s, act, reward, isOver, old_history.reshape(-1)))

    def get_data(self):
        # wait for memory to be initialized
        self._init_memory_flag.wait()

        while True:
            idx = self.rng.randint(
                self._populate_job_queue.maxsize * self.update_frequency,
                len(self.mem) - 1,
                size=self.batch_size)
            batch_exp = [self.mem.sample(i) for i in idx]
            batch_exp_refine = [self.mem_refine.sample(i) for i in idx]

            yield self._process_batch(batch_exp) + self._process_batch(batch_exp_refine)
            self._populate_job_queue.put(1)

    def _process_batch(self, batch_exp):
        state = np.asarray([e[0] for e in batch_exp], dtype='float32')
        action = np.asarray([e[1] for e in batch_exp], dtype='int32')
        reward = np.asarray([e[2] for e in batch_exp], dtype='float32')
        isOver = np.asarray([e[3] for e in batch_exp], dtype='bool')
        history = np.asarray([e[4] for e in batch_exp], dtype='float32')
        return [state, action, reward, isOver, history]

    def _setup_graph(self):
        self.predictor = self.trainer.get_predictor(*self.predictor_io_names)
        self.predictor_refine = self.trainer.get_predictor(*self.predictor_refine_io_names)

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