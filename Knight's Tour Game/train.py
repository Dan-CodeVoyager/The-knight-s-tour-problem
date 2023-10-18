# -*- coding: utf-8 -*-
"""
An implementation of the training pipeline of AlphaZero for Gomoku

@author: Junxiao Song
"""


from __future__ import print_function
import random
import sys
import matplotlib.pyplot as plt
import time

import numpy as np
from collections import defaultdict, deque

import torch

from KnightLogic import Board, Game
from Knight_MCTS import MCTSPlayer
from policy_value import PolicyValueNet

sys.setrecursionlimit(1000000)


class TrainPipeline:
    def __init__(self, init_model=None):
        # params of the board and the game
        self.board_width = 3
        self.board_height = 6
        self.board = Board(width=self.board_width, height=self.board_height)
        self.game = Game(self.board)
        # training params
        self.learn_rate = 0.002
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.temp = 1.0  # the temperature param
        self.n_playout = 400  # 400 num of simulations for each move
        self.c_puct = 5
        self.buffer_size = 30  # 10000
        self.batch_size = 16  # 512  # mini-batch size for training
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.epochs = 10  # 5 num of train_steps for each update
        self.kl_targ = 0.02  # 0.02
        self.check_freq = 2  # 50
        self.game_batch_num = 50  # 1500
        self.best_win_ratio = 0
        # num of simulations used for the pure mcts, which is used as
        # the opponent to evaluate the trained policy
        self.pure_mcts_playout_num = 5  # 1000
        if init_model:
            # start training from an initial policy-value net
            self.policy_value_net = PolicyValueNet(self.board_width, self.board_height, model_file=init_model)
        else:
            # start training from a new policy-value net
            self.policy_value_net = PolicyValueNet(self.board_width, self.board_height)

        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playout,
                                      is_selfplay=1)

    def get_equi_data(self, play_data):
        """
        augment the data set by rotation and flipping
        play_data: [(state, mcts_prob), ..., ...]
        """

        extend_data = []
        # for state, mcts_porb, winner in play_data:
        for i in range(len(play_data)):
            state = play_data[i][0]
            mcts_porb = play_data[i][1]
            label = play_data[i][2]

            equi_mcts_state_1 = np.rot90(state, k=2)  # 旋转180度
            qui_mcts_prob_1 = np.rot90(mcts_porb, k=2)

            equi_mcts_state_2 = np.flipud(np.rot90(state, k=2))  # 旋转180°后在上下翻转
            equi_mcts_prob_2 = np.flipud(np.rot90(mcts_porb, k=2))

            equi_mcts_state_3 = np.flipud(state)  # 上下翻转
            equi_mcts_prob_3 = np.flipud(mcts_porb)

            extend_data.append((equi_mcts_state_1, qui_mcts_prob_1, label))
            extend_data.append((equi_mcts_state_2, equi_mcts_prob_2, label))
            extend_data.append((equi_mcts_state_3, equi_mcts_prob_3, label))

        # print('extend_data is', extend_data)

        return extend_data

    def collect_selfplay_data(self, n_games=1):
        """collect self-play data for training"""
        for i in range(n_games):
            win, play_data = self.game.start_self_play(self.mcts_player, temp=self.temp)
            play_data = list(play_data)[:]
            # augment the data
            play_data = self.get_equi_data(play_data)
            self.data_buffer.extend(play_data)
            np.random.shuffle(self.data_buffer)

    def policy_update(self):
        """update the policy-value net"""
        global loss, entropy, kl, new_v
        mini_batch = random.sample(self.data_buffer, self.batch_size)

        # print('mini_batch', mini_batch)
        state_batch = [data[0] for data in mini_batch]
        # print('state_batch', state_batch)
        mcts_probs_batch = [data[1] for data in mini_batch]
        # print('mcts_probs_batch', mcts_probs_batch)
        label_batch = [data[2] for data in mini_batch]

        old_probs, old_v = self.policy_value_net.policy_value(state_batch)

        for i in range(self.epochs):
            # print('epoch', i)
            s = self.learn_rate * self.lr_multiplier
            # print('policy update', state_batch, mcts_probs_batch, label_batch)
            loss, entropy = self.policy_value_net.train_step(state_batch, mcts_probs_batch, label_batch, s)
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            # print('new_probs', new_probs)
            # print('new_v', new_v)
            kl = np.mean(np.sum(old_probs * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)), axis=1))
            if kl > self.kl_targ * 4:  # 4， early stopping if D_KL diverges badly
                break
        # adaptively adjust the learning rate
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5   # 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5  # 1.5

        explained_var_old = (1 - np.var(np.array(label_batch) - np.array(old_v)) / np.var(np.array(label_batch)))

        explained_var_new = (1 - np.var(np.array(label_batch) - new_v) / np.var(np.array(label_batch)))

        print(("kl:{:.5f},"
               "lr_multiplier:{:.3f},"
               "loss:{},"
               "entropy:{},"
               "explained_var_old:{:.3f},"
               "explained_var_new:{:.3f}"
               ).format(kl,
                        self.lr_multiplier,
                        loss,
                        entropy,
                        explained_var_old,
                        explained_var_new))

        return loss, entropy

    def policy_evaluate(self, n_games=5):  # n=10
        """
        Evaluate the trained policy by playing against the pure MCTS player
        Note: this is only for monitoring the progress of training
        """

        current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                         c_puct=self.c_puct,
                                         n_playout=self.n_playout,
                                         is_selfplay=1)

        win = 0

        for i in range(n_games):
            w, _ = self.game.start_self_play(current_mcts_player)
            w = int(w)
            win += w
        win_ratio = win / n_games
        print('We win', win, 'game among', n_games, 'games, and the win ratio is', win_ratio)
        # print("num_playouts:{}, win".format(self.pure_mcts_playout_num, win_ratio))
        return win_ratio

    def run(self):
        """run the training pipeline"""
        loss_list = []
        entropy_list = []
        succ_rate_list = []
        try:
            for i in range(self.game_batch_num):
                print(i+1, 'game is running')
                self.collect_selfplay_data(self.play_batch_size)
                if len(self.data_buffer) > self.batch_size:
                    loss, entropy = self.policy_update()
                    loss_list.append(loss)
                    entropy_list.append(entropy)
                # check the performance of the current model, and save the model params
                if (i+1) % self.check_freq == 0:
                    print("current self-play batch: {}".format(i+1))
                    win_ratio = self.policy_evaluate()
                    succ_rate_list.append(win_ratio)
                    self.policy_value_net.save_model('./current_policy.model')
                    if win_ratio > self.best_win_ratio:
                        print("New best policy!!!!!!!!")
                        self.best_win_ratio = win_ratio
                        # update the best_policy
                        self.policy_value_net.save_model('./best_policy.model')
                        if (self.best_win_ratio == 1.0 and
                                self.pure_mcts_playout_num < 5000):
                            self.pure_mcts_playout_num += 1000
                            self.best_win_ratio = 0.0

            print('There are', self.board.win, 'games over', self.game_batch_num, 'self-play.')
            plt.figure()
            plt.plot(loss_list, color='red')
            plt.ylim((0, 5))
            plt.xlabel("self-play")
            plt.ylabel("loss")
            plt.title("Loss over self-play games")

            plt.figure()
            plt.plot(entropy_list, color='blue')
            plt.ylim((0, 5))
            plt.xlabel("self-play")
            plt.ylabel("entropy")
            plt.title("Entropy over self-play games")

            plt.figure()
            plt.plot(succ_rate_list, color='green')
            plt.ylim((0, 1))
            plt.xlabel("self-play")
            plt.ylabel("Effectiveness")
            plt.title("Evaluation: Effectiveness over self-play games")
            plt.show()

        except KeyboardInterrupt:
            print('\n\rquit')


if __name__ == '__main__':
    training_pipeline = TrainPipeline()
    training_pipeline.run()
    print(time.process_time())
