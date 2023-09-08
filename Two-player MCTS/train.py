# -*- coding: utf-8 -*-
"""
An implementation of the training pipeline of AlphaZero for Gomoku

@author: Junxiao Song
"""


from __future__ import print_function
import time
import math
from collections import defaultdict, deque
from game_logic import Board, Game
from knight_MCTS import MCTSPlayer
from policy_value import PolicyValueNet


class TrainPipeline:
    def __init__(self, init_model=None):
        # params of the board and the game
        self.board_width = 20
        self.board_height = 20
        self.board = Board(width=self.board_width, height=self.board_height)
        self.game = Game(self.board)
        # training params
        self.temp = 1.0  # the temperature param
        self.n_playout = 400  # 400 num of simulations for each move
        self.c_puct = 5  # math.sqrt(2), 5
        self.buffer_size = 10000  # 10000
        self.batch_size = 512  # 512  # mini-batch size for training
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.game_batch_num = 5000  # 1500
        if init_model:
            # start training from an initial policy-value net
            self.policy_value_net = PolicyValueNet(self.board_width, self.board_height, model_file=init_model)
        else:
            # start training from a new policy-value net
            self.policy_value_net = PolicyValueNet(self.board_width, self.board_height)

        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playout)

    def collect_selfplay_data(self, n_games=1):
        """collect self-play data for training"""
        for i in range(n_games):
            win, play_data = self.game.start_self_play(self.mcts_player, temp=self.temp)
            #play_data = list(play_data)[:]
            # print('play_data', play_data)
            self.episode_len = self.board.countDiff()
            # augment the data
            # play_data = self.get_equi_data(play_data)
            # self.data_buffer.extend(play_data)

    def run(self):
        """run the training pipeline"""
        win = 0
        try:
            for i in range(self.game_batch_num):
                print(i+1, 'game is running')
                self.collect_selfplay_data(self.play_batch_size)
                print("batch i:{}, episode_len:{}".format(i+1, self.episode_len))
                if self.board.is_win():
                    win += 1
                    break
            print('succ is', win / self.game_batch_num)
        except KeyboardInterrupt:
            print('\n\rquit')


if __name__ == '__main__':
    training_pipeline = TrainPipeline()
    training_pipeline.run()
    print(time.process_time())
