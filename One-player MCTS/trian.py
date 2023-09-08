from __future__ import print_function
import sys
import matplotlib.pyplot as plt
import math
import time
from MCTS import Board, Game, MCTSPlayer

sys.setrecursionlimit(1000000)


class TrainPipeline:
    def __init__(self):
        self.board_width = 6
        self.board_height = 8
        self.board = Board(width=self.board_width, height=self.board_height)
        self.game = Game(self.board)
        self.temp = 1  # the temperature param
        self.n_playout = 400  # 400 num of simulations for each move
        self.c_puct = 6  # math.sqrt(3), 5
        self.mcts_player = MCTSPlayer(c_puct=self.c_puct, n_playout=self.n_playout)
        self.n_games =1000000

    def run(self):
        """run the training pipeline"""
        succ_rate_list = []
        try:
            for i in range(self.n_games):
                print(i+1, 'game is running')
                l = self.game.start_self_play(self.mcts_player, temp=self.temp)
                print('The path is', l)
                succ = l / (self.board_width * self.board_height)
                succ_rate_list.append(succ)
                if succ == 1:
                    break

            print(self.board.complete)
            print(self.board.closed)
            # print('closed_rate is', self.board.closed / self.board.complete)
            print('succ_rate is', self.board.complete / self.n_games)

            plt.figure()
            plt.plot(succ_rate_list, color='green')
            plt.ylim((0, 1))
            plt.xlabel("MCTS Game")
            plt.ylabel("Effectiveness")
            plt.title("Effectiveness over games")
            plt.show()

        except KeyboardInterrupt:
            print('\n\rquit')


if __name__ == '__main__':
    training_pipeline = TrainPipeline()
    training_pipeline.run()
    print(time.process_time())
