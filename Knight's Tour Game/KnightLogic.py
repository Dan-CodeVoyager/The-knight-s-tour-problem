"""
Author: Dan Jia
Date: May 8, 2023.
Board class.
Board data:
  -1 = unvisited squares
  first dim is column , 2nd is row:
  states[1][7] is the square in column 2,
  at the opposite end of the board in row 8.
  Squares are stored and manipulated as (x,y) tuples.
  x is the column, y is the row.
"""


from __future__ import print_function
import numpy as np
import random


class Board(object):

    def __init__(self, **kwargs):
        self.width = int(kwargs.get('width', 7))
        self.height = int(kwargs.get('height', 5))
        self.players = [0, 1]  # player 0 and player 1
        self.labels = 0
        self.win = 0
        self.KNIGHT_MOVES = [(2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1)]

    def init_board(self):
        self.current_player = None
        self.current_position = None

        self.move_player1 = 0
        self.move_player2 = self.width * self.height - 1
        self.changed_player = -1

        self.states = - np.ones((1, self.width, self.height))
        self.candidates = None
        self.end = 0

        self.visited_squares = self.width * self.height

        self.player1_move_list = []
        self.player2_move_list = []

        # generate two start squares of the two knights
        self.player1 = (random.randint(0, self.width - 1), random.randint(0, self.height - 1))
        self.states[0][self.player1] = self.move_player1
        self.player1_move_list.append(self.player1)

        for dx, dy in self.KNIGHT_MOVES:
            x = self.player1[0] + dx
            y = self.player1[1] + dy
            if 0 <= x < self.width and 0 <= y < self.height:
                self.player2 = (x, y)
                self.states[0][x, y] = self.move_player2
                break
        self.player2_move_list.append(self.player2)

        # print('player 1', self.player1)
        # print('player 2', self.player2)
        # print(self.states)

    def get_states(self):
        return self.states

    def countDiff(self):
        """
        Counts the #move of the given state
        (-1 for unvisited square)
        """
        count = 0
        for y in range(self.height):
            for x in range(self.width):
                if self.states[0][x][y] != -1:
                    count += 1

        # print('the number of visited squares is', count)

        return count

    def knight_walk(self):
        """
        To get legal moves, possible actions in given position
        :param pos: current position
        :return: all the next possible squares
        """

        _, pos = self.get_current_player()

        successors = []
        for dx, dy in self.KNIGHT_MOVES:
            x = pos[0] + dx
            y = pos[1] + dy
            if 0 <= x < self.width and 0 <= y < self.height and self.states[0][x][y] == -1:
                successors.append((x, y))

        self.candidates = successors

        return successors

    def has_legal_moves(self):
        successors = self.knight_walk()
        if len(successors) > 0:
            return True
        else:
            return False

    def execute_move(self, move):
        """
        Perform the given move on the board
        :param move: possible moves
        :return: a sequence, that is current state

        """

        self.current_player, self.current_position = self.get_current_player()

        if self.current_player == 0:
            self.states[0][self.current_position[0]][self.current_position[1]] = self.move_player1
            self.move_player1 += 1
            self.states[0][move[0]][move[1]] = self.move_player1
            self.player1_move_list.append(move)
        else:
            self.states[0][self.current_position[0]][self.current_position[1]] = self.move_player2
            self.move_player2 -= 1
            self.states[0][move[0]][move[1]] = self.move_player2
            self.player2_move_list.append(move)

        # print(self.states)

        return self.states

    def find_next_pos(self):
        candidate_pool = self.knight_walk()
        least_neighbour_pos = None
        temp = 8
        for move in candidate_pool:
            next_node_visits = 0
            for dx, dy in self.KNIGHT_MOVES:
                x = move[0] + dx
                y = move[1] + dy
                if 0 <= x < self.width and 0 <= y < self.height and self.states[0][x][y] == -1:
                    next_node_visits += 1
                if temp <= next_node_visits:
                    temp = next_node_visits
                    least_neighbour_pos = move
        return temp, least_neighbour_pos

    def is_win(self):
        if self.countDiff() == self.visited_squares:
            a = self.player1_move_list[-1]
            b = self.player2_move_list[-1]
            for move in self.KNIGHT_MOVES:
                if (a[0] == b[0] + move[0]) and (a[1] == b[1] + move[1]):
                    # print(self.states)
                    return True
        else:
            return False

    def is_game_end(self):
        """
        Check whether the game is ended or not
        :return end, win
        """
        win = self.is_win()
        if win:
            return True, True
        else:
            if self.end == 0:
                return False, False
            if self.end == 1 and self.has_legal_moves():
                return False, False
            if self.end == 3:
                return True, False
            else:
                return True, False

    def get_current_player(self):
        if self.changed_player == -1:
            if len(self.player1_move_list) == len(self.player2_move_list):
                self.current_player = 0
                self.current_position = self.player1_move_list[-1]
            else:
                self.current_player = 1
                self.current_position = self.player2_move_list[-1]
        elif self.changed_player == 0:
            self.current_player = 0
            self.current_position = self.player1_move_list[-1]
        elif self.changed_player == 1:
            self.current_player = 1
            self.current_position = self.player2_move_list[-1]

        return self.current_player, self.current_position


class Game(object):
    """game server"""

    def __init__(self, b):
        self.board = b

    def start_play(self, player1, player2, start_player=0):
        """start a game between two players"""

        if start_player not in (0, 1):
            raise Exception('start_player should be either 0 (player1 first) '
                            'or 1 (player2 first)')

        self.board.init_board()
        p1, p2 = self.board.players
        player1.set_player_ind(p1)
        player2.set_player_ind(p2)
        players = {p1: player1, p2: player2}

        while True:
            current_player, _ = self.board.get_current_player()
            player_in_turn = players[current_player]
            # print('player_in_turn is', player_in_turn)
            move = player_in_turn.get_action(self.board)
            # print('move', move)
            # print('moving ...')
            self.board.execute_move(move)
            # print(self.board.states)
            end, winner = self.board.is_game_end()
            if end:
                return winner

    def start_self_play(self, player, temp=1e-3):
        """
        start a self-play game using an MCTS player, reuse the search tree,
        and store the self-play data: (state, mcts_probs, reward) for training
        """

        self.board.init_board()
        states, mcts_probs, current_players, labels = [], [], [], []

        while True:

            move, move_probs = player.get_action(self.board, temp=temp, return_prob=1)

            # store the data
            states.append(self.board.states)
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player)

            # perform a move
            # if (type(move) is not int):
            self.board.execute_move(move)

            # print(self.board.states)

            end, win = self.board.is_game_end()
            if end:
                if win is True:
                    # print(self.board.states)
                    print('A closed tour is found!!!!!!!!!!!')
                    self.board.labels = 1
                    self.board.win += 1
                    labels.append(self.board.labels)
                # self.board.labels = self.board.countDiff() / (self.board.width * self.board.height)
                self.board.labels = -1
                labels.append(self.board.labels)
                print(self.board.states)
                print('The path length is', self.board.countDiff())
                print('The real leaf value is', self.board.labels)
                # return self.board.labels, zip(states, mcts_probs, labels)
                return win, zip(states, mcts_probs, labels)
