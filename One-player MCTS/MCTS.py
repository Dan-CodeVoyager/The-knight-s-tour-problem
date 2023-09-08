import random
import time
import numpy as np
import copy


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


class TreeNode(object):
    """
    A node in the MCTS tree.
    Each node keeps track of its own value Q, prior probability P, and
    its visit-count-adjusted prior score u.
    """

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self._n_visits = 0
        self._Q_s_a = 0
        self._u = 0
        self._P = prior_p

    def expand(self, board):
        """
        Expand tree by creating new children.

        move_priors: a list of tuples of actions and their prior probability
        according to the policy function.
        """
        moves = board.knight_walk()
        for action in moves:
            if action not in self._children:
                self._children[action] = TreeNode(self, self._P)

        return self._children

    def select(self, c_puct):
        """
        Select action among children that gives maximum action value Q
        plus bonus u(P).

        Return: A tuple of (action, next_node)
        """

        """
            for node in expand_children:
            self._children[node] = self.get_value(c_puct)
            print(node, '->', self._children[node])
        """

        return max(self._children.items(), key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        """
        Update node values from leaf evaluation.

        leaf_value: the value of subtree evaluation from the current player's
        perspective.
        """

        # Count visit.
        self._n_visits += 1
        # Update Q, a running average of values for all visits.
        self._Q_s_a += 1.0 * (leaf_value - self._Q_s_a) / self._n_visits
        # self._Q_s_a += 1.0 * leaf_value / self._n_visits

    def update_recursive(self, leaf_value):
        """
        Like a call to update(), but applied recursively for all ancestors.
        """

        # If it is not root, this node's parent should be updated first.
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        """
        Calculate and return the value for this node.

        It is a combination of leaf evaluations Q, and this node's prior
        adjusted for its visit count, u.

        c_puct: a number in (0, inf) controlling the relative impact of
        value Q, and prior probability P, on this node's score.
        """

        self._u = (c_puct * self._P * np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q_s_a + self._u

    def is_leaf(self):
        """
        Check if leaf node (i.e. no nodes below this have been expanded).
        """
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS(object):
    """
    An implementation of Monte Carlo Tree Search.
    """

    def __init__(self, c_puct=5, n_playout=10000):

        self._root = TreeNode(None, 1.0)
        self._c_puct = c_puct
        self._n_playout = n_playout
        self.succ = []

    def _playout(self, board):
        """
        Run a single play-out from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        """

        leaf_value = 0
        node = self._root
        while 1:
            if node.is_leaf():
                break
            # Greedily select next move.
            action, node = node.select(self._c_puct)
            board.execute_move(action)
            # print(board.states)

        """
        Evaluate the leaf using a network which outputs a list of
        (action, probability) tuples p and also a score v in [-1, 1]
        for the current player.
        """

        end, win = board.is_game_end()

        if not end:
            node.expand(board)
        else:
            # for end state，return the "true" leaf_value
            if win is True:
                leaf_value = board.labels
            else:
                # leaf_value = board.visited_squares / (board.width * board.height)
                leaf_value = 0

        # Update value and visit count of nodes in this traversal.
        node.update_recursive(-leaf_value)

    def get_move_probs(self, board, temp=1e-3):
        """
        Run all play_outs sequentially and return the available actions and
        their corresponding probabilities.
        board: the current game state
        temp: temperature parameter in (0, 1] controls the level of exploration
        """

        for n in range(self._n_playout):
            state_copy = copy.deepcopy(board)
            self._playout(state_copy)

        # calc the move probabilities based on visit counts at the root node
        act_visits = [(act, node._n_visits) for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))

        return acts, act_probs

    def update_with_move(self, last_move):
        """
        Step forward in the tree, keeping everything we already know about the subtree.
        """

        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"


class MCTSPlayer(object):
    """AI player based on MCTS"""

    def __init__(self, c_puct=5, n_playout=2000):
        self.mcts = MCTS(c_puct, n_playout)

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, board, temp=1e-3):

        available_moves = board.knight_walk()

        move_probs = np.zeros((board.width, board.height))

        if len(available_moves) > 0:
            acts, probs = self.mcts.get_move_probs(board, temp)
            for act in acts:
                i = 0
                a = act[0]
                b = act[1]
                move_probs[a][b] = probs[i]
                i += 1
            move = np.random.choice(len(acts), p=0.75 * probs + 0.25 * np.random.dirichlet(0.3 * np.ones(len(probs))))
            # move = np.random.choice(len(acts), p=probs)
            self.mcts.update_with_move(acts[move])

            return acts[move], move_probs

    def __str__(self):
        return "MCTS {}".format(self.player)


class Board(object):

    def __init__(self, **kwargs):
        self.width = int(kwargs.get('width', 8))
        self.height = int(kwargs.get('height', 6))
        self.labels = 0
        self.complete = 0
        self.closed = 0
        self.KNIGHT_MOVES = [(2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1)]

    def init_board(self):
        self.c = 0
        self.current_player = None
        self.current_position = None

        self.move_player = 0
        self.states = - np.ones((self.width, self.height))
        self.candidates = None

        self.visited_squares = self.width * self.height

        self.player_move_list = []

        # generate start square
        self.player = (random.randint(0, self.width - 1), random.randint(0, self.height - 1))
        self.states[self.player] = self.move_player
        self.player_move_list.append(self.player)

    def get_current_position(self):
        self.current_position = self.player_move_list[-1]
        return self.current_position

    def execute_move(self, move):
        """
        Perform the given move on the board
        :param move: possible moves
        :return: a sequence, that is current state

        """

        self.current_position = self.get_current_position()

        self.states[self.current_position[0]][self.current_position[1]] = self.move_player
        self.move_player += 1
        self.states[move[0]][move[1]] = self.move_player
        self.player_move_list.append(move)

        # print(self.states)

        return self.states

    def knight_walk(self):
        """
        To get legal moves, possible actions in given position
        :param pos: current position
        :return: all the next possible squares
        """

        pos = self.get_current_position()

        successors = []
        for dx, dy in self.KNIGHT_MOVES:
            x = pos[0] + dx
            y = pos[1] + dy
            if 0 <= x < self.width and 0 <= y < self.height and self.states[x][y] == -1:
                successors.append((x, y))

        self.candidates = successors

        return successors

    def countDiff(self):
        """
        Counts the #move of the given state
        (-1 for unvisited square)
        """
        count = 0
        for y in range(self.height):
            for x in range(self.width):
                if self.states[x][y] != -1:
                    count += 1

        # print('the number of visited squares is', count)

        return count

    def has_legal_moves(self):
        successors = self.knight_walk()
        if len(successors) > 0:
            return True
        else:
            return False

    def is_win(self):
        if self.countDiff() == self.visited_squares:
            # self.complete += 1
            end = self.player_move_list[-1]
            for move in self.KNIGHT_MOVES:
                if (end[0] == self.player[0] + move[0]) and (end[1] == self.player[1] + move[1]):
                    # self.closed += 1
                    self.labels = 2
                    self.c = 1
                    print('A closed tour is found!!!!!!!')
            print("Congratulations！！！ The knight walk all square exact once!!!")
            self.labels = 1
            print(self.states)
            return True
        else:
            self.labels = 0
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
            if self.has_legal_moves():
                return False, False
            else:
                self.labels = 0
                return True, False


class Game(object):

    def __init__(self, b):
        self.board = b

    def start_self_play(self, player, temp=1):
        """
        start a self-play game using an MCTS player, reuse the search tree,
        and store the self-play data: (state, mcts_probs, reward) for training
        """

        self.board.init_board()
        # states, mcts_probs, labels = [], [], []

        while True:
            move, move_probs = player.get_action(self.board, temp=temp)

            # store the data
            # states.append(self.board.states)
            # mcts_probs.append(move_probs)

            # perform a move
            self.board.execute_move(move)

            end, win = self.board.is_game_end()

            if end:
                if win:
                    self.board.complete += 1
                    if self.board.c == 1:
                        self.board.closed += 1
                l = self.board.countDiff()
                return l
