"""
Monte Carlo Tree Search in AlphaGo Zero style, which uses a policy-value
network to guide the tree search and evaluate the leaf nodes

@author: Dan Jia, inspired by Junxiao Song
"""

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

        children = board.knight_walk()
        for child in children:
            if child not in self._children:
                self._children[child] = TreeNode(self, self._P)

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
        # self._Q_s_a += leaf_value / self._n_visits

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

    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000):
        """
        policy_value_fn: a function that takes in a board state and outputs
        a list of (action, probability) tuples and also a score in [-1, 1]
        (i.e. the expected value of the end game score from the current
        player's perspective) for the current player.
        c_puct: a number in (0, inf) that controls how quickly exploration
        converges to the maximum-value policy. A higher value means
        relying on the prior more.
        """

        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout
        self.succ = []

    def _playout(self, board):
        """
        Run a single play-out from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        """

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

        # action_probs, leaf_value = self._policy(board)

        leaf_value = 0
        # Check for end of game.
        end, win = board.is_game_end()

        if not end:
            node.expand(board)
        else:
            # for end state，return the "true" leaf_value
            if win is True:
                leaf_value = 1
            else:
                # leaf_value = board.visited_squares / (board.width * board.height)
                leaf_value = -1

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

    def __init__(self, policy_value_fn, c_puct=5, n_playout=2000):
        self.mcts = MCTS(policy_value_fn, c_puct, n_playout)

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, board, temp=1e-3, return_prob=0):
        sensible_moves = board.knight_walk()
        """
        -1. -1. -1. -1. -1.
        -1. -1. -1. -1. -1.
        -1. -1. -1. -1. -1.
        -1. -1. -1. -1. -1.
        -1. -1. -1. -1. -1.
         0. -1. -1. -1. -1.
        -1. -1. 34. -1. -1.
        """
        # the pi vector returned by MCTS as in the alphaGo Zero paper
        move_probs = np.zeros((board.width, board.height))

        if len(sensible_moves) > 0:
            acts, probs = self.mcts.get_move_probs(board, temp)
            for act in acts:
                i = 0
                a = act[0]
                b = act[1]
                move_probs[a][b] = probs[i]
                i += 1
            # add Dirichlet Noise for exploration (needed for self-play training)
            move = np.random.choice(len(acts),
                                    p=0.75 * probs + 0.25 * np.random.dirichlet(0.3 * np.ones(len(probs))))
            # p = 0.75 * probs + 0.25 * np.random.dirichlet(0.3 * np.ones(len(probs)))
            # print('move [', move, '] is', acts[move])
            # print('p is', p)
            # update the root node and reuse the search tree
            self.mcts.update_with_move(acts[move])

            if return_prob:
                return acts[move], move_probs
            else:
                return acts[move]

        else:
            if board.end == 0:
                p, _ = board.get_current_player()
                # print('current player is ', p)
                board.changed_player = 2 ** p % 2
                # print('changed player is ', board.changed_player)
                board.end = 1
            new_sensible_moves = board.knight_walk()
            # the pi vector returned by MCTS as in the alphaGo Zero paper
            if len(new_sensible_moves) > 0:
                acts, probs = self.mcts.get_move_probs(board, temp)
                for act in acts:
                    i = 0
                    a = act[0]
                    b = act[1]
                    move_probs[a][b] = probs[i]
                    i += 1
                move = np.random.choice(len(acts),
                                        p=0.75 * probs + 0.25 * np.random.dirichlet(0.3 * np.ones(len(probs))))
                # update the root node and reuse the search tree
                self.mcts.update_with_move(acts[move])

                if return_prob:
                    return acts[move], move_probs
                else:
                    return acts[move]

            if len(new_sensible_moves) == 0:
                board.init_board()
                return self.get_action(board)

    def __str__(self):
        return "MCTS {}".format(self.player)
