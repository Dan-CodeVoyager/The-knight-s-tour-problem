import numpy as np
import random
import matplotlib.pyplot as plt
import time


class Ant(object):
    def __init__(self, board_size=8, num_ants=8, n_iterations=100000, alpha=1):
        self.board_size = board_size
        self.squares = board_size * board_size
        self.num_ants = num_ants
        self.MOVES = ((1, -2), (2, -1), (2, 1), (1, 2), (-1, 2), (-2, -1), (-2, 1), (-1, -2))
        self.alpha = alpha
        self.n_iterations = n_iterations
        self.start_squares = []
        self.total_sequence = 0
        self.Q = 1.0
        self.rau = 0.25
        self.pheromone = np.zeros(shape=(self.squares, self.squares), dtype=float)
        self.prob = np.zeros(shape=(self.squares, self.squares), dtype=float)
        self.best_solution = np.zeros(shape=())

        self.open_table_squares = np.zeros(shape=(self.board_size, self.board_size))

        ants = random.sample(range(0, self.squares), num_ants)
        for item in ants:
            start_row = int(item / self.board_size)
            start_col = int(item % self.board_size)
            start = (start_row, start_col)
            self.start_squares.append(start)

        self.complete_search = 0
        self.partly_search = 0

        self.path = []
        self.best_ant_path = []
        self.best = []

    def __clean_data(self):
        for i in range(self.board_size):
            for j in range(self.board_size):
                self.open_table_squares[i][j] = -1

        self.total_sequence = 0

        self.path = []

    def __legal_moves_from(self, row, col):
        for row_offset, col_offset in self.MOVES:
            move_row = row + row_offset
            move_col = col + col_offset
            if 0 <= move_row < self.board_size and 0 <= move_col < self.board_size:
                allowed = move_row * self.board_size + move_col
                yield allowed

    def __choice_next_square(self, row, col):
        allowed_list = []
        max_prob = 0
        current = row * self.board_size + col
        next_square = 0
        # print("current", current)
        next_squares = self.__legal_moves_from(row, col)
        for next_square in next_squares:
            r = int(next_square / self.board_size)
            c = int(next_square % self.board_size)
            if self.open_table_squares[r][c] == -1:
                allowed_list.append(next_square)
        # print("allowed_list", allowed_list)
        s = sum(self.pheromone[current][allowed] for allowed in allowed_list)
        if not allowed_list:
            return None
        else:
            for allowed_square in allowed_list:
                self.prob[current][allowed_square] = pow(self.pheromone[current][allowed_square], self.alpha) / s
                if max_prob < self.prob[current][allowed_square]:
                    max_prob = self.prob[current][allowed_square]
                    next_square = allowed_square
                # print("prob[", current, "][", allowed_square, "] = ", self.prob[current][allowed_square])
            next_row = int(next_square / self.board_size)
            next_col = int(next_square % self.board_size)
            self.total_sequence += 1
            self.open_table_squares[next_row][next_col] = self.total_sequence

            return next_square

    def __move(self, current_row, current_col):
        current = (current_row, current_col)
        # print("current:", current)
        self.path.append(current)
        allowed = self.__choice_next_square(current_row, current_col)
        # print("allowed:", allowed)
        while allowed is not None:
            allowed_row = int(allowed / self.board_size)
            allowed_col = int(allowed % self.board_size)
            return self.__move(allowed_row, allowed_col)
        # print(self.path)
        length = len(self.path)
        if self.total_sequence + 1 == self.squares:
            self.complete_search += 1
            print(self.open_table_squares)
            print("A complete search is found!")
        else:
            self.partly_search += 1
            # print("A partly search!")

        return self.path, length

    def __init_pheromone(self):
        for row in range(self.board_size):
            for col in range(self.board_size):
                current = row * self.board_size + col
                allowed_squares = self.__legal_moves_from(row, col)
                for sequence in allowed_squares:
                    self.pheromone[current][sequence] = 1e-6
                # print("pheromone[", current, "][", sequence, "] = ", self.pheromone[current][sequence])
        return self.pheromone

    def __update_pheromone(self, row, col):
        self.pheromone = (1 - self.pheromone) * self.rau
        path, length = self.__move(row, col)
        square_id = []
        for move_x, move_y in path:
            s = move_x * self.board_size + move_y
            square_id.append(s)
        # print("path is", path)
        # print("s", square_id)
        t = np.zeros(shape=(self.squares, self.squares), dtype=float)
        if len(square_id) == self.squares:
            t = 0
        else:
            for i in range(len(square_id)):
                if i == len(square_id) - 1:
                    break
                else:
                    t[square_id[i]][square_id[i + 1]] = self.Q * (len(square_id) - i - 1) / (self.squares - 1 - i)
                    # print("t", t)
        return t, path, length

    def __run_ants(self):
        t = np.zeros(shape=(self.squares, self.squares), dtype=float)
        ants = self.start_squares
        max_length = 0
        for ant in ants:
            # print("Ant ", ant)
            self.__clean_data()
            self.open_table_squares[ant[0], ant[1]] = self.total_sequence
            each_ant_t, path, length = self.__update_pheromone(ant[0], ant[1])
            t += each_ant_t
            # print("path is", path)
            if length >= max_length:
                max_length = length
                self.best_ant_path = path
        # print("best path for this iteration", self.best_ant_path)
        self.best.append(len(self.best_ant_path))
        self.pheromone += t

    def run_iterations(self):
        num = 0
        self.pheromone = self.__init_pheromone()
        for i in range(self.n_iterations):
            # print(i + 1, "iteration")
            num = i + 1
            self.__run_ants()
            if len(self.best_ant_path) == self.squares:
                break
        success_rate = float(self.complete_search / (num * self.num_ants))
        # print("the most visited squares over iterations:", self.best)
        print("success rate is ", success_rate)
        print("process time is ", time.process_time())
        plt.plot(self.best, color='b')
        plt.ylim((0, self.squares))
        plt.xlabel("iteration")
        plt.ylabel("the number of the visited squares")
        plt.title("Length of the Best Path over Iterations")
        plt.show()


if __name__ == '__main__':
    # n = int(input("board_size="))
    a = Ant()
    a.run_iterations()
