import time

KNIGHT_MOVES = [(2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1)]


class KnightTour:
    def __init__(self, board_size):
        self.board_size = board_size  # tuple
        self.board = []
        self.closed = 0
        self.open = 0
        self.total_move = 0
        for i in range(board_size[0]):
            temp = []
            for j in range(board_size[1]):
                temp.append(0)
            self.board.append(temp) # empty cell
        self.move = 1

    def print_board(self):
        print('board:')
        for i in range(self.board_size[0]):
            print(self.board[i])

    def warnsdroff(self, start_pos, GUI=False):
        x_pos,  y_pos = start_pos
        x_pos_start, y_pos_start = start_pos
        self.board[x_pos][y_pos] = self.move

        if not GUI:
            while self.move <= self.board_size[0] * self.board_size[1]:
                self.move += 1
                next_pos = self.find_next_pos((x_pos, y_pos))
                if next_pos:
                    x_pos, y_pos = next_pos
                    self.board[x_pos][y_pos] = self.move
                else:
                    self.print_board()
                    x = x_pos_start - x_pos
                    y = y_pos_start - y_pos
                    if abs(x) == 1 and abs(y) == 2:
                        if abs(x) == 2 and abs(y) == 1:
                            print("A closed tour is found!")
                            self.closed += 1
                    else:
                        self.open += 1
                        print("A open tour is found")
                    return self.board
        else:
            if self.move <= self.board_size[0] * self.board_size[1]:
                self.move += 1
                next_pos = self.find_next_pos((x_pos, y_pos))
                return next_pos

    def find_next_pos(self, current_pos):
        empty_neighbours = self.find_neighbours(current_pos)
        if len(empty_neighbours) == 0:
            return
        least_neighbour = 8
        least_neighbour_pos = ()
        for neighbour in empty_neighbours:
            neighbours_of_neighbour = self.find_neighbours(pos=neighbour)
            if len(neighbours_of_neighbour) <= least_neighbour:
                least_neighbour = len(neighbours_of_neighbour)
                least_neighbour_pos = neighbour
        return least_neighbour_pos

    def find_neighbours(self, pos):
        neighbours = []
        for dx, dy in KNIGHT_MOVES:
            x = pos[0] + dx
            y = pos[1] + dy
            if 0 <= x < self.board_size[0] and 0 <= y < self.board_size[1] and self.board[x][y] == 0:
                neighbours.append((x, y))
        return neighbours

    def run(self, sim_num, start_pos):
        for i in range(sim_num):
            print(i+1, "Iteration")
            self.warnsdroff(start_pos)
            self.move = 1
        print("The rate of closed tour is", self.closed/sim_num)
        print("The rate of open tour is", self.open / sim_num)
        print("The success rate of tour is", (self.closed + self.open) / sim_num)


a = KnightTour((11, 11))
a.run(1, (0, 0))
print("The process running time is ", time.process_time()/100)
