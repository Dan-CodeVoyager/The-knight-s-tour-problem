import time

a = 10
b = 11
board = [[0 for x in range(b)] for y in range(a)]
result = [[0 for x in range(b)] for y in range(a)]
counter = 0
res_counter = 0
run_time = 0
back_time = 0


def makeMove(board, currX, currY):
    global counter
    global res_counter
    global run_time
    global back_time

    if counter == a*b:
        print("The success rate is ", 1 - back_time/run_time)
        print("running time is ", time.process_time())
        return True

    if currX > (a-1) or currX < 0 or currY > (b-1) or currY < 0 or board[currX][currY] == 1:
        return False
    else:
        run_time += 1
        counter += 1
        res_counter += 1

    board[currX][currY] = 1

    # visit the next square
    # (1, 2)
    if makeMove(board, currX+1, currY+2):
        result[currX][currY] = res_counter
        res_counter -= 1
        return True
    # (2, 1)
    elif makeMove(board, currX+2, currY+1):
        result[currX][currY] = res_counter
        res_counter -= 1
        return True
    # (2, -1)
    elif makeMove(board, currX+2, currY-1):
        result[currX][currY] = res_counter
        res_counter -= 1
        return True
    # (1, -2)
    elif makeMove(board, currX+1, currY-2):
        result[currX][currY] = res_counter
        res_counter -= 1
        return True
    # (-1, -2)
    elif makeMove(board, currX-1, currY-2):
        result[currX][currY] = res_counter
        res_counter -= 1
        return True
    # (-2, -1)
    elif makeMove(board, currX-2, currY-1):
        result[currX][currY] = res_counter
        res_counter -= 1
        return True
    # (-2, 1)
    elif makeMove(board, currX-2, currY+1):
        result[currX][currY] = res_counter
        res_counter -= 1
        return True
    # (-1, 2)
    elif makeMove(board, currX-1, currY+2):
        result[currX][currY] = res_counter
        res_counter -= 1
        return True

    # backtracking
    else:
        if currX > (a-1) or currX < 0 or currY > (b-1) or currY < 0 or board[currX][currY] == 1:
            back_time += 1
            counter -= 1
            res_counter -= 1
            board[currX][currY] = 0
        return False


if __name__ == '__main__':
    for i in range(1):
        makeMove(board, 0, 0)
        l = len(result)
        for i in range(l):
            print(result[i])
