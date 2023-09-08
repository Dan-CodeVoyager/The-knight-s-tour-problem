import numpy as np
import time

import main

if __name__ == '__main__':
    success = []
    for i in range(2):
        _, succ = list(main.knights_tour(5, 7))
        success.append(succ)
    print("The success rate is ", success)
    print(np.mean(success))
    print("The time cost is ", time.process_time()/2)
