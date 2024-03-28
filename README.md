The knight’s tour problem is an instance of the Hamiltonian path problem that is a typical NP-hard problem. 
A knight makes L-shape moves on a chessboard and tries to visit all the squares exactly once. 
The tour is closed if a knight can finish a complete tour and end on a square that is a neighborhood of its starting square; 
Otherwise, it is open. 

Many algorithms and heuristics have been proposed to solve this problem. The most well-known one is Warnsdorff’s heuristic. 
Warnsdorff’s idea is to prioritize moving to the square that has the lowest number of non-visited neighbors, i.e., it is a greedy heuristic. 
Although this heuristic is fast, it does not always return a closed tour, and it can get stuck into local optima easily. 
Also, it only works on boards of certain dimensions. Leveraging recent advances in sequential learning, our goal is to develop a new strategy based on reinforcement learning. 
Ideally, it should be able to find a closed tour on chessboards of any size. 

We investigate different formulations of this problem as a single-player and two-player game and compare different reinforcement learning techniques: 
value-based methods, policy optimization, and actor-critic methods. 
Furthermore, we present a method based on Monte Carlo tree search (MCTS), and we then combine it with a Convolutional Neural Network (CNN) to solve this problem. 
Compared to previous work, our approach breaks away from greedy heuristics and introduces uncertainty to balance the trade-off between exploration and exploitation. 
We evaluate baselines and our proposed models w.r.t. effectiveness, efficiency, and rate of closed tours. 
The experiment results show that the pure MCTS outperforms the baselines on chessboards smaller than eight by eight. 
However, it has no advantage on larger chessboards, e.g., 20 by 20 chessboard, because searching for a solution using UCB heuristic sampling is difficult.



The CNN-based MCTS is more suitable for a large chessboard. 
Although efficiency is low due to network training, the effectiveness and rate of closed tours have been improved for both small and large chessboards. 
In the future, we will search for a complete tour of irregular chessboards with the pre-training technique.
