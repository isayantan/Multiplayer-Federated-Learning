# PEARL-SGD

This repository documents the code to reproduce the experiments reported in the paper:
> [Multiplayer Federated Learning: Reaching Equilibrium with Less Communication](https://arxiv.org/pdf/2501.08263?)

In this work, we introduce Multiplayer Federated Learning (MpFL), a novel framework that models the clients in the FL environment as players in a game-theoretic context, aiming to reach an equilibrium. Each player tries to optimize their utility function in this scenario, which may not align with the collective goal. Within MpFL, we propose Per-Player Local Stochastic Gradient Descent (PEARL-SGD), an algorithm in which each player/client performs local updates independently and periodically communicates with other players.

![Algorithm](images/algorithm.png)

This repository evaluates the performance of PEARL-SGD for solving different N-player games. If you use this code for your research, please cite the paper as follows:

```
@article{yoon2025multiplayer,
  title={Multiplayer federated learning: Reaching equilibrium with less communication},
  author={Yoon, TaeHo and Choudhury, Sayantan and Loizou, Nicolas},
  journal={arXiv preprint arXiv:2501.08263},
  year={2025}
}
```

## Table of Contents

<!--ts-->
   * [Quadratic Minimax Game](#quadratic-minimax-game)
   * [Quadratic n-player Game](#quadratic-n-player-game)
   * [Heatmap](#heatmap)
<!--te-->

## Quadratic Minimax Game
In Figure 2 of our paper, we compare the performance of PEARL-SGD for different values of synchronization interval $\tau \in \{ 1, 2, 4, 5, 8 \}$. 
![Quadratic Minimax Game](images/fig2.png)
To reproduce the plots in Figure 2, please run the codes in 
  - [Figure 2a](codes/QGv21.ipynb)
  - [Figure 2b](codes/QGv19.ipynb)
  - [Figure 2c](codes/QGv17.ipynb)
  - [Figure 2d](codes/QGv16.ipynb)


