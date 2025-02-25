# PEARL-SGD

This repository documents the code to reproduce the experiments reported in the paper:
> [Multiplayer Federated Learning: Reaching Equilibrium with Less Communication](https://arxiv.org/pdf/2501.08263?)

In this work, we introduce Multiplayer Federated Learning (MpFL), a novel framework that models the clients in the FL environment as players in a game-theoretic context, aiming to reach an equilibrium. Each player tries to optimize their utility function in this scenario, which may not align with the collective goal. Within MpFL, we propose Per-Player Local Stochastic Gradient Descent (PEARL-SGD), an algorithm in which each player/client performs local updates independently and periodically communicates with other players.

![Algorithm](images/algorithm.png)
