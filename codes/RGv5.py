"""
This script implements the RobotGame from model.py and evaluates the performance of PEARL-SGD.

Usage:
- Set the GPU number and other hyperparameters as needed.
- Run the script to execute the algorithm and save the results in 'results/RGv3.pkl'.

The script initializes the game, sets initial values, and runs the algorithm for different local steps and step sizes.
It stores the functional value of each player for each configuration.
"""

import math
import pickle
import numpy as np
from tqdm import tqdm
from typing import Dict, List
import torch

from model import RobotGame as RobotGame

# Set the device
GPU = '4'      # set the GPU number
device = torch.device('cuda:'+GPU if torch.cuda.is_available() else 'cpu')

# Algorithm hyperparameters
N_COMM = int(5*1e3)
# N_COMM = 5
N_LOCAL_STEP = 5

# Set the model hyperparameters
RANDOM_SEED = 1024
NOISE_FACTOR = 10
N_PLAYER = 5
N_DATA = 1
N_TRIAL = 5

# Set the random seed for reproducibility
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Initialize the game
game = RobotGame(device=device)

# Game constants
Lmax, mu, L= game.Lmax, game.mu, game.L  
ell = (L**2)/mu

# Print the problem constants
print(r'ell = {}'.format(ell))
print(r'mu = {}'.format(mu))
print(r'Lmax = {}'.format(Lmax))
print(r'L = {}'.format(L))
print(r'Condition Number = {}'.format(ell/mu))

# q_start = torch.stack([torch.randn(N_DIM, requires_grad= True) for _ in range(N_PLAYER)], dim=0).to(device)
q_start = torch.randn(N_PLAYER, requires_grad=True).to(device)

# Compute the initial distance
init_dist = game.opt_dist(q_start)

# Dictionary to store the functional value of each player after communication rounds for every trial
functional_values: Dict[int, List[torch.Tensor]] = {}

for player in range(N_PLAYER):
    # Tensor to store functional values for each trial and communication round
    functional_values[player] = torch.zeros(N_TRIAL, N_COMM, dtype=torch.float32).to(device)

# Run the algorithm
for trial in tqdm(range(N_TRIAL)):
    # set the stepszie for the number of local step
    lr_x = 1 / (ell.item() * N_LOCAL_STEP + 2 * (N_LOCAL_STEP - 1) * Lmax.item() * np.sqrt(ell.item() / mu.item()))  # For minimizing q_i

    # Initialize the variables x
    q = q_start.clone().detach().requires_grad_(True)
    
    # Communication loop within the local step (N_COMM rounds)
    # Each round consists of n_local_step updates for each variable (x1 and x2)
    # Communication is performed between the local updates, and the variables are synchronized after each round of updates
    for comm in tqdm(range(N_COMM)):
        q_new = torch.zeros(N_PLAYER, requires_grad= True).to(device=device)
        
        for player in range(N_PLAYER):
            # save the current values of q before independent updates
            q_local = q.clone().detach().requires_grad_(True)
            
            # perform n_local_step update
            for local_step in range(N_LOCAL_STEP):
                q_local.grad = None
                loss = game.objective_function(player, q_local)
                loss.backward()
                
                with torch.no_grad():
                    noise = NOISE_FACTOR * torch.randn_like(q_local.grad[player])
                    q_local[player] -= lr_x * (q_local.grad[player] + noise)  # Update q_local[player]
            
            # After N independent updates, copy q_local[player] in q_new[player]
            with torch.no_grad():
                q_new[player].copy_(q_local[player])
        
        # After N independent updates, synchronize q
        with torch.no_grad():
            q.copy_(q_new)
            
        # Store the functional value of each player after communication rounds
        for player in range(N_PLAYER):
            with torch.no_grad():
                functional_values[player][trial, comm] = game.objective_function(player, q).item()
        
# Save the relative_errors and problem constants
data_to_save = {
    "functional_values": functional_values,  
    "Lmax": Lmax, 
    "mu": mu,
    "ell": ell
}

with open('results/RGv5.pkl', 'wb') as f:
    pickle.dump(data_to_save, f)