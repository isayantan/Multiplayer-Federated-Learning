"""
Deterministic Theoretical Step-size
"""

import numpy as np
from tqdm import tqdm
from typing import Dict, List
import torch
import pickle

from model import QuadGame

# Set the device
GPU = '4'      # set the GPU number
device = torch.device('cuda:'+GPU if torch.cuda.is_available() else 'cpu')

# Algorithm hyperparameters
N_COMM = int(2000)
N_LOCAL_STEP = [1, 2, 4, 5, 8, 20]

# Generate a quadratic game
# Set the model hyperparameters
RANDOM_SEED = 1000
N_DIM = 10
N_DATA = 100  # deterministic problem

# Set the min and max eigenvalues of the matrices
L_A, mu_A, L_B, L_C, mu_C = 1, 0.01, 10, 1, 0.01

# Set the random seed for reproducibility
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Initialize the game
game = QuadGame(N_DIM, N_DATA, L_A, mu_A, L_B, L_C, mu_C, device=device)
Lmax, mu, ell = game.Lmax, game.mu, game.ell  # store the game constants

# Set the initial values of x1 and x2
x1_start, x2_start = torch.randn(N_DIM, requires_grad=True).to(device), torch.randn(N_DIM, requires_grad=True).to(device)

# Compute the initial distance
init_dist = game.opt_dist(x1_start, x2_start)

# Dictionary to store the relative errors
relative_errors: Dict[int, List[torch.Tensor]] = {}

# Run the algorithm
for n_local_step in tqdm(N_LOCAL_STEP):
    relative_errors[n_local_step] = [init_dist/init_dist]
    # set the stepszie for the number of local step
    lr_x1 = 1/(ell * n_local_step + 2 * (n_local_step - 1) * Lmax * np.sqrt(ell/mu))  # For minimizing x1
    lr_x2 = 1/(ell * n_local_step + 2 * (n_local_step - 1) * Lmax * np.sqrt(ell/mu))  # For maximizing x2
    
    # Initialize the variables x1 and x2
    x1, x2 = x1_start.clone(), x2_start.clone()
    
    # Communication loop within the local step (N_COMM rounds)
    # Each round consists of n_local_step updates for each variable (x1 and x2)
    # Communication is performed between the local updates, and the variables are synchronized after each round of updates
    for _ in range(N_COMM):
        # Save current values of x1 and x2 before independent updates
        x1_new, x2_new = x1.clone(), x2.clone()
        
        # Retain gradients for non-leaf tensors
        x1_new.retain_grad()
        x2_new.retain_grad()

        # Perform N updates on x (minimization step)
        for _ in range(n_local_step):
            loss_x1 = game.objective_function(x1_new, x2)  # x2 is held constant during x1 updates
            loss_x1.backward()
            
            with torch.no_grad():
                x1_new -= lr_x1 * x1_new.grad  # x1 update (minimizing)
            x1_new.grad.zero_()

        # Perform N updates on y (maximization step)
        for _ in range(n_local_step):
            loss_x2 = game.objective_function(x1, x2_new)  # x1 is held constant during x2 updates
            loss_x2.backward()

            with torch.no_grad():
                x2_new += lr_x2 * x2_new.grad  # x2 update (maximizing)
            x2_new.grad.zero_()

        # After N independent updates, synchronize x1 and x2
        with torch.no_grad():
            x1.copy_(x1_new)  # Update x1
            x2.copy_(x2_new)  # Update x2
         
        # store the relative error for this choice of local steps
        relative_errors[n_local_step].append(game.opt_dist(x1, x2) / init_dist)
    
# Save the relative_errors and problem constants
data_to_save = {
    "relative_errors": relative_errors,  
    "Lmax": Lmax, 
    "mu": mu,
    "ell": ell
}

with open('results/QGv21.pkl', 'wb') as f:
    pickle.dump(data_to_save, f)
