"""
We fix \tau and fine-tune \gamma to check which one works best
"""

import numpy as np
from tqdm import tqdm
from typing import Dict, List
import torch
import pickle
import math

from model import QuadGame

# Set the device
GPU = '4'      # set the GPU number
device = torch.device('cuda:'+GPU if torch.cuda.is_available() else 'cpu')

# Algorithm hyperparameters
N_COMM = int(1e4)
N_LOCAL_STEP = [1, 2, 4, 5, 8, 20]
STEP_SIZE = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]

# Generate a quadratic game
# Set the model hyperparameters
RANDOM_SEED = 1024
N_DIM = 10
N_DATA = 1  # deterministic problem

# Set the min and max eigenvalues of the matrices
L_A, mu_A, L_B, L_C, mu_C = 1, 0.01, 1, 1, 0.01

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
# Dictionary to store the fine-tune stepsizes
finetune_stepsize: Dict[int, float] = {}

# Run the algorithm
for n_local_step in tqdm(N_LOCAL_STEP):
    # track whether a valid (non-Nan) error has been found
    found_valid_error = False
    for idx, step_size in tqdm(enumerate(STEP_SIZE)):
        errors = [init_dist/init_dist] 
        
        # set the stepszie for the number of local step
        lr_x1 = step_size  # For minimizing x1
        lr_x2 = step_size  # For maximizing x2
        
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

            # Perform n_local_step updates on x (minimization step)
            for _ in range(n_local_step):
                loss_x1 = game.objective_function(x1_new, x2)  # x2 is held constant during x1 updates
                loss_x1.backward()
                
                with torch.no_grad():
                    x1_new -= lr_x1 * x1_new.grad  # x1 update (minimizing)
                x1_new.grad.zero_()

            # Perform n_local_step updates on y (maximization step)
            for _ in range(n_local_step):
                loss_x2 = game.objective_function(x1, x2_new)  # x1 is held constant during x2 updates
                loss_x2.backward()

                with torch.no_grad():
                    x2_new += lr_x2 * x2_new.grad  # x2 update (maximizing)
                x2_new.grad.zero_()

            # After n_local_step independent updates, synchronize x1 and x2
            with torch.no_grad():
                x1.copy_(x1_new)  # Update x1
                x2.copy_(x2_new)  # Update x2
            
            # store the relative error for this choice of local steps
            errors.append(game.opt_dist(x1, x2) / init_dist)
            
        print(f'Stepsize = {step_size}, Local step = {n_local_step}, error = {errors[-1]}')
        
        # Handle NaN values for errors
        if math.isnan(errors[-1]):
            continue
        
        # If this is the first valid step store it
        if not found_valid_error:
            relative_errors[n_local_step] = errors
            finetune_stepsize[n_local_step] = step_size
            found_valid_error = True
        elif errors[-1] < relative_errors[n_local_step][-1]:
            # Update the relative_errors[n_local_step] if the current error is smaller than the previous one
            relative_errors[n_local_step] = errors
            finetune_stepsize[n_local_step] = step_size
    print(f'Done with n_local_step = {n_local_step}')
    
# Save the relative_errors and problem constants
data_to_save = {
    "relative_errors": relative_errors,
    "finetune_stepsize": finetune_stepsize,  
    "Lmax": Lmax, 
    "mu": mu,
    "ell": ell
}

with open('results/QGv17.pkl', 'wb') as f:
    pickle.dump(data_to_save, f)
