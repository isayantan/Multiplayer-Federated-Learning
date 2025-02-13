import numpy as np
from tqdm import tqdm
from typing import Dict
import torch
import pickle

from model import QuadraticGame

# Algorithm hyperparameters
N_COMM = 100
N_LOCAL_STEP = [i for i in range(1, 101)]
STEP_SIZE = [i*1e-5 for i in range(1, 1001, 10)]

# Generate a quadratic game
# Set the model hyperparameters
RANDOM_SEED = 1024
N_DIM = 10

# Set the min and max eigenvalues of the matrices
L_A, mu_A, L_B, L_C, mu_C = 1, 0.01, 10, 1, 0.01

# Set the random seed for reproducibility
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Initialize the game
game = QuadraticGame(N_DIM, L_A, mu_A, L_B, L_C, mu_C)

# Set the initial values of x1 and x2
x1_start, x2_start = torch.randn(N_DIM, requires_grad=True), torch.randn(N_DIM, requires_grad=True)

# Compute the initial distance
init_dist = game.opt_dist(x1_start, x2_start)

# Dictionary to store the relative errors
relative_errors: Dict[int, Dict[float, float]] = {}

# Run the algorithm
for n_local_step in tqdm(N_LOCAL_STEP):
    relative_errors[n_local_step] = {}
    for step_size in STEP_SIZE:
        lr_x1 = step_size  # For minimizing x1
        lr_x2 = step_size  # For maximizing x2
        
        # Initialize the variables x1 and x2
        x1, x2 = x1_start.clone(), x2_start.clone()
        
        # Communication loop within the local step (N_COMM rounds)
        # Each round consists of N_LOCAL_STEP updates for each variable (x1 and x2)
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
                
        # store the relative error for this pair of (local_step, step_size)
        relative_errors[n_local_step][step_size] = game.opt_dist(x1, x2) / init_dist
        

# Save the relative_errors dictionary to a file
with open('results/QGdetv1.pkl', 'wb') as f:
    pickle.dump(relative_errors, f)