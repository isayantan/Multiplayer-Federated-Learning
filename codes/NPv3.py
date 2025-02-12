"""
N-Player Game Theoretical Stepsize Stochastic
"""

import numpy as np
from tqdm import tqdm
from typing import Dict, List
import torch
import pickle

from model import NPGame

# Set the device
GPU = '4'      # set the GPU number
device = torch.device('cuda:'+GPU if torch.cuda.is_available() else 'cpu')

# Algorithm hyperparameters
N_COMM = int(15*1e3)
N_LOCAL_STEP = [1, 2, 4, 5, 8, 20]
# STEP_SIZE = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]

# Generate a quadratic game
# Set the model hyperparameters
RANDOM_SEED = 1024
N_DIM = 10
N_DATA = 100  # deterministic problem
N_BATCH = 10
N_TRIAL = 5
N_PLAYER = 5

# Set the min and max eigenvalues of the matrices
L_A, mu_A, L_B, mu_B = 1, 0.01, 10, 0

# Set the random seed for reproducibility
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Initialize the game
game = NPGame(N_PLAYER, N_DIM, N_DATA, L_A, mu_A, L_B, mu_B, device=device)
Lmax, mu, L= game.Lmax, game.mu, game.L  # store the game constants
ell = (L**2)/mu

# Print the problem constants
print(r'ell = {}'.format(ell))
print(r'mu = {}'.format(mu))
print(r'Lmax = {}'.format(Lmax))
print(r'L = {}'.format(L))
print(r'Condition Number = {}'.format(ell/mu))


# Set the initial values of x
x_start = torch.stack([torch.randn(N_DIM, requires_grad= True) for _ in range(N_PLAYER)], dim=0).to(device)

# Compute the initial distance
init_dist = game.opt_dist(x_start)

# Dictionary to store the relative errors
relative_errors: Dict[int, List[torch.Tensor]] = {}

# Run the algorithm
for n_local_step in tqdm(N_LOCAL_STEP):
    relative_errors[n_local_step] = []
    for _ in tqdm(range(N_TRIAL)):
        trial_errors = [init_dist/init_dist]
        # set the stepszie for the number of local step
        lr_x = 1/(ell * n_local_step + 2 * (n_local_step - 1) * Lmax * np.sqrt(ell/mu))  # For minimizing x_i
    
        # Initialize the variables x
        x = x_start.clone().detach().requires_grad_(True)
        
        # Communication loop within the local step (N_COMM rounds)
        # Each round consists of n_local_step updates for each variable (x1 and x2)
        # Communication is performed between the local updates, and the variables are synchronized after each round of updates
        for _ in range(N_COMM):
            x_new = torch.zeros((N_PLAYER, N_DIM), requires_grad= True).to(device=device)
            
            # Generate index for the current communication round
            index = [list(np.random.choice(N_DATA, N_BATCH, replace=False)) for _ in range(n_local_step)]
            
            # For each player run in parallel
            for player in range(N_PLAYER):
                # save the current values of x before independent updates
                x_local = x.clone().detach().requires_grad_(True)
                
                # perform n_local_step update
                for local_step in range(n_local_step):
                    x_local.grad = None
                    loss = game.objective_function(player, x_local, index= index[local_step])
                    loss.backward()
                    
                    with torch.no_grad():
                        x_local[player] -= lr_x * x_local.grad[player]  # Update x_local[player]
                    # x_local.grad.zero_()  # Zero the gradients for the next update
                
                # After N independent updates, copy x_local[player] in x_new[player]
                with torch.no_grad():
                    x_new[player].copy_(x_local[player])
            
            # After N independent updates, synchronize x
            with torch.no_grad():
                x.copy_(x_new)
        
            # Compute the distance between x and x_optimal after the communication round
            trial_errors.append(game.opt_dist(x) / init_dist)
         
        # store the relative error for this choice of local steps
        relative_errors[n_local_step].append(torch.tensor(trial_errors))
        
# Save the relative_errors and problem constants
data_to_save = {
    "relative_errors": relative_errors,  
    "Lmax": Lmax, 
    "mu": mu,
    "ell": ell
}

with open('results/NPv3.pkl', 'wb') as f:
    pickle.dump(data_to_save, f)