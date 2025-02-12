# Import packages
import torch
import numpy as np
from typing import List

# Define Quadratic Game Objects
class QuadraticGame:
    def __init__(self, n_dim: int, L_A: float,
                 mu_A: float, L_B: float,
                 L_C: float, mu_C: float):
        
        # function to generate symmetric matrix having eigenvalues in a specific range [mu, L]
        def generate_mx(mu:float, L:float, dim:int) -> List[List[float]]:
            evalues = np.random.uniform(mu, L, dim)
            evalues[0], evalues[-1] = mu, L
            rndm_mx = np.random.normal(0, 1, (dim, dim))
            _, Q = np.linalg.eig(rndm_mx.T @ rndm_mx)
            return Q @ np.diag(evalues) @ Q.T
        
        # generate the matrices A, B, C
        A = generate_mx(mu_A, L_A, n_dim)
        B = generate_mx(0, L_B, n_dim)
        C = generate_mx(mu_C, L_C, n_dim)

        # generate the vectors a, c
        a = np.random.normal(0, 1, n_dim)
        c = np.random.normal(0, 1, n_dim)

        # total model
        M = np.block([[A, B], [-B, C]])
        z = np.concatenate((a, c))

        # smoothness constant
        self.smoothness = np.sqrt(max(np.linalg.eig(M.T @ M)[0]))

        # optimal vector
        x_optimal = - np.linalg.inv(M) @ z
        
        # store A, B and C as ttoch tensors
        self.A = torch.tensor(A, dtype=torch.float32)
        self.B = torch.tensor(B, dtype=torch.float32)
        self.C = torch.tensor(C, dtype=torch.float32)
        
        # store vectors a and c as torch tensors
        self.a = torch.tensor(a, dtype=torch.float32)
        self.c = torch.tensor(c, dtype=torch.float32)
          
        # store optimal vector as a torch tensor
        self.x_optimal = torch.tensor(x_optimal, dtype=torch.float32)
     
    # objective function represents a quadratic optimization problem in the form of:
    # f(x1, x2) = 0.5 * x1^T * A * x1 + x1^T * B * x2 - 0.5 * x2^T * C * x2 + a^T * x1 - c^T * x2    
    def objective_function(self, x1: List[float], x2: List[float]) -> float:
        ### objective function
        return (.5 * torch.t(x1) @ self.A @ x1
                + torch.t(x1) @ self.B @ x2
                - .5 * torch.t(x2) @ self.C @ x2
                + torch.t(self.a) @ x1
                - torch.t(self.c) @ x2)
    
    # calculates the optimal distance between a given pair of vectors (x1, x2)
    # and the optimal vector (x_optimal)
    def opt_dist(self, x1: List[float], x2: List[float]) -> float:
        x = torch.cat((x1, x2))
        ### optimal distance
        return torch.norm(x - self.x_optimal) ** 2
    
    
# Quadratic Game with n_data functions
class QuadGame:
    def __init__(self, n_dim: int, n_data: int, 
                 L_A: float, mu_A: float, 
                 L_B: float, 
                 L_C: float, mu_C: float,
                 device=None):
        """
        Initializes a Quadratic Game with n_data functions.

        The class represents a collection of quadratic optimization problems, where each problem is defined by
        a pair of vectors (x1, x2) and matrices A, B, C, and vectors a, c. The objective function of each problem
        is given by:
        f(x1, x2) = 0.5 * x1^T * A * x1 + x1^T * B * x2 - 0.5 * x2^T * C * x2 + a^T * x1 - c^T * x2

        Parameters:
        n_dim (int): The dimensionality of the vectors x1 and x2.
        n_data (int): The number of quadratic optimization problems in the collection.
        L_A (float): The upper bound for the eigenvalues of matrix A.
        mu_A (float): The lower bound for the eigenvalues of matrix A.
        L_B (float): The upper bound for the eigenvalues of matrix B.
        L_C (float): The upper bound for the eigenvalues of matrix C.
        mu_C (float): The lower bound for the eigenvalues of matrix C.
        device (torch.device, optional): The device to store the tensors. If not provided, the tensors will be stored on the CPU.

        Attributes:
        device (torch.device): The device to store the tensors.
        A (torch.Tensor): A tensor of shape (n_data, n_dim, n_dim) containing the matrices A for each problem.
        B (torch.Tensor): A tensor of shape (n_data, n_dim, n_dim) containing the matrices B for each problem.
        C (torch.Tensor): A tensor of shape (n_data, n_dim, n_dim) containing the matrices C for each problem.
        a (torch.Tensor): A tensor of shape (n_data, n_dim) containing the vectors a for each problem.
        c (torch.Tensor): A tensor of shape (n_data, n_dim) containing the vectors c for each problem.
        x_optimal (torch.Tensor): A tensor of shape (2 * n_dim) containing the optimal vector for the collection of problems.
        Lmax (float): max(L1, L2) where Li is the smoothness of i th player.
        mu (float): The strong convexity constant of the problem.
        ell (float) : The cocoercivity constant of the problem.
        """
        # set the device
        self.device = device

        # function to generate symmetric matrix having eigenvalues in a specific range [mu, L]
        def generate_mx(mu:float, L:float, dim:int) -> np.ndarray:
            evalues = np.random.uniform(mu, L, dim)
            evalues[0], evalues[-1] = mu, L
            rndm_mx = np.random.normal(0, 1, (dim, dim))
            _, Q = np.linalg.eig(rndm_mx.T @ rndm_mx)
            return Q @ np.diag(evalues) @ Q.T

        A, B, C, a, c = [], [], [], [], []
        for _ in range(n_data):
            # generate the matrices A, B, C
            A.append(generate_mx(mu_A, L_A, n_dim))
            B.append(generate_mx(0, L_B, n_dim))
            C.append(generate_mx(mu_C, L_C, n_dim))

            # generate the matrices a, c
            a.append(np.random.normal(0, 1, n_dim))
            c.append(np.random.normal(0, 1, n_dim))

        # total model M, z
        M = np.block([[np.mean(A, axis = 0), np.mean(B, axis = 0)], 
                      [- np.mean(B, axis = 0), np.mean(C, axis = 0)]])
        z = np.concatenate((np.mean(a, axis = 0), np.mean(c, axis = 0)))

        # optimal vector
        x_optimal = - np.linalg.inv(M) @ z

        # problem constant
        self.Lmax = max(np.sqrt(max(np.linalg.eig(np.mean(A, axis = 0).T @ np.mean(A, axis = 0))[0])),
                        np.sqrt(max(np.linalg.eig(np.mean(C, axis = 0).T @ np.mean(C, axis = 0))[0])))
        self.mu = min(np.sqrt(min(np.linalg.eig(np.mean(A, axis = 0) @ np.mean(A, axis = 0).T)[0])),
                      np.sqrt(min(np.linalg.eig(np.mean(C, axis = 0) @ np.mean(C, axis = 0).T)[0])))
        
        eval_M = np.linalg.eig(M)[0]
        self.ell = 1/np.min(np.real(1/ eval_M[np.abs(eval_M) > 1e-5]))
        
        # store A, B and C as torch tensors and pass to device
        self.A = torch.tensor(A, dtype=torch.float32).to(self.device)
        self.B = torch.tensor(B, dtype=torch.float32).to(self.device)
        self.C = torch.tensor(C, dtype=torch.float32).to(self.device)

        # store vectors a and c as torch tensors
        self.a = torch.tensor(a, dtype=torch.float32).to(self.device)
        self.c = torch.tensor(c, dtype=torch.float32).to(self.device)
        
        # store optimal vector as torch tensor
        self.x_optimal = torch.tensor(x_optimal, dtype=torch.float32).to(self.device)
    
    def objective_function(self, x1: torch.Tensor, 
                           x2: torch.Tensor, 
                           index: List[int] = None) -> torch.Tensor:
        """
        Computes the objective function of a Quadratic Game.

        The objective function represents a quadratic optimization problem in the form of:
        f(x1, x2) = 0.5 * x1^T * A * x1 + x1^T * B * x2 - 0.5 * x2^T * C * x2 + a^T * x1 - c^T * x2

        Parameters:
        x1 (torch.Tensor): The first vector of the pair.
        x2 (torch.Tensor): The second vector of the pair.
        index (List[int], optional): A list of indices to compute the objective function for specific data points. 
                                     If not provided, the function computes the full objective function.

        Returns:
        torch.Tensor: The computed objective function value.
        """
        # pass to correct device
        x1, x2 = x1.to(self.device), x2.to(self.device)
        if index is not None:
            # compute the objective function for a specific data points
            return (.5 * torch.t(x1) @ torch.mean(self.A[index], dim = 0) @ x1
                    + torch.t(x1) @ torch.mean(self.B[index], dim = 0) @ x2
                    - .5 * torch.t(x2) @ torch.mean(self.C[index], dim = 0) @ x2
                    + torch.t(torch.mean(self.a[index], dim = 0)) @ x1
                    - torch.t(torch.mean(self.c[index], dim = 0)) @ x2)
        else:
            # compute the full objective function
            return (.5 * torch.t(x1) @ torch.mean(self.A, dim = 0) @ x1
                    + torch.t(x1) @ torch.mean(self.B, dim = 0) @ x2
                    - .5 * torch.t(x2) @ torch.mean(self.C, dim = 0) @ x2
                    + torch.t(torch.mean(self.a, dim = 0)) @ x1
                    - torch.t(torch.mean(self.c, dim = 0)) @ x2)
        
    def opt_dist(self, x1: torch.Tensor, 
                 x2: torch.Tensor) -> torch.Tensor:
        """
        Calculates the optimal distance between a given pair of vectors (x1, x2)
        and the optimal vector (x_optimal) in the context of a Quadratic Game.

        Parameters:
        x1 (torch.Tensor): The first vector of the pair.
        x2 (torch.Tensor): The second vector of the pair.

        Returns:
        torch.Tensor: The optimal distance between the given pair of vectors and the optimal vector.
        """
        # pass to correct device
        x1, x2 = x1.to(self.device), x2.to(self.device)
        x = torch.cat((x1, x2))
        # optimal distance
        return torch.norm(x - self.x_optimal) ** 2
    

class NPGame:
    def __init__(self, n_player: int, n_dim: int, n_data: int, 
                 L_A: float, mu_A: float,  
                 L_B: float, mu_B: float,
                 device: torch.device):
        """
        Initializes a N-player game with n_player players, each player having n_dim dimensions, 
        and n_data instances. The game is defined by matrices A, B, and vectors a.

        Parameters:
        n_player (int): The number of players in the game.
        n_dim (int): The dimensionality of the strategy space for each player.
        n_data (int): The number of instances or scenarios in the game.
        mu_A (float): The lower bound for the eigenvalues of matrix A.
        L_A (float): The upper bound for the eigenvalues of matrix A.

        mu_B (float): The lower bound for the eigenvalues of matrix B.
        L_B (float): The upper bound for the eigenvalues of matrix B.
        device (torch.device): The device to store the tensors.

        Attributes:
        A (torch.Tensor): A tensor of shape (n_player, n_data, n_dim, n_dim) containing the matrices A for each player and instance.
        B (torch.Tensor): A tensor of shape (n_player, n_player, n_data, n_dim, n_dim) containing the matrices B for each pair of players and instance.
        a (torch.Tensor): A tensor of shape (n_player, n_data, n_dim) containing the vectors a for each player and instance.
        n_player (int): The number of players in the game.
        device (torch.device): The device to store the tensors.
        x_optimal (torch.Tensor): A tensor of shape (n_player, n_dim) containing the optimal strategy for each player.
        """
        
        self.n_player = n_player
        self.device = device
        def generate_mx(mu:float, L:float, dim:int) -> np.ndarray:
            evalues = np.random.uniform(mu, L, dim)
            evalues[0], evalues[-1] = mu, L
            rndm_mx = np.random.normal(0, 1, (dim, dim))
            _, Q = np.linalg.eig(rndm_mx.T @ rndm_mx)
            return Q @ np.diag(evalues) @ Q.T

        # Initialize the tensors of matrices and vectors with zeros
        A, B = torch.zeros(n_player, n_data, n_dim, n_dim), torch.zeros(n_player, n_player, n_data, n_dim, n_dim) 
        a = torch.zeros(n_player, n_data, n_dim)

        # Store the matrices and vectors
        for idx in range(n_data):
            for player in range(n_player):
                A[player, idx] = torch.tensor(generate_mx(mu_A, L_A, n_dim)).to(self.device)
                a[player, idx] = torch.tensor(np.random.normal(0, 1,n_dim)).to(self.device)

            for player1 in range(n_player):
                for player2 in range(player1+1, n_player):
                    B[player1, player2, idx] = torch.tensor(generate_mx(mu_B, L_B, n_dim)).to(self.device)
                    B[player2, player1, idx] = - B[player1, player2, idx].transpose(-2, -1)

        # Compute the full M matrix and z vector
        M = torch.zeros(n_player * n_dim, n_player * n_dim)
        z = torch.zeros(n_player * n_dim)
        
        for player in range(n_player):
            z[player * n_dim:(player + 1) * n_dim] = torch.mean(a[player], dim=0)
            M[player * n_dim:(player + 1) * n_dim, player * n_dim:(player + 1) * n_dim] = torch.mean(A[player], dim=0)
            for player2 in range(n_player):
                if player != player2:
                    M[player * n_dim:(player + 1) * n_dim, player2 * n_dim:(player2 + 1) * n_dim] = torch.mean(B[player, player2], dim=0)

        # Solve M @ x_optimal = -z
        x_optimal_flat = torch.linalg.solve(M, -z)

        # Reshape x_optimal to (n_player, n_dim) shape
        x_optimal = x_optimal_flat.view(n_player, n_dim)

        # Compute Problem Constant
        
        # Compute strong monotonicity constant mu, L based on eigenvalues of the mean of A for each player
        mu_values, L_values = [], []
        for i in range(n_player):
            mean_A = torch.mean(A[i], dim=0)  # Average A over data points for each player
            mu_values.append(torch.linalg.matrix_norm(mean_A, ord = -2))  # Take the smallest real eigenvalue
            L_values.append(torch.linalg.matrix_norm(mean_A, ord = 2))
        self.mu = min(mu_values)
        self.Lmax = max(L_values)
        
        # mu_values, L_values = [], []
        # for i in range(n_player):
        #     mean_A = torch.mean(A[i], dim=0)  # Average A over data points for each player
        #     eigenvalues = torch.linalg.eigvals(mean_A)  # Get the eigenvalues of the mean_A matrix
        #     mu_values.append(torch.min(eigenvalues.real))  # Take the smallest real eigenvalue
        #     L_values.append(torch.max(eigenvalues.real))
        # self.mu = min(mu_values)  # Take the minimum over all players' mu values
        # self.Lmax = max(L_values)

        # Compute Lipschitz constant L
        self.L = torch.linalg.matrix_norm(M, ord = 2)
        
        # Pass to self
        self.M, self.z = M, z
        self.A, self.B, self.a = A.to(self.device), B.to(self.device), a.to(self.device) 
        self.x_optimal = x_optimal.to(self.device)
        
    def objective_function(self, player: int, x: torch.Tensor,
                           index: List[int] = None) -> torch.Tensor:
        """
        Computes the objective function for a specific player in a N-player game.

        Parameters:
        player (int): The index of the player for whom the objective function is computed.
        x (torch.Tensor): The tensor representing the strategy of all players.
        index (List[int], optional): A list of indices to compute the objective function for specific data points. 
                                     If not provided, the function computes the full objective function.

        Returns:
        torch.Tensor: The computed objective function value for the specified player.
        """
        # pass to correct device
        x = x.to(self.device)
        if index is not None:
            coupling_term  = 0
            for player2 in range(self.n_player):
                if player2 != player:
                    coupling_term += torch.t(x[player]) @ torch.mean(self.B[player][player2][index], dim = 0) @ x[player2]

            return (.5 * torch.t(x[player]) @ torch.mean(self.A[player][index], dim = 0) @ x[player]
                    + torch.t(torch.mean(self.a[player][index], dim = 0)) @ x[player]
                    + coupling_term)

        else:
            coupling_term  = 0
            for player2 in range(self.n_player):
                if player2 != player:
                    coupling_term += torch.t(x[player]) @ torch.mean(self.B[player][player2], dim = 0) @ x[player2]

            return (.5 * torch.t(x[player]) @ torch.mean(self.A[player], dim = 0) @ x[player]
                    + torch.t(torch.mean(self.a[player], dim = 0)) @ x[player]
                    + coupling_term)
            
        
    def opt_dist(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates the optimal distance between a given strategy tensor (x) and the optimal strategy tensor (x_optimal) in the context of a N-player game.

        Parameters:
        x (torch.Tensor): The tensor representing the strategy of all players.

        Returns:
        torch.Tensor: The optimal distance between the given strategy tensor and the optimal strategy tensor.
        """
        # pass to correct device
        x = x.to(self.device)
        return torch.norm(x - self.x_optimal) ** 2
            
    

class NCGame:
    def __init__(self, n_player: int, n_dim: int, device: torch.device):
        """
        Initializes a Nash-Cournot game with n_player players, each player having n_dim dimensions.

        Parameters:
        n_player (int): The number of players in the game.
        n_dim (int): The dimensionality of the strategy space for each player.
        device (torch.device): The device to store the tensors.
        """
        self.device = device
        
    def objective_function(self, player: int, x: torch.Tensor,
                           index: List[int] = None) -> torch.Tensor:
        """
        Computes the objective function for a specific player in a Nash-Cournot game.

        Parameters:
        player (int): The index of the player for whom the objective function is computed.
        x (torch.Tensor): The tensor representing the strategy of all players.
        index (List[int], optional): A list of indices to compute the objective function for specific data points.
        
        Returns:    
        Computes the objective function value for the specified player.
        """
        # pass to correct device
        x = x.to(self.device)
        return (1 + player) * 0.1 * x[player] - torch.sum(x) * x[player]
    
    def opt_dist(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates the optimal distance between a given strategy tensor (x) and the optimal strategy tensor (x_optimal) in the context of a Nash-Cournot game.

        Parameters:
        x (torch.Tensor): The tensor representing the strategy of all players.

        Returns:
        torch.Tensor: The optimal distance between the given strategy tensor and the optimal strategy tensor.
        """
        # pass to correct device
        x = x.to(self.device)
        return 
    
        
class NCGame_v2:
    def __init__(self, n_player: int, n_data: int,
                 n_dim: int, device: torch.device):
        """
        Initializes a Nash-Cournot game with n_player players, each player having n_dim dimensions.

        Parameters:
        n_player (int): The number of players in the game.
        n_dim (int): The dimensionality of the strategy space for each player.
        device (torch.device): The device to store the tensors.
        """
        self.device = device
        self.n_player = n_player
        self.n_dim = n_dim
        
        # Initialize the tensors of matrices and vectors with zeros
        A, B = torch.zeros(n_player, n_data, n_dim, n_dim), torch.zeros(n_player, n_player, n_data, n_dim, n_dim) 
        a = torch.zeros(n_player, n_data, n_dim)
        
        # Store the matrices and vectors
        for idx in range(n_data):
            for player in range(n_player):
                A[player, idx] = torch.tensor(np.random.normal(0, 1, n_dim)).to(self.device)
                a[player, idx] = torch.tensor(np.random.normal(0, 1, n_dim)).to(self.device)
                
            for player1 in range(n_player):
                for player2 in range(player1+1, n_player):
                    B[player1, player2, idx] = torch.tensor(np.random.normal(0, 1, n_dim)).to(self.device)
                    B[player2, player1, idx] = - B[player1, player2, idx].transpose(-2, -1)
        
        # Compute the full M matrix and z vector
        M = torch.zeros(n_player * n_dim, n_player * n_dim)
        z = torch.zeros(n_player * n_dim)
        
        for player in range(n_player):
            z[player * n_dim:(player + 1) * n_dim] = torch.mean(a[player], dim=0)
            M[player * n_dim:(player + 1) * n_dim, player * n_dim:(player + 1) * n_dim] = torch.mean(A[player], dim=0)
            for player2 in range(n_player):
                if player != player2:
                    M[player * n_dim:(player + 1) * n_dim, player2 * n_dim:(player2 + 1) * n_dim] = torch.mean(B[player, player2], dim=0)
                    
                    
        # Solve M @ x_optimal = -z
        x_optimal_flat = torch.linalg.solve(M, -z)

        # Reshape x_optimal to (n_player, n_dim) shape
        x_optimal = x_optimal_flat.view(n_player, n_dim)

        # Compute Problem Constant
        
        # Compute strong monotonicity constant mu, L based on eigenvalues of the mean of A for each player
        mu_values, L_values = [], []
        for i in range(n_player):
            mean_A = torch.mean(A[i], dim=0)  # Average A over data points for each player
            mu_values.append(torch.linalg.matrix_norm(mean_A, ord = -2))  # Take the smallest real eigenvalue
            L_values.append(torch.linalg.matrix_norm(mean_A, ord = 2))
        self.mu = min(mu_values)
        self.Lmax = max(L_values)
        
        # Compute Lipschitz constant L
        self.L = torch.linalg.matrix_norm(M, ord = 2)
        
        # Pass to self
        self.M, self.z = M, z
        self.A, self.B, self.a = A.to(self.device), B.to(self.device), a.to(self.device) 
        self.x_optimal = x_optimal.to(self.device)
        
        
    def objective_function(self, player: int, x: torch.Tensor,
                           index: List[int] = None) -> torch.Tensor:
        """
        Computes the objective function for a specific player in a N-player game.

        Parameters:
        player (int): The index of the player for whom the objective function is computed.
        x (torch.Tensor): The tensor representing the strategy of all players.
        index (List[int], optional): A list of indices to compute the objective function for specific data points. 
                                     If not provided, the function computes the full objective function.

        Returns:
        torch.Tensor: The computed objective function value for the specified player.
        """
        # pass to correct device
        x = x.to(self.device)
        if index is not None:
            coupling_term  = 0
            for player2 in range(self.n_player):
                if player2 != player:
                    coupling_term += torch.t(x[player]) @ torch.mean(self.B[player][player2][index], dim = 0) @ x[player2]

            return (.5 * torch.t(x[player]) @ torch.mean(self.A[player][index], dim = 0) @ x[player]
                    + torch.t(torch.mean(self.a[player][index], dim = 0)) @ x[player]
                    + coupling_term)

        else:
            coupling_term  = 0
            for player2 in range(self.n_player):
                if player2 != player:
                    coupling_term += torch.t(x[player]) @ torch.mean(self.B[player][player2], dim = 0) @ x[player2]

            return (.5 * torch.t(x[player]) @ torch.mean(self.A[player], dim = 0) @ x[player]
                    + torch.t(torch.mean(self.a[player], dim = 0)) @ x[player]
                    + coupling_term)
            
            
    def opt_dist(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates the optimal distance between a given strategy tensor (x) and the optimal strategy tensor (x_optimal) in the context of a N-player game.

        Parameters:
        x (torch.Tensor): The tensor representing the strategy of all players.

        Returns:
        torch.Tensor: The optimal distance between the given strategy tensor and the optimal strategy tensor.
        """
        # pass to correct device
        x = x.to(self.device)
        return torch.norm(x - self.x_optimal) ** 2
    
    

class RobotGame:
    def __init__(self, device: torch.device, n_player: int = 5):
        """
        Initializes a robot game with n_player players, each player having n_dim dimensions.

        The game is defined by cost functions for each player, where each player's cost depends on their own strategy
        and the strategies of other players. The objective is to minimize these cost functions.

        `Link to paper <https://www.sciencedirect.com/science/article/pii/S0005109824002061?ref=pdf_download&fr=RR-2&rr=8e115e45c9a1067c>`_
        """
        
        self.device = device
        self.n_player = n_player
        
        # Initialize the tensors of vectors with zeros
        self.c = torch.zeros(n_player).to(self.device)
        self.d = torch.zeros(n_player).to(self.device)
                
        for i in range(n_player):
            self.c[i] = 10 + (i + 1) / 6
            self.d[i] = (i + 1) / 6
            
        self.h = torch.tensor([[0, 5, -7, 9, -8], [-5, 0, -6, 2, -9],
                               [7, 6, 0, 7, -4], [-9, -2, -7, 0, -2],
                               [8, 9, 4, 2, 0]], dtype=torch.float32).to(self.device)
        self.q_target = torch.tensor([1, -4, 8, -9, 13], dtype=torch.float32).to(self.device)
        
        # write the operator as F(q) = Mq + z where M is a matrix and z is a vector
        # computation of matrix M
        M = torch.zeros((n_player, n_player), dtype=torch.float32).to(self.device)
        for i in range(n_player):
            for j in range(n_player):
                M[i, j] = self.c[i] + (self.n_player - 1) * self.d[i] if i == j else -self.d[i]
        self.M = M
        
        # compute Lipschitz constant
        self.L = torch.linalg.norm(M, ord=2)
        
        # compute strong monotonicity constant
        self.mu = torch.min(torch.linalg.eigvals(M).real)
                
        # compute maximum of Lipschitz constants of each player
        self.Lmax = torch.max(self.c + (self.n_player - 1) * self.d)
        
        # computation of vector z
        z = torch.zeros(n_player, dtype=torch.float32).to(self.device)
        for i in range(n_player):
            z[i] = - self.c[i] * self.q_target[i] - self.d[i] * torch.sum(self.h[i])        
        self.z = z
        
        # optimal strategy
        self.optimal_q = torch.linalg.solve(M, -z)
        
    def objective_function(self, player: int, q: torch.Tensor) -> torch.Tensor:
        """
        Computes the objective function for a specific player in a robot game.

        Parameters:
        player (int): The index of the player for whom the objective function is computed.
        q (torch.Tensor): The tensor representing the strategies of all players.

        Returns:
        torch.Tensor: The computed objective function value for the specified player.
        """
        # pass to correct device
        q = q.to(self.device)
        
        term2 = 0
        for j in range(self.n_player):
            term2 += torch.square(q[player] - q[j] - self.h[player][j])
        
        return (0.5 * self.c[player] * torch.square(q[player] - self.q_target[player]) 
                + 0.5 * self.d[player] * term2)
        
        
    # function to compute the optimal distance between a given strategy tensor (q)
    # and the optimal strategy tensor (optimal_q) in the context of a robot game
    def opt_dist(self, q: torch.Tensor) -> torch.Tensor:
        """
        Calculates the optimal distance between a given strategy tensor (q) and the optimal strategy tensor (optimal_q) in the context of a robot game.

        Parameters:
        q (torch.Tensor): The tensor representing the strategies of all players.

        Returns:
        torch.Tensor: The optimal distance between the given strategy tensor and the optimal strategy tensor.
        """
        # pass to correct device
        q = q.to(self.device)
        return torch.norm(q - self.optimal_q) ** 2
    
    