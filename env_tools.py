import numpy as np
import torch
import networkx as nx
from matplotlib import pyplot as plt


def load_env(env_name):
    if env_name == 'burger':
        n=8
        m=n
        T=16
        C=np.identity(n)
        return Burger(C,m,n, T)
    else:
        raise ValueError('Invalid environment name')


def burger(t, delta_t, y, u): 
    """
    1D Burgers' equation for fluid dynamics simulation.
    y: n x 1 (state vector)
    u: n x 1 (control input)
    t: current time step
    delta_t: time step size
    """
    nu = 0.01  # Kinematic viscosity (can be adjusted)
    # dydt = -y * (np.roll(y, -1) - np.roll(y, 1)) / (2 * delta_t) + nu * (np.roll(y, -1) - 2 * y + np.roll(y, 1)) / delta_t**2
    # fill with initial and final values
    y_b = np.roll(y, -1)
    y_b[-1] = y[-1]
    y_a = np.roll(y, 1)
    y_a[0] = y[0]
    dydt = -y * (y_b - y_a) / (2 * delta_t) + nu * (y_b - 2 * y + y_a) / delta_t**2
    return dydt + u  # Adding control influence to the state evolution

def null_space(A, rtol=1e-10):
    if A.numel() == 0:
        return torch.empty(A.shape[1], 0)
    u, s, v = torch.svd(A)
    rank = (s > rtol * s.max()).sum().item()
    return v[:, rank:]

def compute_optimal_input(n, m, T, X0, U, X_f, y_0, y_f, X_bar):

    # Compute K_X0 and K_U
    K_X0 = null_space(X0, 1e-10)
    K_U = null_space(U, 1e-10)

    # Compute xf_c
    if K_U.shape[1] > 0:
        xf_c = y_f - (X_f @ K_U @ torch.pinverse(X0 @ K_U, rcond=1e-10)) @ y_0
    else:
        xf_c = y_f

    # Update U, X_bar, X_f
    if K_X0.shape[1] > 0:
        U = U @ K_X0
        X_bar = X_bar @ K_X0
        X_f = X_f @ K_X0

    # Ensure the dimensions of Q and R are compatible with X_bar and U
    Q_dim = X_bar.shape[1]
    R_dim = U.shape[1]
    Q = 50 * torch.eye(Q_dim)
    R = torch.eye(R_dim)

    # Compute data-driven input
    K_f = null_space(X_f, 1e-10)
    if K_f.shape[1] > 0:
        L = torch.linalg.cholesky(X_bar.T @ Q @ X_bar + U.T @ R @ U, upper=False)
        W, S, V = torch.svd_lowrank(L @ K_f, q=min(m * (T - 1) - n, K_f.shape[1]))
        u_opt = U @ torch.pinverse(X_f, rcond=1e-10) @ xf_c - U @ K_f @ torch.pinverse(W @ S @ V.T, rcond=1e-10) @ L @ torch.pinverse(X_f, rcond=1e-10) @ xf_c
    else:
        u_opt = U @ torch.pinverse(X_f, rcond=1e-10) @ xf_c

    return u_opt

class EnvBase:
    def __init__(self):
        pass
    def reset(self):
        pass
    def step(self):
        pass
    def get_state(self):
        pass


class Burger(EnvBase):
    def __init__(self, C,m,n, T, device='cpu'):
        super().__init__()
        if type(C)==np.ndarray or type(C)==np.matrix:
            self.C = torch.tensor(C, dtype=torch.float32).to(device)
        else:
            self.C = C.to(device)
        self.num_nodes = n
        self.num_driver = m
        self.num_observation = n
        assert n==m, "Invalid input"
        self.max_T = T #action len
        self.device = device
        self.delta_t = 1
        self.reset()
    
    def reset(self, start=None):
        theta1 = np.zeros(self.num_nodes) # initial phase
        # theta2 = np.mod(4 * np.pi * np.arange(self.num_nodes) / self.num_nodes, 2 * np.pi) # final phase
        self.omega = torch.zeros(self.num_nodes).to(self.device)
        if start is None:
            self.current_state = torch.tensor(theta1, dtype=torch.float32).to(self.device)
        else:
            self.current_state = start
        self.T = 0
        return self.current_state, self.current_state, self.T
    
    def step(self, u):
        terminal = False

        if self.T == self.max_T*self.delta_t:
            terminal = True
        assert len(u) == self.num_driver, f"Invalid input, {len(u)},{self.num_driver}"
        # TODO: implement the burger's equation
        
        out = burger(self.T, self.delta_t, self.current_state.cpu().numpy(), u.cpu().numpy())
        
        self.T += self.delta_t
        if not isinstance(out,torch.Tensor):
            out = torch.tensor(out, dtype=torch.float32).to(u.device)
        self.current_state = out
        return torch.matmul(self.C, self.current_state), self.current_state, terminal
    
    def get_state(self):
        return self.current_state
    
    def from_actions_to_obs(self, actions, start=None):
        assert len(actions) == self.max_T, f"Invalid actions, {len(actions)},{self.max_T}"
        if start is not None:
            self.reset(start)
        else:
            self.reset()
        observations = []
        for a in actions:
            obs, _, _ = self.step(a)
            observations.append(obs)
        return torch.stack(observations).to(self.device)
    
    def from_actions_to_obs_longer(self, actions, start=None, continues=2):
        assert len(actions) == self.max_T, "Invalid actions"
        if start is not None:
            self.reset(start)
        else:
            self.reset()
        observations = []
        for a in actions:
            obs, _, _ = self.step(a)
            observations.append(obs)
        for i in range(continues*self.max_T):
            obs, _, _ = self.step(torch.zeros(self.num_driver))
            observations.append(obs)
        return torch.stack(observations).to(self.device)
    
    def from_actions_to_obs_direct(self, actions, start=None):
        return self.from_actions_to_obs(actions, start)
    
    def calculate_model_based_control(self, y_f):
        # Minimum-energy model-based control
        # print("Not implemented")
        return torch.zeros(self.max_T, self.num_driver), torch.zeros(self.num_nodes)
    
    def calculate_data_driven_control(self, y_f):

        U = torch.randn(self.num_driver * self.max_T, 80).to(self.device) # 
        Y = torch.matmul(self.C_o, U)
        u_dd = torch.pinverse(Y @ torch.pinverse(U)) @ y_f
        u_dd_r = u_dd.reshape(self.max_T, self.num_driver).flip([0])
        u_dd_approx = U @ torch.pinverse(Y) @ y_f
        u_dd_approx_r = u_dd_approx.reshape(self.max_T, self.num_driver).flip([0])
        y_f_hat = self.from_actions_to_obs_direct(u_dd_r)
        y_f_hat_approx = self.from_actions_to_obs_direct(u_dd_approx_r)
        
        return u_dd_r, y_f_hat, u_dd_approx_r, y_f_hat_approx
    
    def calculate_data_driven_control_with_data(self, y_f, U, Y):
        u_dd = torch.pinverse(Y @ torch.pinverse(U)) @ y_f
        u_dd_r = u_dd.reshape(self.max_T, self.num_driver).flip([0])
        u_dd_approx = U @ torch.pinverse(Y) @ y_f
        u_dd_approx_r = u_dd_approx.reshape(self.max_T, self.num_driver).flip([0])
        y_f_hat = self.from_actions_to_obs_direct(u_dd_r)
        y_f_hat_approx = self.from_actions_to_obs_direct(u_dd_approx_r)
        
        return u_dd_r, y_f_hat, u_dd_approx_r, y_f_hat_approx
    
    def calculate_data_driven_control_with_data_complete(self, y_f, y_0, U, Y_bar, Y_f):
        X_0 = Y_bar[:, 0].reshape(Y_bar.shape[0], -1).transpose()
        X_bar = Y_bar[:, 1:].reshape(Y_bar.shape[0], -1).transpose()
        X_f = Y_f.reshape(Y_f.shape[0], -1).transpose()
        U = U.reshape(U.shape[0], -1).transpose()

        # change to torch tensor
        X_0 = torch.tensor(X_0, dtype=torch.float32).to(self.device)
        X_bar = torch.tensor(X_bar, dtype=torch.float32).to(self.device)
        X_f = torch.tensor(X_f, dtype=torch.float32).to(self.device)
        U = torch.tensor(U, dtype=torch.float32).to(self.device)
        y_0 = torch.tensor(y_0, dtype=torch.float32).to(self.device)
        y_f = torch.tensor(y_f, dtype=torch.float32).to(self.device)
        
        u_dd = compute_optimal_input(self.num_nodes, self.num_driver, self.max_T, X_0, U, X_f, y_0, y_f, X_bar)
        u_dd_r = u_dd.reshape(self.max_T, self.num_driver).flip([0])
        y_f_hat = self.from_actions_to_obs_direct(u_dd_r, y_0)
        return u_dd_r, y_f_hat
    
    def calculate_C_0(self):
        # print("Not implemented")
        pass