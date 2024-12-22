import numpy as np
import torch
import matplotlib.pyplot as plt
import tqdm

from ..generate_burgers import generate_initial_u, generate_control_sequence, burgers_update


class Burgers:
    def __init__(self,
                    n=128,
                    m=128,
                    T=8,
                    x_range=[-5, 5],
                    energy_penalty=0.01,
                    device='cpu',
                    seed=12345
                    ):
        '''
        n: number of nodes
        m: number of driver nodes
        T: Number of time steps
        T_sim: Total simulation time (a physical time)
        energy_penalty: penalty coefficent for energy. Energy is calculated as the sum of the square of the action vector.
        '''
        super().__init__()
        assert n==m, "Invalid input"
        self.num_nodes = n
        self.num_driver = m
        self.state_dim = n
        self.action_dim = m
        self.T = T
        self.energy_penalty = energy_penalty
        self.x_range = x_range
        self.x=np.linspace(x_range[0],x_range[1],n) # spatial grid

        self.device = device
        self.seed = seed

        self.delta_t = 0.1 # physical time length
        self.dx = (x_range[1]-x_range[0])/n # spatial length, used as dx in the numerical solution of differential equation. (Assume equal spacing)
        self.current_state=np.zeros(n)
        self.t=0 # Current time index. An integer between 0 and T-1.

        np.random.seed(self.seed)


    def add_target_state(self, target_state): # TODO
        '''
        Add a target state to the environment.
        '''
        self.target_state = target_state

    def calculate_state_reward(self, state):
        '''
        Calculate the reward for a given state. This is not the total reward, just the state part.
        '''
        return -np.sum(np.square(state - self.target_state))

    def calculate_action_reward(self, action):
        '''
        Calculate the reward for a given action. This is not the total reward, just the action part.
        '''
        return -np.sum(np.square(action))

    def reset(self, start=None):
        '''
        Reset the environment to a initial state.
        '''
        if start is not None: # If the initial state is given, use it.
            self.current_state = start
            self.t = 0
            return self.current_state
        else: # If the initial state is not given, randomly initialize the state.
            self.current_state = np.random.randn(self.num_nodes)
            self.t = 0
            return self.current_state

    def step(self, action):
        '''
        action: (m,), the control signal input
        '''
        next_state = burgers_update(self.current_state, action, self.dx, self.delta_t)
        reward = self.calculate_state_reward(next_state)+self.calculate_action_reward(action)*self.energy_penalty
        done = False if self.t < self.T-1 else True
        self.current_state = next_state # update current state
        self.t += 1 # update time index
        return next_state, reward, done

    def get_state(self):
        return self.current_state

    def get_dataset(self, nu=0.01): # TODO # 暂时没用
        '''
        Generate a dataset of state-action pairs.
        '''
        Y_bar = []  # Holds states from y_0 to y_{T-1} # The final Y_bar should be a numpy array of shape (N, T, n)
        Y_f = []    # Holds final state y_T # The final Y_f should be a numpy array of shape (N, n)
        U = []      # Holds control sequence u_0 to u_{T-1} # The final U should be a numpy array of shape (N, T, n)

        # Sample initial state u(0, x)
        u = generate_initial_u(self.x)

        # Sample control sequence w(t, x) at 10 time steps
        w = []
        for t_idx in range(self.T):
            t = t_idx * 0.1  # Time stamps are from 0.0 to 0.9 in steps of 0.1
            w_t = generate_control_sequence(self.x, t)
            w.append(w_t)

        # Initialize lists to store the state trajectory
        y = [u]  # Starting state

        # Generate the state trajectory using the Burgers equation
        for t_idx in range(1, self.T):
            u_prev = y[-1]  # Previous state is the last element in the list
            u_current = u_prev + self.delta_t * burgers(u_prev, w[t_idx-1], self.delta_t, nu=nu)
            y.append(u_current)
            # print(u_current.max())

        # Store the data
        Y_bar.append(np.stack(y[:-1]))  # Y_bar contains states from y_0 to y_{T-1}
        Y_f.append(y[-1])               # Y_f contains the final state y_T
        U.append(np.stack(w[:-1]))           # U contains control sequence w_0 to w_{T-1}

        Y_bar=np.array(Y_bar) # (N, T, n)
        Y_f=np.array(Y_f) # (N, n)
        U=np.flip(np.array(U), axis=1) # (N, T, n)

        state_loss = np.sum(np.square(Y_bar - Y_f[:, np.newaxis, :]), axis=-1) # (N, T)
        energy_loss = np.sum(np.square(U), axis=-1) * self.energy_penalty # (N, T)
        rewards = -state_loss - energy_loss # (N, T)
        done=np.zeros_like(rewards) # (N, T)
        done[:,-1]=1
        done=done.astype(bool) # (N, T)
        data_dict = { # d4rl API
            'observations': Y_bar,
            'actions': U,
            'rewards': rewards,
            'terminals': done,
            'timeouts': done,
            'Y_f': Y_f,
            'meta_data': {
                'num_nodes': self.num_nodes, 
                'input_dim': self.num_nodes, 
                'control_horizon': self.T - 1,
                'num_samples': num_samples,
            },
        }

        return data_dict

































