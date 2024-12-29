import numpy as np
import torch
import matplotlib.pyplot as plt
import tqdm

from generate_burgers import generate_initial_y, generate_control_sequence, burgers_update, get_trajectory, roughness


class Burgers:
    def __init__(self,
                    x_range=(-5,5),
                    nx = 128, # Number of spatial nodes (grid points)
                    nt = 500, # Number of time steps
                    dt= 0.001, # Temporal interval
                    energy_penalty=0.01,
                    device='cpu',
                    seed=12345
                    ):
        '''
        Initialize the environment.

        x_range: the range of spatial grid.
        nx: number of spatial nodes (grid points).
        nt: number of time steps.
        dt: temporal interval.
        energy_penalty: penalty coefficent for energy. Energy is calculated as the sum of the square of the action vector.
        '''
        super().__init__()
        
        self.nt = nt
        self.dt = dt
        self.T = nt * dt
        
        self.x_range = x_range
        self.nx = nx
        self.dx = (x_range[1]-x_range[0])/nx # spatial length, used as dx in the numerical solution of differential equation. (Assume equal spacing)
        self.x=np.linspace(x_range[0],x_range[1],nx) # spatial grid

        self.energy_penalty = energy_penalty

        self.device = device
        self.seed = seed

        self.current_state=np.zeros(nx)
        self.current_t_idx=0 # Current time index

        np.random.seed(self.seed)

    def reset(self, start=None):
        '''
        Reset the environment to a initial state.
        '''
        if start is not None: # If the initial state is given, use it.
            self.current_state = start
        else: # If the initial state is not given, randomly initialize the state.
            self.current_state = generate_initial_y(self.x)
        self.current_t_idx=0 # Reset time index
        return self.current_state

    def set_target_state(self, target_state):
        '''
        Set the target state for the environment.
        '''
        self.target_state = target_state

    def step(self, action):
        '''
        action: (m,), the control signal input
        '''
        assert hasattr(self, "target_state"), "You need to set the target state before taking a step."
        next_state = burgers_update(self.current_state, action, self.dt, self.dx)
        reward = -np.square(action).mean() * self.energy_penalty - ((next_state - self.target_state)**2).mean()
        if roughness(next_state)/self.dx > 10:
            done = True
            reward = -10000
        elif self.current_t_idx +1 == self.nt:
            done = True
        else:
            done = False
        self.current_t_idx += 1
        return next_state, reward, done

    def get_state(self):
        return self.current_state

    def compute_trajectory(self): # 暂时没用
        '''
        Generate a trajectory of the system using the given control sequence.
        '''
        assert self.t==0, "You need to reset the environment before generating a trajectory."
        y0 = self.current_state
        state_trajectory, action_trajectory, final_state, timeouts, terminals, rewards = \
            get_trajectory(y0, nt=self.nt, dt=self.dt, nx=self.nx, dx=self.dx)
        self.state_trajectory = state_trajectory
        self.action_trajectory = action_trajectory
        self.final_state = final_state
        self.timeouts = timeouts
        self.terminals = terminals
        self.rewards = rewards
        self.t=self.nt
































