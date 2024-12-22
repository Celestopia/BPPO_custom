import numpy as np
import matplotlib.pyplot as plt
import pickle
import tqdm

def burgers_update(y, u, dx, dt, nu=0.01):
    """
    Return the updated state vector y using the Lax-Friedrichs algorithm with viscosity.

    :param y: (n,), state vector.
    :param u: (m,), control signal.
    :param dx: the spatial step size.
    :param dt: the temporal step size.
    :param nu: kinematic viscosity.

    :return: (n,), the updated state vector.
    """
    assert type(y) == np.ndarray and type(u) == np.ndarray, "y and u must be numpy arrays"
    assert y.ndim == 1 and u.ndim == 1, "y and u must be 1 dimensional arrays"
    assert y.shape == u.shape, "y and u must have the same shape"
    
    y_new = np.copy(y)
    for i in range(1, len(y) - 1): # Loop over the interior points and apply the Lax-Friedrichs scheme with viscosity
        convective_term = (y[i + 1] - y[i - 1]) / (2 * dx) # Convective term (advection)
        diffusive_term = (y[i + 1] - 2 * y[i] + y[i - 1]) / (dx**2) # Diffusive term (viscosity)
        y_new[i] = 0.5 * (y[i+1] + y[i-1]) - (dt/(2*dx))*y[i]*convective_term + nu*dt*diffusive_term + 0.1*u[i]*dt # Lax-Friedrichs update with viscosity
    # Apply 0 boundary conditions
    y_new[0] = 0
    y_new[-1] = 0
    return y_new


def generate_initial_u(x):
    """
    Generate the initial state u(0, x) as a superposition of two Gaussian functions.
    """
    # Sampling parameters for the Gaussian functions
    a1 = np.random.uniform(0, 2)
    a2 = np.random.uniform(-2, 0)
    mu1 = np.random.uniform(-1, 1)
    mu2 = np.random.uniform(-0.5, 0.5)
    sigma1 = np.random.uniform(0.3, 0.6)
    sigma2 = np.random.uniform(0.4, 0.8)
    
    u = a1 * np.exp(-((x - mu1)**2) / (2 * sigma1**2)) + a2 * np.exp(-((x - mu2)**2) / (2 * sigma2**2))
    u[0] = 0
    u[-1] = 0
    return u


def generate_control_sequence(x, t):
    """
    Generate control sequence w(t, x) as a superposition of 8 Gaussian functions.
    """
    w = np.zeros_like(x)
    for i in range(8):
        ind = np.random.binomial(1, 0.5)
        if i==0:
            ai = np.random.uniform(-1.5,1.5)
        else:
            if ind:
                ai = np.random.uniform(-1.5,1.5)
            else:
                ai = 0
        b1_i = np.random.uniform(0, 1)
        b2_i = np.random.uniform(0, 1)
        sigma1_i = np.random.uniform(0.05, 0.2)
        sigma2_i = np.random.uniform(0.05, 0.2)
        w += ai * np.exp(-((x - b1_i)**2) / (2 * sigma1_i**2)) * np.exp(-((t - b2_i)**2) / (2 * sigma2_i**2))
    return w


def roughness(y):
    '''
    The average absolute difference between adjacent points in the input vector.
    '''
    assert type(y) == np.ndarray, "y must be a numpy array"
    assert y.ndim == 1, "y must be a 1 dimensional array"
    return np.sum(np.abs(y[1:] - y[:-1])) / (y.shape[0] - 1)


def load_burgers(
        x_range=(-5,5),
        nt = 500, # Number of time steps
        nx = 128, # Number of spatial nodes (grid points)
        dt= 0.001, # Temporal interval
        N = 3, # Number of samples (trajectories) to generate
        visualize=False # Whether to show the animation of state trajectory evolution
        ):
    dx = (x_range[1]-x_range[0])/nx # Spatial interval
    x = np.linspace(*x_range, nx) # Initialize the spatial grid
    y = generate_initial_u(x) # Set the initial condition
    n = nx # state dimension. In this case, the state is all function values at the spatial grid points, so n=nx.
    
    Y_bar_list = [] # Holds states trajectories of N samples
    Y_f_list = [] # Holds final state of N samples
    U_list = [] # Holds control sequence trajectories of N samples
    
    # Simulate the system for N samples
    for _ in tqdm.tqdm(range(N), desc="Generating samples"):
        y_list = [] # Holds states from y_0 to y_{nt}
        u_list = [] # Holds control sequence u_0 to u_{nt-1}
        
        y = generate_initial_u(x) # Sample initial state u(0, x)
        y_list.append(y) # Append the initial state to Y_bar

        # Sample control sequence w(t, x) at 10 time steps
        w = []
        for t_idx in range(nt): # Iterate over time steps
            u=generate_control_sequence(y,t_idx*dt) # (n,), Control vector at time t_idx*dt
            y_new=burgers_update(y, u, dx, dt) # (n,), Updated state vector at time t_idx*dt
            u_list.append(u) # u_{t_idx}
            y_list.append(y_new) # y_bar_{t_idx+1}
            y=y_new # Update y
        
        # Store the data
        Y_bar_list.append(np.stack(y_list[:-1])) # Y_bar contains states from y_0 to y_{nt-1}
        Y_f_list.append(y) # Y_f contains the final state y_{nt}
        U_list.append(np.stack(u_list[:])) # U contains control sequence u_0 to u_{nt-1}
    
    # Set the observations, actions, and final states
    Y_bar=np.array(Y_bar_list) # (N, nt, n)
    U=np.array(U_list) # (N, nt, n)
    Y_f=np.array(Y_f_list) # (N, n)

    # Set timeout flag
    timeouts = np.zeros((N,nt), dtype=bool) # (N, nt)
    timeouts[:,-1] = True # Set the last time step of each trajectory as timeout
    
    # Set terminal flag
    terminals = np.zeros((N,nt), dtype=bool) # (N, nt)
    for i in tqdm.tqdm(range(N), desc="Setting terminal flags"): # Iterate over trajectories
        for t_idx in range(nt): # Iterate over time steps
            if roughness(Y_bar[i,t_idx])/dx > 10: # If the roughness is so large that the average first derivative among all spatial nodes is greater that 10, set it as terminal
                terminals[i,t_idx:] = True # Set all subsequent time steps of this trajectory as terminal
                break

    # Set rewards
    rewards = np.zeros((N,nt)) # (N, nt)
    for i in tqdm.tqdm(range(N), desc="Setting rewards"): # Iterate over trajectories
        if True in terminals[i]: # If the trajectory diverges (terminates early)
            terminal_timestep=list(terminals[i]).index(True) # Get the index of the first terminal time step
            roughness_turn_timestep=None # The last time step at which the roughness turns from going smaller to going larger
            for t_idx in reversed(range(1, terminal_timestep)): # Iterate over time steps
                if roughness(Y_bar[i,t_idx+1])/dx < roughness(Y_bar[i,t_idx])/dx: # Find the last time step at which the roughness turns from going smaller to going larger
                    Y_f[i]=Y_bar[i,t_idx] # (n,); The final state is reset to this time step
                    roughness_turn_timestep=t_idx
                    break
            rewards[i,:roughness_turn_timestep] = \
                -((Y_bar[i,:roughness_turn_timestep] - Y_f[i])**2).mean(axis=-1) / \
                (nt*(Y_f[i]**2).mean()) # The rewards before the terminal time step are set as the negative L2 distance between the final state and the current state
            rewards[i,roughness_turn_timestep:] = -np.inf # The rewards after the terminal time step are set as -inf
        else:
            rewards[i,:] = \
                -((Y_bar[i,:] - Y_f[i])**2).mean(axis=-1) / \
                (nt*(Y_f[i]**2).mean()) # If the trajectory converges (doesn't terminate early), set the reward as the negative L2 distance between the final state and the current state
            

    # Save the data into a dictionary
    data_dict = { # d4rl API
            'observations': Y_bar,
            'actions': U,
            'rewards': rewards,
            'terminals': terminals,
            'timeouts': timeouts,
            'Y_f': Y_f,
            'meta_data': {
                'num_nodes': nx, 
                'input_dim': nx, 
                'control_horizon': nt-1,
                'num_samples': N,
            },
        }
    print("Y_bar shape: ", Y_bar.shape) # (N, nt, n)
    print("Y_f shape: ", Y_f.shape) # (N, n)
    print("U shape: ", U.shape) # (N, nt, n)
    print("Terminals shape: ", terminals.shape) # (N, nt)
    print("Timeouts shape: ", timeouts.shape) # (N, nt)
    print("Rewards shape: ", rewards.shape) # (N, nt)
    #data_name = f'burger_{n}_{n}_{T-1}_{N}_{delta_t}.pkl'

    #plt.plot(rewards[0])
    #plt.show()
    if visualize == True:
        visualize_state_trajectory(x, Y_bar[0], dt, nt)
    return data_dict

def visualize_state_trajectory(x, Y, dt, nt):
    """
    Visualize the state trajectories of a sample.

    :param x: (nx,), the spatial grid.
    :param Y: (T-1, nx), the state trajectories.
    :param dt: the temporal interval.
    :param nt: the number of time steps.
    """
    from matplotlib.animation import FuncAnimation
    # Set up the figure and axis for plotting
    fig, ax = plt.subplots()
    line, = ax.plot(x, Y[0], label="Burgers equation solution")
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlim(x.min(), x.max())
    ax.set_title("Burgers Equation Evolution")
    ax.set_xlabel("x")
    ax.set_ylabel("u(x,t)")
    time_text = ax.text(0.5, 0.95, "",
                        transform=ax.transAxes,
                        horizontalalignment='center',
                        verticalalignment='center')

    # Animation update function
    def update(frame):
        line.set_ydata(Y[frame])
        time_text.set_text(f'Current Time: {frame * dt:.3f}/{nt*dt:.3f}')  # 更新时间提示
        return line, time_text

    # Create the animation
    ani = FuncAnimation(fig, update, frames=nt, blit=True, interval=50)
    plt.show()

if __name__ == '__main__':
    load_burgers(visualize=True)
    



