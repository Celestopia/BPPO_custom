from generate_burgers import load_burgers


x_range = [-5.0, 5.0]
state_dim = 100
dt = 0.001
dataset = load_burgers(
                x_range=x_range,
                nt = 500, # Number of time steps
                nx = state_dim, # Number of spatial nodes (grid points)
                dt= dt, # Temporal interval
                N = 1, # Number of samples (trajectories) to generate
                visualize=True # Whether to show the animation of state trajectory evolution
                )