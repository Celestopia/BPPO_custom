import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os
import sys
import tqdm
# Modify the working path so that this.ipynb file can import other modules like in the root directory
current_dir = os.path.dirname(os.path.abspath('__file__'))
sys.path.append(os.path.join(current_dir, '..'))


# Load the dataset
dataset_names=["burgers", "kuramoto", "power", "ip"]
dataset_name=dataset_names[3] # can be changed

if dataset_name=="burgers":
    test_data_path=os.path.join("datasets", "burgers_50_10_128.pkl")
elif dataset_name=="kuramoto":
    test_data_path=os.path.join("datasets", "kuramoto_100_15_8_8.pkl")
elif dataset_name=="power":
    test_data_path=os.path.join("datasets", "power_100_31_18_9.pkl")
elif dataset_name=="ip":
    test_data_path=os.path.join("datasets", "inverted_pendulum_100_127_2_1.pkl")

test_data_dict=pickle.load(open(test_data_path,"rb"))
print(f"Dataset: {dataset_name}")
print("Test data loaded from: ", test_data_path)

Y_bar=test_data_dict['data']['Y_bar']
Y_f=test_data_dict['data']['Y_f']
U=test_data_dict['data']['U']
print("Y_bar shape: ", Y_bar.shape)
print("Y_f shape: ", Y_f.shape)
print("U shape: ", U.shape)

N=Y_bar.shape[0]
nt=Y_bar.shape[1]
state_dim=Y_bar.shape[2]



def visualize_state_trajectory(x, Y, dt, nt, frame_interval=20):
    r"""
    Visualize the state trajectories of a sample.
    The total number of frames is nt.

    :param x: (nx,), the spatial grid.
    :param Y: (T, nx), the state trajectory.
    :param dt: the temporal interval.
    :param nt: the number of time steps.
    :param frame_interval: the frame interval (in milliseconds) of the animation.
    """
    from matplotlib.animation import FuncAnimation

    # Set up the figure and axis for plotting
    fig, ax = plt.subplots(figsize=(9,6))
    line, = ax.plot(x, Y[0], label="State")
    #ax.set_ylim(-1.5, 1.5)
    ax.set_xlim(x.min(), x.max())
    ax.set_title("State Trajectory")
    ax.set_xlabel("x")
    ax.set_ylabel("u(x,t)")
    time_text = ax.text(0.5, 0.95, "",
                        transform=ax.transAxes,
                        horizontalalignment='center',
                        verticalalignment='center')

    # Animation update function
    def update(frame):
        line.set_ydata(Y[frame])
        time_text.set_text(f'Current Time: {frame * dt:.3f}/{nt*dt:.3f}')
        return line, time_text

    # Create the animation
    ani = FuncAnimation(fig, update, frames=nt, blit=True, interval=frame_interval)
    plt.legend()
    plt.show()

# Plot the state trajectory
x=np.arange(state_dim)
state_trajectory=Y_bar[0] # Select a trajectory to visualize. Can be changed.
dt=0.01 # Physical time interval. Can be changed.
frame_interval=50 # Animation frame interval (in milliseconds). Can be changed.

visualize_state_trajectory(x, state_trajectory, dt, nt, frame_interval=frame_interval)

