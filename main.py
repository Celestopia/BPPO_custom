import numpy as np
import torch
import matplotlib.pyplot as plt
import tqdm
import os
import random
import time
import pickle
from tensorboardX import SummaryWriter
import warnings
warnings.filterwarnings("ignore")
print("\n===================================START===================================\n")

# ----------------------------------------- 1. Hyperparameter Settings -------------------------------------------------------------
import argparse
parser = argparse.ArgumentParser()

# Experiment Settings
parser.add_argument('--dataset_name', type=str, default='ip', help='Name of the dataset. Choices are ["burgers", "kuramoto", "ip", "power"]')
parser.add_argument('--seed', type=int, default=20250112, help='Random seed')
parser.add_argument('--log_freq', type=int, default=1000, help='Frequency of logging to tensorboard')
parser.add_argument('--result_dir', type=str, default='logs111111', help='Directory to save all results')
parser.add_argument('--save_fig', type=bool, default=True, help='Whether to save figures of final states')
#parser.add_argument('--train_data_path', type=str, default=os.path.join('datasets', 'kuramoto_99900_15_8_8.pkl'), help='Path to the training dataset')
#parser.add_argument('--test_data_path', type=str, default=os.path.join('datasets', 'kuramoto_100_15_8_8.pkl'), help='Path to the testing dataset')
parser.add_argument('--train_data_path', type=str, default=os.path.join('datasets','inverted_pendulum_99900_127_2_1.pkl'), help='Path to the training dataset')
parser.add_argument('--test_data_path', type=str, default=os.path.join('datasets','inverted_pendulum_100_127_2_1.pkl'), help='Path to the testing dataset')
#parser.add_argument('--train_data_path', type=str, default=os.path.join('datasets','power_99900_31_18_9.pkl'), help='Path to the training dataset')
#parser.add_argument('--test_data_path', type=str, default=os.path.join('datasets','power_100_31_18_9.pkl'), help='Path to the testing dataset')
#parser.add_argument('--train_data_path', type=str, default=os.path.join('datasets','burgers_2000_10_128_1735571539.pkl'), help='Path to the training dataset')
#parser.add_argument('--train_data_path', type=str, default=os.path.join('datasets','burgers_90000_10_128_128.pkl'), help='Path to the training dataset')
#parser.add_argument('--test_data_path', type=str, default=os.path.join('datasets','burgers_50_10_128_128.pkl'), help='Path to the testing dataset')

# Value Network Settings
parser.add_argument('--v_steps', type=int, default=100000, help='Number of steps to train value network')
parser.add_argument('--v_hidden_dim', type=int, default=512, help='Number of hidden units in value network')
parser.add_argument('--v_depth', type=int, default=3, help='Number of layers in value network')
parser.add_argument('--v_lr', type=float, default=1e-4, help='Learning rate for value network')
parser.add_argument('--v_batch_size', type=int, default=64, help='Batch size for value network')

# Q Network Settings
parser.add_argument('--q_bc_steps', type=int, default=100000, help='Number of steps to train Q network with behavior cloning')
parser.add_argument('--q_pi_steps', type=int, default=10, help='Number of steps to train Q network with policy iteration')
parser.add_argument('--q_hidden_dim', type=int, default=512, help='Number of hidden units in Q network')
parser.add_argument('--q_depth', type=int, default=3, help='Number of layers in Q network')
parser.add_argument('--q_lr', type=float, default=1e-4, help='Learning rate for Q network')
parser.add_argument('--q_batch_size', type=int, default=64, help='Batch size for Q network')
parser.add_argument('--target_update_freq', type=int, default=2, help='Frequency of updating target Q network')
parser.add_argument('--tau', type=float, default=0.005, help='Soft update rate for target Q network parameters')
parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor for calculating the return')

# Behavior Cloning Settings
parser.add_argument('--bc_steps', type=int, default=100000, help='Number of steps to train behavior cloning')
parser.add_argument('--bc_lr', type=float, default=1e-4, help='Learning rate for behavior cloning')
parser.add_argument('--bc_hidden_dim', type=int, default=512, help='Number of hidden units in behavior cloning')
parser.add_argument('--bc_depth', type=int, default=3, help='Number of layers in behavior cloning')
parser.add_argument('--bc_batch_size', type=int, default=64, help='Batch size for behavior cloning')

# BPPO Settings
parser.add_argument('--bppo_steps', type=int, default=20000, help='Number of steps to train BPPO')
parser.add_argument('--bppo_hidden_dim', type=int, default=512, help='Number of hidden units in BPPO')
parser.add_argument('--bppo_depth', type=int, default=3, help='Number of layers in BPPO')
parser.add_argument('--bppo_lr', type=float, default=1e-4, help='Learning rate for BPPO')
parser.add_argument('--bppo_batch_size', type=int, default=64, help='Batch size for BPPO.')
parser.add_argument('--clip_ratio', type=float, default=0.25, help='PPO clip ratio. The probability ratio between new and old policy is clipped to be in the range [1-clip_ratio, 1+clip_ratio].')
parser.add_argument('--entropy_weight', type=float, default=0.00, help='Weight of entropy loss in PPO and BPPO. Can be set to 0.01 for medium tasks.')
parser.add_argument('--decay', type=float, default=0.96, help='Decay rate of PPO clip ratio')
parser.add_argument('--omega', type=float, default=0.9, help='Related to setting the weight of advantage (see BPPO code)')
parser.add_argument('--is_clip_decay', type=bool, default=True, help='Whether to decay the clip_ratio during training')
parser.add_argument('--is_bppo_lr_decay', type=bool, default=True, help='Whether to decay the learning rate of BPPO during trainining')
parser.add_argument('--is_update_old_policy', type=bool, default=True, help='Whether to update the old policy of BPPO in each iteration. The old policy is used to calculate the probability ratio.')

# Other Settings
parser.add_argument('--action_reward_scale', type=float, default=1000.0, help='Scaling factor of action reward')
parser.add_argument('--device', type=str, default='cuda', help='Device to run the code on')

# Parse the arguments
args = parser.parse_args()

# Other hyper settings (which can be calculated from other hyperparameters)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True

time_str=str(int(time.time())) # Time string to identify the current experiment

model_save_dir=os.path.join(args.result_dir, args.dataset_name, f"event_{time_str}") # Directory to save the models
log_dir=os.path.join(args.result_dir, args.dataset_name, f"event_{time_str}") # Directory to save tensorboard logs
fig_save_dir=os.path.join(args.result_dir, args.dataset_name, f"event_{time_str}", "figs") # Directory to save figures

dataset_name=args.dataset_name
device=args.device

# summarywriter logger
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
    print(f'Made log directory at {log_dir}')
logger = SummaryWriter(log_dir=log_dir, comment='')


# ----------------------- 2. Data Preprocessing ----------------------------------------------------------

from memory import ReplayMemory

print(f"Loading {dataset_name} data from {args.train_data_path}...")
train_data_dict=pickle.load(open(args.train_data_path, 'rb'))

# Dataset parameters
N=train_data_dict['data']['Y_bar'].shape[0] # Number of trajectories
nt=train_data_dict['data']['Y_bar'].shape[1] # Number of time steps per trajectory
state_dim=train_data_dict['data']['Y_bar'].shape[2] # Dimension of state space
action_dim=train_data_dict['data']['U'].shape[2] # Dimension of action space
if args.dataset_name=='burgers':
    delta_t=0.1 # Physical time step length
elif args.dataset_name in ['kuramoto', 'ip', 'power']:
    delta_t=0.01 # Physical time step length

print('Observations shape: ', train_data_dict['data']['Y_bar'].shape)
print('Actions shape: ', train_data_dict['data']['U'].shape)
print('Y_f shape: ', train_data_dict['data']['Y_f'].shape)
print('N: ', N)
print('state_dim: ', state_dim)
print('action_dim: ', action_dim)
print('nt: ', nt)
print('delta_t: ', delta_t)

state_trajectory=train_data_dict['data']['Y_bar'] # (N, nt, state_dim)
if args.dataset_name in ['burgers', 'kuramoto']:
    action_trajectory=train_data_dict['data']['U'] # (N, nt, action_dim)
elif args.dataset_name in ['ip', 'power']:
    action_trajectory=np.flip(train_data_dict['data']['U'], axis=1).copy() # (N, nt, action_dim) # use .copy() to avoid negative strides error
        # Note that in ip and power datasets, the control signals are stored in reverse chronological order, so we flip the time axis
final_state=train_data_dict['data']['Y_f'] # (N, state_dim)
next_state_trajectory=np.concatenate((state_trajectory[:,1:,:], 
                                    final_state.reshape(final_state.shape[0],1,-1))
                                   ,axis=1) # (N, nt, state_dim)

# Concatenate states, and compute rewards
concat_states=[]
concat_next_states=[]
rewards=[]
state_rewards=[]
action_rewards=[]
for n in tqdm.tqdm(range(N), desc="Looping over trajectories"):
    for t in range(nt): # Loop over time steps
        concat_state = np.concatenate((state_trajectory[n, t], final_state[n], np.array([t])))
        concat_next_state = np.concatenate((next_state_trajectory[n, t], final_state[n], np.array([t])))
        state_reward = -((state_trajectory[n, t]-final_state[n])**2).mean() # minus square distance between current state and final state
        action_reward = -args.action_reward_scale*np.sum(action_trajectory[n, t]**2) # minus sum of square action values
        reward = state_reward + action_reward
        concat_states.append(concat_state)
        concat_next_states.append(concat_next_state)
        rewards.append(reward)
        state_rewards.append(state_reward)
        action_rewards.append(action_reward)
concat_states=np.array(concat_states) # (N*nt, state_dim*2+1)
concat_next_states=np.array(concat_next_states) # (N*nt, state_dim*2+1)
rewards=np.array(rewards) # (N*nt,)
print('Concatenated states shape: ', concat_states.shape)
print('rewards shape: ', rewards.shape)

average_state_reward=np.array(state_rewards).mean()
average_action_reward=np.array(action_rewards).mean()
print("Average state_reward:", average_state_reward)
print("Average action_reward:", average_action_reward)
print("Average action_reward/state_reward:", average_action_reward/average_state_reward) # Check if the reward ratio is reasonable.

memory_data_dict={}
memory_data_dict['observations']=torch.tensor(concat_states, dtype=torch.float32)
memory_data_dict['actions']=torch.tensor(action_trajectory, dtype=torch.float32).reshape(N*nt, action_dim)
memory_data_dict['rewards']=torch.tensor(rewards, dtype=torch.float32)
memory_data_dict['next_observations']=torch.tensor(concat_next_states, dtype=torch.float32)

memory = ReplayMemory()
memory.load_dataset(data_dict=memory_data_dict)


# --------------------------------- 3. Model Training -----------------------------------------------------------------------

# Initilize models
from critic import ValueLearner, QSarsaLearner
from models import BC, BPPO, NaturalEvolution

value = ValueLearner(device=args.device,
                    state_dim=2*state_dim+1,
                    hidden_dim=args.v_hidden_dim,
                    depth=args.v_depth,
                    value_lr=args.v_lr,
                    batch_size=args.v_batch_size)
Q_bc = QSarsaLearner(device=args.device,
                    state_dim=2*state_dim+1,
                    action_dim=action_dim,
                    hidden_dim=args.q_hidden_dim,
                    depth=args.q_depth,
                    Q_lr=args.q_lr,
                    target_update_freq=args.target_update_freq,
                    tau=args.tau,
                    gamma=args.gamma,
                    batch_size=args.q_batch_size)
bc = BC(device=args.device,
                    state_dim=2*state_dim+1,
                    hidden_dim=args.bc_hidden_dim,
                    depth=args.bc_depth,
                    action_dim=action_dim,
                    policy_lr=args.bc_lr,
                    batch_size=args.bc_batch_size)
bppo = BPPO(device=args.device,
                    state_dim=2*state_dim+1,
                    hidden_dim=args.bppo_hidden_dim,
                    depth=args.bppo_depth,
                    action_dim=action_dim,
                    policy_lr=args.bppo_lr,
                    clip_ratio=args.clip_ratio,
                    entropy_weight=args.entropy_weight,
                    decay=args.decay,
                    omega=args.omega,
                    batch_size=args.bppo_batch_size)
ne = NaturalEvolution(action_dim=action_dim)

# Value training
value_save_path = os.path.join(model_save_dir, 'value_state_dict.pt')
if os.path.exists(value_save_path):
    value.load(value_save_path)
else:
    for step in tqdm.tqdm(range(int(args.v_steps)), desc='value updating ......'):
        value_loss = value.update(memory)
        if step % int(args.log_freq) == 0:
            print(f"Step: {step}, Loss: {value_loss:.6f}")
            logger.add_scalar('value_loss', value_loss, global_step=(step+1))
    value.save(value_save_path)


# Q_bc training
Q_save_path = os.path.join(model_save_dir, 'Q_bc_state_dict.pt')
if os.path.exists(Q_save_path):
    Q_bc.load(Q_save_path)
else:
    for step in tqdm.tqdm(range(int(args.q_bc_steps)), desc='Q_bc updating ......'):
        Q_bc_loss = Q_bc.update(memory, pi=None)
        if step % int(args.log_freq) == 0:
            print(f"Step: {step}, Loss: {Q_bc_loss:.6f}")
            logger.add_scalar('Q_bc_loss', Q_bc_loss, global_step=(step+1))
    Q_bc.save(Q_save_path)


# bc training
bc_save_path = os.path.join(model_save_dir, 'bc_state_dict.pt')
if os.path.exists(bc_save_path):
    bc.load(bc_save_path)
else:
    for step in tqdm.tqdm(range(int(args.bc_steps)), desc='bc updating ......'):
        bc_loss = bc.update(memory)
        if step % int(args.log_freq) == 0:
            print(f"Step: {step}, Loss: {bc_loss:.4f}")
            logger.add_scalar('bc_loss', bc_loss, global_step=(step+1))
    bc.save(os.path.join(model_save_dir, 'bc_state_dict.pt'))
    bc.load(bc_save_path)


# bppo training
bppo.load(bc_save_path)
Q = Q_bc
is_clip_decay=args.is_clip_decay
is_bppo_lr_decay=args.is_bppo_lr_decay
for step in tqdm.tqdm(range(int(args.bppo_steps)), desc='bppo updating ......'):
    if step > 200:
        is_clip_decay = False
        is_bppo_lr_decay = False
    bppo_loss = bppo.update(memory, Q=Q, value=value, is_clip_decay=is_clip_decay, is_lr_decay=is_bppo_lr_decay)
    if step % 100 == 0:
        print(f"Epoch: {step+1}, Loss: {bppo_loss:.4f}")
    logger.add_scalar('bppo_loss', bppo_loss, global_step=(step+1))
bppo.save(os.path.join(model_save_dir, f'bppo_state_dict_{time_str}.pt'))
logger.close()

# ------------------------------ 4. Testing and Result Saving--------------------------------------------

# Decide the system update function
if dataset_name=="burgers":
    from dynamics.burgers_dynamics import burgers_update
    update_func=burgers_update
elif dataset_name=="power":
    from dynamics.swing_dynamics import swing_update
    update_func=swing_update
elif dataset_name=="ip":
    from dynamics.ip_dynamics import ip_update
    update_func=ip_update
elif dataset_name=="kuramoto":
    from dynamics.kuramoto_dynamics import kuramoto_update
    update_func=kuramoto_update


def evolve(initial_state, target_state, update_func, model):
    '''
    For a single trajectory.
    Return the final state vector following a dynamics specified by update_func, given an initial state vector and an action.

    update_func:
    - input: state, action, delta_t, t
    - output: (state_dim,), updated state vector
    '''
    global state_dim, action_dim, nt, delta_t, device
    assert initial_state.shape == target_state.shape == (state_dim,), "initial_state and target_state should have shape (state_dim,)"
    s=initial_state # (state_dim,), the state vector (original meaning)
    S=np.concatenate((initial_state, target_state, np.array([0]))) # (state_dim*2+1,), the concatenated state vector (with target state and time step)
    actions=[] # A list to store the actions

    for i in range(nt): # Loop over time steps to update the state vector
        a = model.select_action(torch.FloatTensor(S).unsqueeze(0).to(device), is_sample=False
                                ).cpu().data.numpy().flatten() # Decide the action using the model
        actions.append(a)
        s=update_func(state=s, action=a, delta_t=delta_t, t=i*delta_t) # get the updated state vector (of original meaning)
        S=np.concatenate((s,target_state,np.array([i]))) # Compute and update the concatenated state vector

    energy=np.sum([(a**2).sum()*delta_t for a in actions]) # J(u)=int_0^T u^2 du, the total energy of the control input
    return s, energy # final state, (state_dim,); energy, scalar

def evaluate_target_loss_and_energy(initial_states, target_states, update_func, model):
    '''
    For multiple trajectories.
    Evaluate the target loss and energy of a model on a set of initial and target states.
    initial_states: (n_trajectories, state_dim)
    target_states: (n_trajectories, state_dim)
    '''
    final_losses=[]
    energys=[]
    final_states=[]
    n_trajectories=initial_states.shape[0]
    for i in tqdm.tqdm(range(n_trajectories), desc='Looping over trajectories'):
        initial_state = initial_states[i] # (state_dim,), initial state of the i-th trajectory
        target_state = target_states[i] # (state_dim,), target state of the i-th trajectory
        final_state, energy=evolve(initial_state, target_state, update_func, model) # With control input
        target_loss = ((final_state-target_state)**2).mean() # MSE loss between target state and final state
        final_losses.append(target_loss)
        energys.append(energy)
        final_states.append(final_state)
    return (np.array(final_losses), np.array(energys), np.array(final_states))


# Load test data
test_data_dict=pickle.load(open(args.test_data_path, 'rb'))

test_initial_states=test_data_dict['data']['Y_bar'][:,0,:] # (n_test_trajectories, state_dim)
test_target_states=test_data_dict['data']['Y_f'] # (n_test_trajectories, state_dim)

bppo_target_losses, bppo_energys, bppo_final_states = \
        evaluate_target_loss_and_energy(test_initial_states, test_target_states, update_func, model=bppo)
bppo_target_loss, bppo_energy = bppo_target_losses.mean(), bppo_energys.mean()
print(f"BPPO target loss: {bppo_target_loss:.8f}, Energy: {bppo_energy:.8f}")

bc_target_losses, bc_energys, bc_final_states = \
        evaluate_target_loss_and_energy(test_initial_states, test_target_states, update_func, model=bc)
bc_target_loss, bc_energy = bc_target_losses.mean(), bc_energys.mean()
print(f"BC target loss: {bc_target_loss:.8f}, Energy: {bc_energy:.8f}")

ne_target_losses, ne_energys, ne_final_states = \
        evaluate_target_loss_and_energy(test_initial_states, test_target_states, update_func, model=ne)
ne_target_loss, ne_energy = ne_target_losses.mean(), ne_energys.mean()
print(f"Natural target loss: {ne_target_loss:.8f}, Energy: {ne_energy:.8f}")


# Visualize the final states

def plot_state(model_name='BPPO', traj_index=0, show_fig=False, save_fig=True, fig_save_dir=None):
    '''
    Plot the final states
    '''
    global test_target_states
    global bppo_target_losses, bppo_energys, bppo_final_states
    global bc_target_losses, bc_energys, bc_final_states
    global ne_target_losses, ne_energys, ne_final_states
    assert traj_index<test_target_states.shape[0], 'trajectory index out of range'
    assert model_name in ['BPPO', 'BC'], 'Invalid model name'
    
    target_state=test_target_states[traj_index]
    natural_final_state=ne_final_states[traj_index]
    natural_target_loss=ne_target_losses[traj_index]

    if model_name=='BPPO':
        final_state=bppo_final_states[traj_index]
        target_loss=bppo_target_losses[traj_index]
        energy=bppo_energys[traj_index]
    
    elif model_name=='BC':
        final_state=bc_final_states[traj_index]
        target_loss=bc_target_losses[traj_index]
        energy=bc_energys[traj_index]

    title_text='Trajectory {}; {}\nTarget Loss: {:.6f}; Energy: {:.6f}\nNatural Evolution Loss: {:.6f}; Energy: {:.6f}'.format(
                    traj_index, model_name,
                    target_loss, energy,
                    natural_target_loss, 0.0)

    plt.figure(figsize=(9,6))
    #plt.plot(initial_state, label='initial_state')
    plt.plot(target_state, label='target_state', lw=3, linestyle='--')
    plt.plot(final_state, label='final_state (with control input)', lw=3)
    plt.plot(natural_final_state, label='final_state (natural evolution)', alpha=0.5, lw=1)
    plt.title(title_text)
    plt.legend()
    if save_fig == True and fig_save_dir is not None:
        save_path=os.path.join(fig_save_dir, f'{model_name}_trajectory_{traj_index}.png')
        plt.savefig(save_path)
        print(f'Figure saved to {save_path}')
    if show_fig is True:
        plt.show()
    elif show_fig is False:
        plt.close()

if fig_save_dir is not None:
    if not os.path.exists(fig_save_dir):
        os.makedirs(fig_save_dir)
        print(f'Directory {fig_save_dir} created')

# Randomly select some trajectories to visualize
show_fig=False
plot_state(model_name='BPPO', traj_index=5, show_fig=show_fig, save_fig=args.save_fig, fig_save_dir=fig_save_dir)
plot_state(model_name='BPPO', traj_index=17, show_fig=show_fig, save_fig=args.save_fig, fig_save_dir=fig_save_dir)
plot_state(model_name='BPPO', traj_index=22, show_fig=show_fig, save_fig=args.save_fig, fig_save_dir=fig_save_dir)
plot_state(model_name='BPPO', traj_index=31, show_fig=show_fig, save_fig=args.save_fig, fig_save_dir=fig_save_dir)
plot_state(model_name='BPPO', traj_index=49, show_fig=show_fig, save_fig=args.save_fig, fig_save_dir=fig_save_dir)
plot_state(model_name='BC', traj_index=5, show_fig=show_fig, save_fig=args.save_fig, fig_save_dir=fig_save_dir)
plot_state(model_name='BC', traj_index=17, show_fig=show_fig, save_fig=args.save_fig, fig_save_dir=fig_save_dir)
plot_state(model_name='BC', traj_index=22, show_fig=show_fig, save_fig=args.save_fig, fig_save_dir=fig_save_dir)
plot_state(model_name='BC', traj_index=31, show_fig=show_fig, save_fig=args.save_fig, fig_save_dir=fig_save_dir)
plot_state(model_name='BC', traj_index=49, show_fig=show_fig, save_fig=args.save_fig, fig_save_dir=fig_save_dir)



# Save the experiment information

experiment_info = {
    'Results': {
        'BPPO': {
            'bppo_target_loss': bppo_target_loss,
            'bppo_energy;': bppo_energy,
        },
        'BC': {
            'bc_target_loss': bc_target_loss,
            'bc_energy': bc_energy,
        },
        'Natural Evolution': {
            'natural_target_loss': ne_target_loss,
            'natural_energy': ne_energy,
        }
    },

    'Experiment Settings': {
        'dataset_name': args.dataset_name,
        'seed': args.seed,
        'train_data_path': args.train_data_path,
        'test_data_path': args.test_data_path,
        'result_dir': args.result_dir,
        'model_save_dir': model_save_dir,
        'log_dir': log_dir,
        'fig_save_dir': fig_save_dir,
        'log_freq': args.log_freq
    },

    'Value Network Settings': {
        'v_steps': args.v_steps,
        'v_hidden_dim': args.v_hidden_dim,
        'v_depth': args.v_depth,
        'v_lr': args.v_lr,
        'v_batch_size': args.v_batch_size
    },

    'Q Network Settings': {
        'q_bc_steps': args.q_bc_steps,
        'q_pi_steps': args.q_pi_steps,
        'q_hidden_dim': args.q_hidden_dim,
        'q_depth': args.q_depth,
        'q_lr': args.q_lr,
        'q_batch_size': args.q_batch_size,
        'target_update_freq': args.target_update_freq,
        'tau': args.tau,
        'gamma': args.gamma,
    },

    'BC Settings': {
        'bc_steps': args.bc_steps,
        'bc_lr': args.bc_lr,
        'bc_hidden_dim': args.bc_hidden_dim,
        'bc_depth': args.bc_depth,
        'bc_batch_size': args.bc_batch_size
    },

    'BPPO Settings': {
        'bppo_steps': args.bppo_steps,
        'bppo_hidden_dim': args.bppo_hidden_dim,
        'bppo_depth': args.bppo_depth,
        'bppo_lr': args.bppo_lr,
        'bppo_batch_size': args.bppo_batch_size,
        'clip_ratio': args.clip_ratio,
        'entropy_weight': args.entropy_weight,
        'decay': args.decay,
        'omega': args.omega,
        'is_clip_decay': args.is_clip_decay,
        'is_bppo_lr_decay': args.is_bppo_lr_decay,
        'is_update_old_policy': args.is_update_old_policy
    },

    'Others': {
        'N': N,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'nt': nt,
        'action_reward_scale': args.action_reward_scale,
        'average state reward': average_state_reward,
        'average action reward': average_action_reward,
    },
}

import json
json.dump(experiment_info, open(os.path.join(log_dir, 'experiment_info.json'), 'w'), indent=4)
print('Experiment Information saved to', os.path.join(log_dir, 'experiment_info.json'))

print('Successfully finished!\n')
print("\n====================================END====================================\n")
