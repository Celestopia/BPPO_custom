import torch
import numpy as np
import os
import time
from tqdm import tqdm
import argparse
from tensorboardX import SummaryWriter

from buffer import OfflineReplayBuffer
from critic import ValueLearner, QPiLearner, QSarsaLearner
from bppo import BehaviorCloning, BehaviorProximalPolicyOptimization

# ---------------------------------------------------------------------------------

# Hyperparameters

# Experiment
env_name='burger'
path='logs'
log_freq=int(2e3)
seed=20241218
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
N=100 # Number of trajectories to collect for offline dataset

# For Value
v_steps=int(2e6)/1000
v_hidden_dim = 512
v_depth = 3
v_lr = 1e-4
v_batch_size = 512

# For Q
q_bc_steps=int(2e6)/1000
q_pi_steps=10
q_hidden_dim = 1024
q_depth = 2
q_lr = 1e-4
q_batch_size = 512
target_update_freq=2
tau=0.005
gamma=0.99
is_offpolicy_update=False

# For BC
bc_steps=int(5e5)/1000
bc_lr = 1e-4
bc_hidden_dim = 1024
bc_depth = 2
bc_batch_size = 512

# For BPPO
bppo_steps=int(1e3)/1000
bppo_hidden_dim = 1024
bppo_depth = 2
bppo_lr = 1e-4
bppo_batch_size = 512
clip_ratio=0.25
entropy_weight=0.00
decay=0.96
omega=0.9
is_clip_decay=True
is_bppo_lr_decay=True
is_update_old_policy=True
is_state_norm=False


# Other Settings
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device=torch.device('cpu')
state_dim = 8
action_dim = 8


# ---------------------------------------------------------------------------------


# path
current_time = time.strftime("%Y_%m_%d__%H_%M_%S", time.localtime())
path = os.path.join(path, str(seed))
os.makedirs(os.path.join(path, current_time))
print(f'Made log directory at {os.path.join(path, current_time)}')



# offline dataset to replay buffer
dataset = { # TODO
    "observations": torch.rand(N, state_dim),
    "actions": torch.rand(N, action_dim),
    "rewards": torch.rand(N, 1),
    "terminals": np.random.randint(2, size=(N, 1)),
    "timeouts": np.random.randint(2, size=(N, 1)),
}


replay_buffer = OfflineReplayBuffer(device, state_dim, action_dim, len(dataset['actions']))
replay_buffer.load_dataset(dataset=dataset)
replay_buffer.compute_return(gamma)


# for hopper-medium-v2 task, don't use state normalize
if is_state_norm:
    mean, std = replay_buffer.normalize_state()
else:
    mean, std = np.zeros(state_dim), np.ones(state_dim)

# summarywriter logger
logger_path = os.path.join(path, current_time)
logger = SummaryWriter(log_dir=logger_path, comment='')


# --------------------------------------------------------------------------


# initilize
value = ValueLearner(device=device,
                        state_dim=state_dim,
                        hidden_dim=v_hidden_dim,
                        depth=v_depth,
                        value_lr=v_lr,
                        batch_size=v_batch_size)

Q_bc = QSarsaLearner(device=device,
                        state_dim=state_dim,
                        action_dim=action_dim,
                        hidden_dim=q_hidden_dim, depth=q_depth,
                        Q_lr=q_lr,
                        target_update_freq=target_update_freq,
                        tau=tau,
                        gamma=gamma,
                        batch_size=q_batch_size)

if is_offpolicy_update: 
    Q_pi=QPiLearner(device=device,
                        state_dim=state_dim,
                        action_dim=action_dim,
                        hidden_dim=q_hidden_dim,
                        depth=q_depth,
                        Q_lr=q_lr,
                        target_update_freq=target_update_freq,
                        tau=tau,
                        gamma=gamma,
                        batch_size=q_batch_size)
bc=BehaviorCloning(device=device,
                        state_dim=state_dim,
                        hidden_dim=bc_hidden_dim,
                        depth=bc_depth,
                        action_dim=action_dim,
                        policy_lr=bc_lr,
                        batch_size=bc_batch_size)
bppo=BehaviorProximalPolicyOptimization(device=device,
                        state_dim=state_dim,
                        hidden_dim=bppo_hidden_dim,
                        depth=bppo_depth,
                        action_dim=action_dim,
                        policy_lr=bppo_lr,
                        clip_ratio=clip_ratio,
                        entropy_weight=entropy_weight,
                        decay=decay,
                        omega=omega,
                        batch_size=bppo_batch_size)


# ---------------------------------------------------------------------------------

# value training 
value_path = os.path.join(path, 'value.pt')
if os.path.exists(value_path):
    value.load(value_path)
else:
    for step in tqdm(range(int(v_steps)), desc='value updating ......'): 
        value_loss = value.update(replay_buffer)
        
        if step % int(log_freq) == 0:
            print(f"Step: {step}, Loss: {value_loss:.4f}")
            logger.add_scalar('value_loss', value_loss, global_step=(step+1))
    value.save(value_path)

# Q_bc training
Q_bc_path = os.path.join(path, 'Q_bc.pt')
if os.path.exists(Q_bc_path):
    Q_bc.load(Q_bc_path)
else:
    for step in tqdm(range(int(q_bc_steps)), desc='Q_bc updating ......'): 
        Q_bc_loss = Q_bc.update(replay_buffer, pi=None)
        if step % int(log_freq) == 0:
            print(f"Step: {step}, Loss: {Q_bc_loss:.4f}")
            logger.add_scalar('Q_bc_loss', Q_bc_loss, global_step=(step+1))
    Q_bc.save(Q_bc_path)

if is_offpolicy_update:
    Q_pi.load(Q_bc_path)

# bc training
best_bc_path = os.path.join(path, 'bc_best.pt')
if os.path.exists(best_bc_path):
    bc.load(best_bc_path)
else:
    best_bc_score = 0    
    for step in tqdm(range(int(bc_steps)), desc='bc updating ......'):
        bc_loss = bc.update(replay_buffer)
        if step % int(log_freq) == 0:
            current_bc_score = bc.offline_evaluate(env_name, seed, mean, std)
            if current_bc_score > best_bc_score:
                best_bc_score = current_bc_score
                bc.save(best_bc_path)
                np.savetxt(os.path.join(path, 'best_bc.csv'), [best_bc_score], fmt='%f', delimiter=',')
            print(f"Step: {step}, Loss: {bc_loss:.4f}, Score: {current_bc_score:.4f}")
            logger.add_scalar('bc_loss', bc_loss, global_step=(step+1))
            logger.add_scalar('bc_score', current_bc_score, global_step=(step+1))
    bc.save(os.path.join(path, 'bc_last.pt'))
    bc.load(best_bc_path)

# bppo training
bppo.load(best_bc_path)
best_bppo_path = os.path.join(path, current_time, 'bppo_best.pt')
Q = Q_bc
best_bppo_score = bppo.offline_evaluate(env_name, seed, mean, std)
print('best_bppo_score:',best_bppo_score,'-------------------------')
for step in tqdm(range(int(bppo_steps)), desc='bppo updating ......'):
    if step > 200:
        is_clip_decay = False
        is_bppo_lr_decay = False
    bppo_loss = bppo.update(replay_buffer, Q, value, is_clip_decay, is_bppo_lr_decay)
    current_bppo_score = bppo.offline_evaluate(env_name, seed, mean, std)
    if current_bppo_score > best_bppo_score:
        best_bppo_score = current_bppo_score
        print('best_bppo_score:',best_bppo_score,'-------------------------')
        bppo.save(best_bppo_path)
        np.savetxt(os.path.join(path, current_time, 'best_bppo.csv'), [best_bppo_score], fmt='%f', delimiter=',')
        if is_update_old_policy:
            bppo.set_old_policy()
    if is_offpolicy_update:
        for _ in tqdm(range(int(q_pi_steps)), desc='Q_pi updating ......'): 
            Q_pi_loss = Q_pi.update(replay_buffer, bppo)
        Q = Q_pi
    print(f"Step: {step}, Loss: {bppo_loss:.4f}, Score: {current_bppo_score:.4f}")
    logger.add_scalar('bppo_loss', bppo_loss, global_step=(step+1))
    logger.add_scalar('bppo_score', current_bppo_score, global_step=(step+1))

logger.close()