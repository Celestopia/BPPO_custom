

Data generation method:
```python
# diffphycon/baselines/BPPO_burgers/train_infer_FOPC.py line 120-143
dataset = h5py.File("../../1D_data/free_u_f_1e5_front_rear_quarter/burgers_train.h5", 'r')['train']
u_data, f_data = (torch.tensor(np.array(dataset['pde_11-128']), device=device).float())/RESCALER, (torch.tensor(np.array(dataset['pde_11-128_f']),device=device).float())/RESCALER
u_data_full = u_data

Ndata = u_data.shape[0]
u_target = u_data[:, -1]

# u_data shape: (Ndata, T, s) i.e. (Ndata, 11, 128)
# f_data shape: (Ndata, T, s) i.e. (Ndata, 10, 128)
# u_target shape: (Ndata, s) i.e. (Ndata, 128)
# u is the state, f is the action

for i in range(num_t):
    state = torch.cat((i*torch.ones_like(u_data[:, 0, [0]])/RESCALER, u_data[:, i], u_target), -1) # Note this is the state used
    action = f_data[:, i]  
    
    if args.is_relative == 1:
        uT_reward = (-1) * rel_error(u_data[:, i+1], u_target)
    elif args.is_relative == 0:
        uT_reward = (-1) * (u_target-u_data[:, i+1]).square().mean(-1)
        
    f_reward = torch.sum(f_data[:, i]**2, -1) 
    reward = uT_reward + (-reward_f)*f_reward
    
    next_state = torch.cat(((i+1)*torch.ones_like(u_data[:, 0, [0]])/RESCALER, u_data[:, i+1], u_target), -1)
    
    memory.push_batch(list(zip(state.cpu().numpy(), action.cpu().numpy(), reward.cpu().numpy(),\
                    next_state.cpu().numpy(), np.repeat(mask_batch[i], Ndata))))
```


```python
# diffphycon/baselines/BPPO_burgers/train_infer_FOPC.py line 148-149
state_dim = int((s)*2+1)
action_dim = s
```

So the "state" is a concatenated vector of real state, target state, and a scalar.


以及上次讨论的时候发现它的数据集全都是二维的，例如state_trajectory的形状是(N, state_dim)，所以对训练的时候到底用了多少个trajectory有疑惑，现在大概看明白了，它把不同trajectory拼接在一起了，所以整个state数据集的形状是(N_trajectory*num_t_per_trajectory, state_dim*2+1)。这个数据集的第一个维度的不同指标代表不同样本（可以来自于不同轨迹），第二个维度代表特征，其中前128个数代表当前state（在grid上的函数值），接下来128个代表当前轨迹的目标值（在属于同一轨迹的样本间共享，所以state[0,128:256]=state[1,128:256]=...=state[9,128:256]；state[10,128:256]=state[11,128:256]=...=state[19,128:256]；...最后一个标量是代表时间点的，例如state[0,257]=0, state[5,257]=5, state[3426,257]=6...