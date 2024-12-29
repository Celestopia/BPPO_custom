

Data generation method:
```python
# diffphycon/baselines/BPPO_burgers/train_infer_FOPC.py line 120-143
dataset = h5py.File("../../1D_data/free_u_f_1e5_front_rear_quarter/burgers_train.h5", 'r')['train']
u_data, f_data = (torch.tensor(np.array(dataset['pde_11-128']), device=device).float())/RESCALER, (torch.tensor(np.array(dataset['pde_11-128_f']),device=device).float())/RESCALER
u_data_full = u_data

Ndata = u_data.shape[0]
u_target = u_data[:, -1]

# u_data shape: (Ndata, T, s) i.e. (Ndata, 11, 128)
# f_data shape: (Ndata, T, s) i.e. (Ndata, 11, 128)
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
