import numpy as np

class ReplayMemory:
    def __init__(self, max_size=1000000):
        self.max_size = max_size
        self.states = None
        self.actions = None
        self.rewards = None
        self.size=0

    def load_dataset(self, data_dict, verbose=True):
        self.states = data_dict['observations']
        self.actions = data_dict['actions']
        self.rewards = data_dict['rewards']
        self.next_states = data_dict['next_observations']
        self.size = self.states.shape[0]

        if verbose:
            print("\nReplay Memory loaded with size:", self.size)
            print("states shape:", self.states.shape)
            print("actions shape:", self.actions.shape)
            print("rewards shape:", self.rewards.shape)
            print("next_states shape:", self.next_states.shape)

    def sample(self, batch_size=32):
        '''
        Samples a batch of experiences from memory.
        '''
        ind = np.random.randint(0, self.size, size=batch_size)
        state_batch = self.states[ind] # (batch_size, 2*state_dim+1)
        action_batch = self.actions[ind] # (batch_size, action_dim)
        reward_batch = self.rewards[ind] # (batch_size, 1)
        next_state_batch = self.next_states[ind] # (batch_size, 2*state_dim+1)
        return state_batch, action_batch, reward_batch, next_state_batch
