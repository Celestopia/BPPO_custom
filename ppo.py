#import gym
import torch
import numpy as np
from copy import deepcopy

from buffer import OnlineReplayBuffer
from net import GaussPolicyMLP
from critic import ValueLearner, QLearner
from utils import orthogonal_initWeights, log_prob_func



class ProximalPolicyOptimization:
    _device: torch.device
    _policy: GaussPolicyMLP
    _optimizer: torch.optim
    _policy_lr: float
    _old_policy: GaussPolicyMLP
    _scheduler: torch.optim
    _clip_ratio: float
    _entropy_weight: float
    _decay: float
    _omega: float
    _batch_size: int


    def __init__(
        self,
        device: torch.device,
        state_dim: int,
        hidden_dim: int, 
        depth: int,
        action_dim: int,
        policy_lr: float,
        clip_ratio: float,
        entropy_weight: float,
        decay: float,
        omega: float,
        batch_size: int
    ) -> None:
        super().__init__()
        self._device = device
        self._policy = GaussPolicyMLP(state_dim, hidden_dim, depth, action_dim).to(device)
        orthogonal_initWeights(self._policy)
        self._optimizer = torch.optim.Adam(
            self._policy.parameters(),
            lr=policy_lr
            )
        self._policy_lr = policy_lr
        self._old_policy = deepcopy(self._policy)
        self._scheduler = torch.optim.lr_scheduler.StepLR(
            self._optimizer,
            step_size=2,
            gamma=0.98
            )
        
        self._clip_ratio = clip_ratio
        self._entropy_weight = entropy_weight
        self._decay = decay
        self._omega = omega
        self._batch_size = batch_size


    def weighted_advantage(
        self,
        advantage: torch.Tensor
    ) -> torch.Tensor:
        '''
        Return the weighted advantage.

        For advantage>0: weight = self._omega
        For advantage<0: weight = 1-self._omega
        '''
        if self._omega == 0.5:
            return advantage
        else:
            weight = torch.zeros_like(advantage)
            index = torch.where(advantage > 0)[0]
            weight[index] = self._omega
            weight[torch.where(weight == 0)[0]] = 1 - self._omega
            weight.to(self._device)
            return weight * advantage


    def loss(
        self, 
        replay_buffer: OnlineReplayBuffer,
        Q: QLearner,
        value: ValueLearner,
        is_clip_decay: bool,
    ) -> torch.Tensor:
        '''
        Calculate the loss on a batch of data from the replay buffer.
        '''
        # -------------------------------------Advantage-------------------------------------
        s, a, _, _, _, _, _, advantage = replay_buffer.sample(self._batch_size)
        old_dist = self._old_policy(s)
        # -------------------------------------Advantage-------------------------------------
        new_dist = self._policy(s)
        
        new_log_prob = log_prob_func(new_dist, a)
        old_log_prob = log_prob_func(old_dist, a)
        ratio = (new_log_prob - old_log_prob).exp() # The probability ratio between new and old policy, $\frac{\pi(a|s)}{\pi_k(a|s)}$
        
        advantage = self.weighted_advantage(advantage)

        loss1 =  ratio * advantage # $\frac{\pi(a|s)}{\pi_k(a|s)}A_{\pi_k}(s,a)$, as proposed in the original paper.

        if is_clip_decay:
            self._clip_ratio = self._clip_ratio * self._decay
        else:
            self._clip_ratio = self._clip_ratio

        loss2 = torch.clamp(ratio, 1 - self._clip_ratio, 1 + self._clip_ratio) * advantage # $CLIP\left(\frac{\pi(a|s)}{\pi_k(a|s)}, 1-\epsilon, 1+\epsilon\right)A_{\pi_k}(s,a)$, as proposed in the original paper.

        entropy_loss = new_dist.entropy().sum(-1, keepdim=True) * self._entropy_weight
        
        loss = -(torch.min(loss1, loss2) + entropy_loss).mean()
        '''
        In the original paper, we want to maximize the objective function.
        Since pytorch optimizer minimize the objective function, we take the negative of the objective function.
        '''
        return loss


    def update(
        self,
        replay_buffer: OnlineReplayBuffer,
        Q: QLearner,
        value: ValueLearner,
        is_clip_decay: bool,
        is_lr_decay: bool
    ) -> float:
        policy_loss = self.loss(replay_buffer, Q, value, is_clip_decay)
        
        self._optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._policy.parameters(), 0.5)
        self._optimizer.step()
        
        if is_lr_decay:
            self._scheduler.step()
        return policy_loss.item()


    def select_action(
        self, s: torch.Tensor, is_sample: bool
    ) -> torch.Tensor:
        dist = self._policy(s)
        if is_sample:
            action = dist.sample()
        else:    
            action = dist.mean
        # clip 
        #action = action.clamp(-1., 1.)
        return action
    

    def evaluate(
        self,
        env: object,
        seed: int,
        eval_episodes: int=10
        ) -> float:
        '''
        Evaluate the model on several episodes of the environment. The number of episodes is specified by `eval_episodes`.
        '''
        #env.seed(seed)
        
        total_reward = 0
        for _ in range(eval_episodes):
            s, done = env.reset(), False
            while not done:
                #s = torch.FloatTensor((np.array(s).reshape(1, -1) - mean) / std).to(self._device)
                a = self.select_action(torch.FloatTensor(s).unsqueeze(0).to(self._device), is_sample=False).cpu().data.numpy().flatten()
                s, r, done = env.step(a)
                total_reward += r
        
        avg_reward = total_reward / eval_episodes
        return avg_reward


    def save(
        self, path: str
    ) -> None:
        torch.save(self._policy.state_dict(), path)
        print('Policy parameters saved in {}'.format(path))
    

    def load(
        self, path: str
    ) -> None:
        self._policy.load_state_dict(torch.load(path, map_location=self._device))
        self._old_policy.load_state_dict(self._policy.state_dict())
        #self._optimizer = torch.optim.Adam(self._policy.parameters(), lr=self._policy_lr)
        print('Policy parameters loaded')

    def set_old_policy(
        self,
    ) -> None:
        '''
        Set the old policy to the current policy
        '''
        self._old_policy.load_state_dict(self._policy.state_dict())
