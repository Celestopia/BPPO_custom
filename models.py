"""
BPPO, BC, and Natural Evolution.
"""
import torch
import numpy as np
from memory import ReplayMemory
from net import GaussPolicyMLP
from critic import ValueLearner, QSarsaLearner
from utils import CONST_EPS, log_prob_func, orthogonal_initWeights
import copy


class NaturalEvolution:
    '''
    Natural Evolution Strategies.
    Used as a baseline for comparison.
    '''
    def __init__(self, action_dim:int):
        self.action_dim = action_dim

    def select_action(
        self, s: torch.Tensor, is_sample: bool
    ) -> torch.Tensor:
        return torch.zeros((s.shape[0], self.action_dim)) # Take no action, i.e. natural evolution.


class BC:
    '''
    Behavior Cloning
    '''
    _device: torch.device
    _policy: GaussPolicyMLP
    _optimizer: torch.optim
    _policy_lr: float
    _batch_size: int
    def __init__(
        self,
        device: torch.device,
        state_dim: int,
        hidden_dim: int, 
        depth: int,
        action_dim: int,
        policy_lr: float,
        batch_size: int
    ) -> None:
        super().__init__()
        self._device = device
        self._policy = GaussPolicyMLP(state_dim, hidden_dim, depth, action_dim).to(device)
        orthogonal_initWeights(self._policy)
        self._optimizer = torch.optim.Adam(
            self._policy.parameters(),
            lr = policy_lr
        )
        self._lr = policy_lr
        self._batch_size = batch_size


    def loss(
        self, memory: ReplayMemory,
    ) -> torch.Tensor:
        state_batch, action_batch, reward_batch, next_state_batch = memory.sample(self._batch_size)
        state_batch = torch.FloatTensor(state_batch).to(self._device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self._device)
        action_batch = torch.FloatTensor(action_batch).to(self._device)
        reward_batch = torch.FloatTensor(reward_batch).to(self._device).unsqueeze(1)
        dist = self._policy(state_batch)
        
        log_prob = log_prob_func(dist, action_batch) 
        loss = (-log_prob).mean()
        
        return loss


    def update(
        self, memory: ReplayMemory,
        ) -> float:
        policy_loss = self.loss(memory)

        self._optimizer.zero_grad()
        policy_loss.backward()
        self._optimizer.step()

        return policy_loss.item()


    def select_action(
        self, s: torch.Tensor, is_sample: bool
    ) -> torch.Tensor:
        dist = self._policy(s)
        if is_sample:
            action = dist.sample()
        else:
            action = dist.mean
        return action

    def save(
        self, path: str
    ) -> None:
        torch.save(self._policy.state_dict(), path)
        print('Behavior policy parameters saved to {}'.format(path))

    def load(
        self, path: str
    ) -> None:
        self._policy.load_state_dict(torch.load(path, map_location=self._device))
        print('Behavior policy parameters loaded from {}'.format(path))



class BPPO:
    '''
    Behavior Proximal Policy Optimization
    '''
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
        self._device = device
        self._policy = GaussPolicyMLP(state_dim, hidden_dim, depth, action_dim).to(device)
        orthogonal_initWeights(self._policy)
        self._optimizer = torch.optim.Adam(self._policy.parameters(), lr=policy_lr)
        self._policy_lr = policy_lr
        self._old_policy = copy.deepcopy(self._policy)
        self._scheduler = torch.optim.lr_scheduler.StepLR(self._optimizer, step_size=2, gamma=0.98)
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
        memory: ReplayMemory,
        Q: QSarsaLearner,
        value: ValueLearner,
        is_clip_decay: bool,
    ) -> torch.Tensor:
        state_batch, action_batch, reward_batch, next_state_batch = memory.sample(self._batch_size)
        state_batch = torch.FloatTensor(state_batch).to(self._device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self._device)
        action_batch = torch.FloatTensor(action_batch).to(self._device)
        reward_batch = torch.FloatTensor(reward_batch).to(self._device).unsqueeze(1)
        
        old_dist = self._old_policy(state_batch)
        a = old_dist.rsample()
        advantage = Q(state_batch, a) - value(state_batch)
        advantage = (advantage - advantage.mean()) / (advantage.std() + CONST_EPS)
        new_dist = self._policy(state_batch)

        new_log_prob = log_prob_func(new_dist, a)
        old_log_prob = log_prob_func(old_dist, a)
        ratio = (new_log_prob - old_log_prob).exp()
        
        advantage = self.weighted_advantage(advantage)
        
        loss1 =  ratio * advantage 
        
        if is_clip_decay:
            self._clip_ratio = self._clip_ratio * self._decay
        else:
            self._clip_ratio = self._clip_ratio

        loss2 = torch.clamp(ratio, 1 - self._clip_ratio, 1 + self._clip_ratio) * advantage 
        
        entropy_loss = new_dist.entropy().sum(-1, keepdim=True) * self._entropy_weight
        
        loss = -(torch.min(loss1, loss2) + entropy_loss).mean()

        return loss


    def update(
        self,
        memory: ReplayMemory,
        Q: QSarsaLearner,
        value: ValueLearner,
        is_clip_decay: bool,
        is_lr_decay: bool
    ) -> float:
        policy_loss = self.loss(memory, Q, value, is_clip_decay)
        
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
        return action

    def save(
        self, path: str
    ) -> None:
        torch.save(self._policy.state_dict(), path)
        print('BPPO Policy parameters saved to {}'.format(path))
    

    def load(
        self, path: str
    ) -> None:
        self._policy.load_state_dict(torch.load(path, map_location=self._device))
        self._old_policy.load_state_dict(self._policy.state_dict())
        #self._optimizer = torch.optim.Adam(self._policy.parameters(), lr=self._policy_lr)
        print('BPPO Policy parameters loaded from {}'.format(path))

    def set_old_policy(
        self,
    ) -> None:
        '''
        Set the old policy to the current policy
        '''
        self._old_policy.load_state_dict(self._policy.state_dict())
