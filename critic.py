import torch
import torch.nn.functional as F
from net import ValueMLP, QMLP
from memory import ReplayMemory

class ValueLearner:
    _device: torch.device
    _value: ValueMLP
    _optimizer: torch.optim
    _batch_size: int

    def __init__(
        self, 
        device: torch.device, 
        state_dim: int, 
        hidden_dim: int, 
        depth: int, 
        value_lr: float, 
        batch_size: int
    ) -> None:
        super().__init__()
        self._device = device
        self._value = ValueMLP(state_dim, hidden_dim, depth).to(device)
        self._optimizer = torch.optim.Adam(
            self._value.parameters(), 
            lr=value_lr,
            )
        self._batch_size = batch_size


    def __call__(
        self, s: torch.Tensor
    ) -> torch.Tensor:
        return self._value(s)


    def update(
        self, memory: ReplayMemory
    ) -> float:
        state_batch, action_batch, reward_batch, next_state_batch = memory.sample(self._batch_size)
        state_batch = torch.FloatTensor(state_batch).to(self._device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self._device)
        action_batch = torch.FloatTensor(action_batch).to(self._device)
        reward_batch = torch.FloatTensor(reward_batch).to(self._device).unsqueeze(1)
        
        value_loss = F.mse_loss(self._value(state_batch), reward_batch)
        
        self._optimizer.zero_grad()
        value_loss.backward()
        self._optimizer.step()

        return value_loss.item()


    def save(
        self, path: str
    ) -> None:
        torch.save(self._value.state_dict(), path)
        print('Value parameters saved to {}'.format(path))


    def load(
        self, path: str
    ) -> None:
        self._value.load_state_dict(torch.load(path, map_location=self._device))
        print('Value parameters loaded from {}'.format(path))



class QSarsaLearner:
    _device: torch.device
    _Q: QMLP
    _optimizer: torch.optim
    _target_Q: QMLP
    _total_update_step: int
    _target_update_freq: int
    _tau: float
    _gamma: float
    _batch_size: int

    def __init__(
        self,
        device: torch.device,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        depth: int,
        Q_lr: float,
        target_update_freq: int,
        tau: float,
        gamma: float,
        batch_size: int
    ) -> None:
        super().__init__()
        self._device = device
        self._Q = QMLP(state_dim, action_dim, hidden_dim, depth).to(device)
        self._optimizer = torch.optim.Adam(
            self._Q.parameters(),
            lr=Q_lr,
            )

        self._target_Q = QMLP(state_dim, action_dim, hidden_dim, depth).to(device)
        self._target_Q.load_state_dict(self._Q.state_dict())
        self._total_update_step = 0
        self._target_update_freq = target_update_freq
        self._tau = tau

        self._gamma = gamma
        self._batch_size = batch_size


    def __call__(
        self, s: torch.Tensor, a: torch.Tensor
    ) -> torch.Tensor:
        return self._Q(s, a)


    def loss(
        self, memory: ReplayMemory, pi
    ) -> torch.Tensor:
        state_batch, action_batch, reward_batch, next_state_batch = memory.sample(self._batch_size)
        state_batch = torch.FloatTensor(state_batch).to(self._device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self._device)
        action_batch = torch.FloatTensor(action_batch).to(self._device)
        reward_batch = torch.FloatTensor(reward_batch).to(self._device).unsqueeze(1)
        
        with torch.no_grad():
            target_Q_value = reward_batch + 1 * self._gamma * self._target_Q(next_state_batch, action_batch)
        
        Q = self._Q(state_batch, action_batch)
        
        loss = F.mse_loss(Q, target_Q_value)
        
        return loss


    def update(
        self, memory: ReplayMemory, pi
    ) -> float:
        Q_loss = self.loss(memory, pi)
        self._optimizer.zero_grad()
        Q_loss.backward()
        self._optimizer.step()

        self._total_update_step += 1
        if self._total_update_step % self._target_update_freq == 0:
            for param, target_param in zip(self._Q.parameters(), self._target_Q.parameters()):
                target_param.data.copy_(self._tau * param.data + (1 - self._tau) * target_param.data)

        return Q_loss.item()


    def save(
        self, path: str
    ) -> None:
        torch.save(self._Q.state_dict(), path)
        print('Q function parameters saved to {}'.format(path))


    def load(
        self, path: str
    ) -> None:
        self._Q.load_state_dict(torch.load(path, map_location=self._device))
        self._target_Q.load_state_dict(self._Q.state_dict())
        print('Q function parameters loaded from {}'.format(path))
