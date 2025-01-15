import torch
import torch.nn as nn
from torch.distributions import Normal
from typing import Tuple



def MLP(
    input_dim: int,
    hidden_dim: int,
    depth: int,
    output_dim: int,
    final_activation: str
) -> torch.nn.modules.container.Sequential:

    layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
    for _ in range(depth -1):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(hidden_dim, output_dim))
    if final_activation == 'relu':
        layers.append(nn.ReLU())
    elif final_activation == 'tanh':
        layers.append(nn.Tanh())
    elif final_activation == 'none':
        pass

    return nn.Sequential(*layers)



class ValueMLP(nn.Module):
    '''
    A MLP with:
    - A input layer: (batch_size, state_dim)
    - Several hidden layers: (batch_size, hidden_dim); "Several"=depth-1
    - A output layer: (batch_size, 1)
    All layers are ReLU activated.
    '''
    _net: torch.nn.modules.container.Sequential

    def __init__(
        self, state_dim: int, hidden_dim: int, depth: int
    ) -> None:
        super().__init__()
        self._net = MLP(state_dim, hidden_dim, depth, 1, 'none')

    def forward(
        self, s: torch.Tensor
    ) -> torch.Tensor:
        return self._net(s)



class QMLP(nn.Module):
    '''
    A MLP with:
    - A input layer: (batch_size, state_dim+action_dim)
    - Several hidden layers: (batch_size, hidden_dim); "Several"=depth-1
    - A output layer: (batch_size, 1)
    All layers are ReLU activated.
    '''
    _net: torch.nn.modules.container.Sequential

    def __init__(
        self, 
        state_dim: int, action_dim: int, hidden_dim: int, depth:int
    ) -> None:
        super().__init__()
        self._net = MLP((state_dim + action_dim), hidden_dim, depth, 1, 'none')

    def forward(
        self, s: torch.Tensor, a: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([s, a], dim=1)
        return self._net(sa)



class GaussPolicyMLP(nn.Module):
    '''
    A MLP with:
    - A input layer: (batch_size, state_dim)
    - Several hidden layers: (batch_size, hidden_dim); "Several"=depth-1
    - Two output layers: (batch_size, action_dim) and (batch_size, action_dim); The two represent the mean and log_std of the Gaussian distribution respectively.
    All layers are ReLU activated, except the output layers with tanh activation.

    The return is a torch.distributions.Normal object, which can be used to sample actions and compute log probabilities.
    '''
    _net: torch.nn.modules.container.Sequential
    _log_std_bound: tuple

    def __init__(
        self, 
        state_dim: int, hidden_dim: int, depth: int, action_dim: int, 
    ) -> None:
        super().__init__()
        self._net = MLP(state_dim, hidden_dim, depth, (2 * action_dim), 'none')
        self._log_std_bound = (-10., 50.)


    def forward(
        self, s: torch.Tensor
    ) -> torch.distributions:

        mu, log_std = self._net(s).chunk(2, dim=-1)
        def soft_clamp(x: torch.Tensor, bound: tuple) -> torch.Tensor:
            low, high = bound
            #x = torch.tanh(x)
            x = low + 0.5 * (high - low) * (x + 1)
            return x
        log_std = torch.clamp(log_std, min=self._log_std_bound[0], max=self._log_std_bound[1]) # The log_std is clamped to be within (-5, 5) to prevent numerical instability.
        std = log_std.exp()

        dist = Normal(mu, std)
        return dist
