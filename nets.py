from torch import nn
from torch import distributions as pyd
import torch
from torch.distributions.utils import _standard_normal


class TruncatedNormal(pyd.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clip(self, x):
        clipped_x = torch.clip(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clipped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape,
                               dtype=self.loc.dtype,
                               device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clip(eps, -clip, clip)
        x = self.loc + eps
        return self._clip(x)


class BasicActor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()
        )

    def forward(self, obs, std=None):
        if std is None:
            std = 0.001
        action_mean = self.policy_net(obs)
        std = torch.ones_like(action_mean) * std
        dist = TruncatedNormal(action_mean, std)
        return dist


class BasicQCritic(nn.Module):
    def __init__(self, state_dim, actor_dim):
        super().__init__()

        self.q_value_net = nn.Sequential(
            nn.Linear(state_dim + actor_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, obs, action):
        q_value = self.q_value_net(torch.concatenate([obs, action], dim=1))
        return q_value

