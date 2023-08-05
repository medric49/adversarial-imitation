import torch

import memories
import utils
from actors import SimpleObsActor


class PGAgent:
    def __init__(self, state_dim, action_dim, lr):
        self.net = SimpleObsActor(state_dim, action_dim).to(utils.device())
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

    def act(self, obs, noisy=False, max_noise=None):
        obs = torch.tensor(obs, dtype=torch.float, device=utils.device())
        obs = torch.unsqueeze(obs, dim=0)

        self.net.eval()
        with torch.no_grad():
            action_dist = self.net(obs, std=0.5)
            action = action_dist.sample(clip=max_noise) if noisy else action_dist.mean
        return action.cpu().numpy()

    def update(self, obs, action, reward, next_obs):
        self.net.train()
        obs = torch.tensor(obs, dtype=torch.float, device=utils.device())
        action = torch.tensor(action, dtype=torch.float, device=utils.device())
        reward = torch.tensor(reward, dtype=torch.float, device=utils.device())

        self.optimizer.zero_grad()
        pred_action_dist = self.net(obs, std=0.5)
        log_prob = pred_action_dist.log_prob(action)
        loss = - (log_prob * reward).mean()
        loss.backward()
        self.optimizer.step()

