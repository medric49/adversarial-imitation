import torch

import utils
from actors import SimpleObsActor


class PGAgent:
    def __init__(self, state_dim, action_dim, lr):
        self.net = SimpleObsActor(state_dim, action_dim).to(utils.device())
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

    def act(self, obs, action_std=None, action_noise_clip=None):
        obs = torch.tensor(obs, dtype=torch.float, device=utils.device())
        obs = torch.unsqueeze(obs, dim=0)

        self.net.eval()
        with torch.no_grad():
            if action_std is not None:
                action = self.net(obs, std=action_std).sample(clip=action_noise_clip)
            else:
                action = self.net(obs).mean
        return action.cpu().numpy()[0]

    def update(self, obs, action, reward, next_obs, action_std):
        self.net.train()
        obs = torch.tensor(obs, dtype=torch.float, device=utils.device())
        action = torch.tensor(action, dtype=torch.float, device=utils.device())
        reward = torch.tensor(reward, dtype=torch.float, device=utils.device())

        self.optimizer.zero_grad()
        pred_action_dist = self.net(obs, std=action_std)
        log_prob = pred_action_dist.log_prob(action)
        loss = - (log_prob * reward).mean()
        loss.backward()
        self.optimizer.step()

        return {
            'batch_log_prob': log_prob.mean().item(),
            'batch_reward_mean': reward.mean().item(),
            'batch_loss': loss.item()
        }

