import torch

import memories
import utils
import nets
import torch.nn.functional as F


class PGAgent:
    def __init__(self, state_dim, action_dim, lr):
        self.actor = nets.BasicActor(state_dim, action_dim).to(utils.device())
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

    def act(self, obs, action_std=None, action_noise_clip=None):
        obs = torch.tensor(obs, dtype=torch.float, device=utils.device())
        obs = torch.unsqueeze(obs, dim=0)

        self.actor.eval()
        with torch.no_grad():
            if action_std is not None:
                action = self.actor(obs, std=action_std).sample(clip=action_noise_clip)
            else:
                action = self.actor(obs).mean
        return action.cpu().numpy()[0]

    def update(self, memory: memories.ReplayMemory, batch_size, action_std):
        obs, action, reward, next_obs = memory.sample_steps(batch_size, reward_to_go=True)
        obs = torch.tensor(obs, dtype=torch.float, device=utils.device())
        action = torch.tensor(action, dtype=torch.float, device=utils.device())
        reward = torch.tensor(reward, dtype=torch.float, device=utils.device())

        # weights = utils.normalize(reward, reward.min().detach(), reward.max().detach())
        weights = utils.standardize(reward, reward.mean().detach(), reward.std().detach())

        self.actor.train()
        self.optimizer.zero_grad()
        pred_action_dist = self.actor(obs, std=action_std)
        log_prob = pred_action_dist.log_prob(action).sum(-1, keepdim=True)
        loss = - (log_prob * weights).mean()
        loss.backward()
        self.optimizer.step()

        return {
            'batch_log_prob': log_prob.mean().item(),
            'batch_reward_mean': reward.mean().item(),
            'batch_loss': loss.item()
        }


class QACAgent:
    def __init__(self, state_dim, action_dim, lr):
        self.actor = nets.BasicActor(state_dim, action_dim).to(utils.device())
        self.critic = nets.BasicQCritic(state_dim, action_dim).to(utils.device())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

    def act(self, obs, action_std=None, action_noise_clip=None):
        obs = torch.tensor(obs, dtype=torch.float, device=utils.device())
        obs = torch.unsqueeze(obs, dim=0)

        self.actor.eval()
        with torch.no_grad():
            if action_std is not None:
                action = self.actor(obs, std=action_std).sample(clip=action_noise_clip)
            else:
                action = self.actor(obs).mean
        return action.cpu().numpy()[0]

    def update(self, memory: memories.ReplayMemory, batch_size, action_std, discount):
        # Update Actor
        obs, action, reward, next_obs = memory.sample_recent_steps(batch_size, reward_to_go=False)
        obs = torch.tensor(obs, dtype=torch.float, device=utils.device())
        action = torch.tensor(action, dtype=torch.float, device=utils.device())

        self.critic.eval()
        self.actor.train()
        self.actor_optimizer.zero_grad()
        with torch.no_grad():
            q_values = self.critic(obs, action)
        pred_action_dist = self.actor(obs, std=action_std)
        log_prob = pred_action_dist.log_prob(action).sum(-1, keepdim=True)
        actor_loss = - (log_prob * q_values).mean()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update Critic
        obs, action, reward, next_obs = memory.sample_steps(batch_size, reward_to_go=False)
        obs = torch.tensor(obs, dtype=torch.float, device=utils.device())
        action = torch.tensor(action, dtype=torch.float, device=utils.device())
        reward = torch.tensor(reward, dtype=torch.float, device=utils.device())
        next_obs = torch.tensor(next_obs, dtype=torch.float, device=utils.device())

        self.actor.eval()
        with torch.no_grad():
            next_action = self.actor(next_obs).mean
            next_q_values = self.critic(next_obs, next_action).flatten()
            td_target = reward + discount * next_q_values

        self.critic.train()
        self.critic_optimizer.zero_grad()
        q_values = self.critic(obs, action).flatten()
        critic_loss = F.mse_loss(q_values, td_target)
        critic_loss.backward()
        self.critic_optimizer.step()

        return {
            'batch_log_prob': log_prob.mean().item(),
            'batch_reward_mean': reward.mean().item(),
            'batch_actor_loss': actor_loss.item(),
            'batch_critic_loss': critic_loss.item(),
            'batch_q_value_mean': q_values.mean().item()
        }

