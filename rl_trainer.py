import numpy as np

import memories
import utils
from agents import PGAgent, QACAgent
from loggers import Logger
from recorders import EpisodeRecorder


class Trainer:
    def __init__(self, config, agent, memory: memories.ReplayMemory, train_env, eval_env):
        self.config = config
        self.agent = agent
        self.memory = memory
        self.train_env = train_env
        self.eval_env = eval_env
        self.logger = Logger('tb')
        self.train_recorder = EpisodeRecorder('train_video')
        self.eval_recorder = EpisodeRecorder('eval_video')
        self.action_std = utils.schedule(self.config.action_std_schedule, 0)
        self.global_step = 0
        self.global_episode = 0

    def eval(self):
        episode_returns = []
        for i in range(self.config.num_eval_episodes):
            timestep = self.eval_env.reset()
            episode_return = 0.

            self.eval_recorder.start_recording(self.eval_env.render(self.config.render_im_width, self.config.render_im_height))

            while not timestep.last():
                action = self.agent.act(timestep.observation)
                timestep = self.eval_env.step(action)
                episode_return += timestep.reward

                self.eval_recorder.record(self.eval_env.render(self.config.render_im_width, self.config.render_im_height))
                if timestep.last():
                    episode_returns.append(episode_return)
                    self.eval_recorder.save(f'{self.global_episode}_{i}_{int(episode_return)}.mp4')

        return_mean = np.mean(episode_returns)
        self.logger.log('eval/episode_return', return_mean, self.global_step)


class PGTrainer(Trainer):
    def __init__(self, config, agent: PGAgent, memory: memories.ReplayMemory, train_env, eval_env, **kwargs):
        super().__init__(config, agent, memory, train_env, eval_env)

    def update(self, action_std):
        return self.agent.update(self.memory, self.config.batch_size, action_std=action_std)

    def train(self):
        update_every_step = utils.Every(self.config.update_every_steps)
        eval_every_step = utils.Every(self.config.eval_every_steps)
        seed_until_step = utils.Until(self.config.num_seed_steps)

        while self.global_step < self.config.num_train_steps:
            self.global_episode += 1
            print(f'*** Episode {self.global_episode}')

            timestep = self.train_env.reset()
            episode_return = 0.

            self.memory.add_timestep(timestep)
            self.train_recorder.start_recording(self.train_env.render(self.config.render_im_width, self.config.render_im_height))

            while not timestep.last():
                self.global_step += 1
                # if seed_until_step(self.global_step):
                #     action = np.random.uniform(self.train_env.action_spec().minimum, self.train_env.action_spec().maximum)
                # else:
                #     action = self.agent.act(timestep.observation, action_std=self.action_std)
                action = self.agent.act(timestep.observation, action_std=self.action_std)
                timestep = self.train_env.step(action)
                episode_return += timestep.reward

                self.memory.add_timestep(timestep)
                self.train_recorder.record(self.train_env.render(self.config.render_im_width, self.config.render_im_height))

                if update_every_step(self.global_step):
                    metrics = None
                    for _ in range(self.config.num_updates):
                        metrics = self.update(action_std=self.action_std)
                    self.logger.log_metrics(metrics, self.global_step, 'train')
                    self.action_std = utils.schedule(self.config.action_std_schedule, self.global_step)

                if timestep.last():
                    self.logger.log('train/episode_return', episode_return, self.global_episode)
                    self.train_recorder.save(f'{self.global_episode}_{int(episode_return)}.mp4')

            if eval_every_step(self.global_step) and not seed_until_step(self.global_step):
                self.eval()


class ACTrainer(Trainer):
    def __init__(self, config, agent, memory: memories.ReplayMemory, train_env, eval_env):
        super().__init__(config, agent, memory, train_env, eval_env)

    def update(self, action_std):
        return self.agent.update(self.memory, self.config.batch_size, action_std=action_std, discount=self.config.discount)

    def train(self):
        update_every_step = utils.Every(self.config.update_every_steps)
        eval_every_step = utils.Every(self.config.eval_every_steps)
        seed_until_step = utils.Until(self.config.num_seed_steps)

        while self.global_step < self.config.num_train_steps:
            self.global_episode += 1
            print(f'*** Episode {self.global_episode}')

            timestep = self.train_env.reset()
            episode_return = 0.

            self.memory.add_timestep(timestep)
            self.train_recorder.start_recording(self.train_env.render(self.config.render_im_width, self.config.render_im_height))

            while not timestep.last():
                self.global_step += 1
                if seed_until_step(self.global_step):
                    action = np.random.uniform(self.train_env.action_spec().minimum, self.train_env.action_spec().maximum)
                else:
                    action = self.agent.act(timestep.observation, action_std=self.action_std)

                timestep = self.train_env.step(action)
                episode_return += timestep.reward

                self.memory.add_timestep(timestep)
                self.train_recorder.record(self.train_env.render(self.config.render_im_width, self.config.render_im_height))

                if not seed_until_step(self.global_step) and update_every_step(self.global_step):
                    metrics = None
                    for _ in range(self.config.num_updates):
                        metrics = self.update(action_std=self.action_std)
                    self.logger.log_metrics(metrics, self.global_step, 'train')
                    self.action_std = utils.schedule(self.config.action_std_schedule, self.global_step)

                if timestep.last():
                    self.logger.log('train/episode_return', episode_return, self.global_episode)
                    self.train_recorder.save(f'{self.global_episode}_{int(episode_return)}.mp4')

            if eval_every_step(self.global_step) and not seed_until_step(self.global_step):
                self.eval()
