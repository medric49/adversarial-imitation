from loggers import Logger
from recorders import EpisodeRecorder


class PGTrainer:
    def __init__(self, config, agent, memory, train_env, eval_env):
        self.config = config
        self.agent = agent
        self.memory = memory
        self.train_env = train_env
        self.eval_env = eval_env
        self.logger = Logger('tb')
        self.train_recorder = EpisodeRecorder('train_video')
        self.eval_recorder = EpisodeRecorder('eval_video')
        self.global_step = 0
        self.global_episode = 0

    def update(self):
        obs, action, reward, next_obs = self.memory.sample_steps(size=64)
        self.agent.update(obs, action, reward, next_obs)

    def train(self):
        while self.global_step < self.config.num_train_steps:
            self.global_episode += 1
            timestep = self.train_env.reset()

            episode_return = 0.
            self.memory.add_timestep(timestep)
            self.train_recorder.start_recording(self.train_env.render(256, 256))

            while not timestep.last():
                self.global_step += 1
                action = self.agent.act(timestep.observation, noisy=True)
                timestep = self.train_env.step(action)

                episode_return += timestep.reward
                self.memory.add_timestep(timestep)
                self.train_recorder.record(self.train_env.render(256, 256))

                if timestep.last():
                    self.update()
                    self.logger.log('train/episode_return', episode_return, self.global_episode)
                    self.train_recorder.save(f'{self.global_episode}_{int(episode_return)}.mp4')
