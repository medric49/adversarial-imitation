import hydra
from dm_control import suite

import agents
import envs
import memories
import rl_trainer
import utils


@hydra.main(config_path='config', config_name='default')
def main(config):
    utils.set_seed_everywhere(config.seed)
    domain = 'finger'
    task = 'turn_easy'
    # domain = 'cartpole'
    # task = 'balance'

    train_env = suite.load(domain, task)
    eval_env = suite.load(domain, task)
    train_env = envs.wrap(train_env)
    eval_env = envs.wrap(eval_env)

    state_dim = train_env.observation_spec().shape[0]
    action_dim = train_env.action_spec().shape[0]
    agent = agents.PGAgent(state_dim, action_dim, lr=config.lr)
    memory = memories.ReplayMemory(config.memory_size, config.discount)

    trainer = rl_trainer.PGTrainer(config, agent, memory, train_env, eval_env)
    trainer.train()


if __name__ == '__main__':
    main()
