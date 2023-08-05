import dm_env.specs
from dm_control import suite, viewer
import numpy as np
from dm_env import specs

env = suite.load('cartpole', 'balance')

timestep = env.reset()
action_spec = env.action_spec()

viewer.launch(env)

while True:
    action = np.random.uniform(action_spec.minimum, action_spec.maximum)
    timestep = env.step(action)
    print(timestep)


