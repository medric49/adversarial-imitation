from typing import NamedTuple, Any

import dm_env
import numpy as np
from dm_control.suite.wrappers import action_scale
from dm_env import Environment, StepType, specs, TimeStep


class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        return getattr(self, attr)


class ObservationOverrideWrapper(dm_env.Environment):
    def __init__(self, env, dtype):
        self._env = env
        self.dtype = dtype
        dim = 0
        for name, spec in env.observation_spec().items():
            item_dim = sum(spec.shape)
            dim += item_dim if item_dim != 0 else 1
        self._observation_spec = specs.Array((dim,), dtype, 'observation')

    def override_timestep(self, timestep):
        observation = []
        for name, value in timestep.observation.items():
            observation.append(value.flatten())
        observation = np.concatenate(observation, dtype=self.dtype)
        return timestep._replace(observation=observation)

    def step(self, action):
        timestep = self._env.step(action)
        return self.override_timestep(timestep)

    def observation_spec(self):
        return self._observation_spec

    def action_spec(self):
        return self._env.action_spec()

    def reset(self):
        timestep = self._env.reset()
        return self.override_timestep(timestep)

    def __getattr__(self, name):
        return getattr(self._env, name)


class ActionOverrideWrapper(dm_env.Environment):
    def __init__(self, env, dtype):
        self._env = env
        wrapped_action_spec = env.action_spec()
        self._action_spec = specs.BoundedArray(wrapped_action_spec.shape,
                                               dtype,
                                               wrapped_action_spec.minimum,
                                               wrapped_action_spec.maximum,
                                               'action')

    def step(self, action):
        action = action.astype(self._env.action_spec().dtype)
        return self._env.step(action)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._action_spec

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ExtendedTimeStepWrapper(Environment):
    def __init__(self, env):
        self._env = env

    def reset(self):
        time_step = self._env.reset()
        return self._augment_time_step(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._augment_time_step(time_step, action)

    def _augment_time_step(self, time_step, action=None):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        return ExtendedTimeStep(observation=time_step.observation,
                                step_type=time_step.step_type,
                                action=action,
                                reward=time_step.reward or 0.0,
                                discount=time_step.discount or 1.0)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class RenderingWrapper(Environment):
    def __init__(self, env):
        self._env = env

    def reset(self) -> TimeStep:
        return self._env.reset()

    def step(self, action) -> TimeStep:
        return self._env.step(action)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)

    def render(self, width, height, camera_id=0):
        return self._env.physics.render(height, width, camera_id=camera_id)


def wrap(env):
    env = RenderingWrapper(env)
    env = ActionOverrideWrapper(env, np.float32)
    minimum = np.ones_like(env.action_spec().minimum) * -1.
    maximum = np.ones_like(env.action_spec().maximum) * 1.
    env = action_scale.Wrapper(env, minimum=minimum, maximum=maximum)
    env = ObservationOverrideWrapper(env, np.float32)
    env = ExtendedTimeStepWrapper(env)
    return env
