import random
import re

import numpy as np
import torch


def device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def schedule(schdl, step):
    try:
        return float(schdl)
    except ValueError:
        match = re.match(r'linear\((.+),(.+),(.+)\)', schdl)
        if match:
            init, final, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
        match = re.match(r'step_linear\((.+),(.+),(.+),(.+),(.+)\)', schdl)
        if match:
            init, final1, duration1, final2, duration2 = [
                float(g) for g in match.groups()
            ]
            if step <= duration1:
                mix = np.clip(step / duration1, 0.0, 1.0)
                return (1.0 - mix) * init + mix * final1
            else:
                mix = np.clip((step - duration1) / duration2, 0.0, 1.0)
                return (1.0 - mix) * final1 + mix * final2
    raise NotImplementedError(schdl)


def standardize(x, mean, std, eps=1e-6):
    return (x - mean) / (std + eps)


def normalize(x, x_min, x_max, eps=1e-6):
    return (x - x_min) / (x_max - x_min + eps)


class Until:
    def __init__(self, until):
        self._until = until

    def __call__(self, step):
        if self._until is None:
            return True
        until = self._until
        return step < until


class Every:
    def __init__(self, every):
        self._every = every

    def __call__(self, step):
        if self._every is None:
            return False
        every = self._every
        if step % every == 0:
            return True
        return False
