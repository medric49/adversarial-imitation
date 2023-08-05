from torch.utils.tensorboard import SummaryWriter


class Logger(object):
    def __init__(self, log_dir):
        self._sw = SummaryWriter(log_dir)

    def log(self, key, value, step):
        self._sw.add_scalar(key, value, step)

    def log_metrics(self, metrics, step, ty):
        for key, value in metrics.items():
            self.log(f'{ty}/{key}', value, step)
