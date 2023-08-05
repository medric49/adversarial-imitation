from pathlib import Path

import imageio


class EpisodeRecorder:
    def __init__(self, save_dir, fps=20):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.frames = None
        self.fps = fps

    def start_recording(self, frame):
        self.frames = [frame]

    def record(self, frame):
        self.frames.append(frame)

    def save(self, filename):
        path = self.save_dir / filename
        imageio.mimsave(str(path), self.frames, fps=self.fps)
