import numpy as np


class Demo(object):

    def __init__(self, observations, random_seed=None):
        self._observations = observations
        self.random_seed = random_seed
        self.base_matrix = None
        self.obj_matrix = None

    def __len__(self):
        return len(self._observations)

    def __getitem__(self, i):
        return self._observations[i]

    def restore_state(self):
        np.random.set_state(self.random_seed)
