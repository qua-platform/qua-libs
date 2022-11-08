import numpy as np
import time


def reshape_for_do2d(data: np.ndarray, n_averages, qdac_x_resolution, qdac_y_resolution, opx_x_resolution,
                     opx_y_resolution):
    reshaped = data.reshape(-1, n_averages, opx_x_resolution, opx_y_resolution)
    averaged = reshaped.mean(axis=1)

    to_stack = averaged.reshape(qdac_x_resolution, qdac_y_resolution, opx_x_resolution, opx_y_resolution)
    stacked = np.hstack([np.vstack(array) for array in to_stack])

    return stacked





class TimingModule:

    def __init__(self):
        self.checkpoints = []
        self.names = []

    def add_checkpoint(self, name):
        self.checkpoints.append(time.time())
        self.names.append(name)
        print(name)

    def print(self):
        differences = np.diff(self.checkpoints)

        for i, time in zip(self.names, differences):
            print(f'{i}: {time}')
