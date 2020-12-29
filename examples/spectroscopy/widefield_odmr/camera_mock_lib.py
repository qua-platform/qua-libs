import numpy as np


class cam_mock:
    def __init__(self):
        self.name = 'camera1'
        self.num = 0

    def __repr__(self):
        return f"{self.name} with {self.num} images preallocated"

    def arm(self):
        print(f"{self.name} armed")

    def allocate_buffer(self, num):
        self.num = num
        print(f"allocated {self.num} in memory")

    def get_image(self):
        print(f"got 1 image!")
        return np.random.randint(0, 255, (480, 640))
