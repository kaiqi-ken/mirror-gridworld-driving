import cv2
import numpy as np


class Visualizer():
    def __init__(self, map_size=10, tile_size=30):
        self.map_size = int(map_size)
        self.tile_size = tile_size
        self.width = int(4 * tile_size)
        self.height = int((2 * map_size + 1) * tile_size)
        self.image = self.reset()

    def reset(self):
        return np.full((self.height, self.width, 3), 255, np.uint8)

    def drawImage(self, ego_car_obs, other_car_obs, env_type='clear'):
        # Restart with blank image first
        self.image = self.reset()
        ego_car_obs = ego_car_obs.astype(int)
        other_car_obs[1::2] = other_car_obs[1::2] * self.map_size
        other_car_obs = other_car_obs.astype(int)
        # Lane Markings
        self.image = cv2.line(self.image, (self.tile_size, 0), (self.tile_size, self.height), (0, 0, 0), 1)
        self.image = cv2.line(self.image, (self.tile_size * 3, 0), (self.tile_size * 3, self.height), (0, 0, 0), 1)
        for i in range(2 * self.map_size + 1):
            if i % 2 == 0:
                self.image = cv2.line(self.image, (self.tile_size * 2, i * self.tile_size), (self.tile_size * 2, (i + 1) * self.tile_size), (0, 0, 0), 1)
        # Ego Vehicle
        self.image = cv2.rectangle(self.image, (self.tile_size * (ego_car_obs[0] + 1), self.map_size * self.tile_size), (self.tile_size * (ego_car_obs[0] + 2), (self.map_size + 1) * self.tile_size), (255, 0, 0), -1)
        # Other Vehicles (front-left, front-right, rear-left, rear-right)
        if env_type == 'clear':
            for i in range(0, other_car_obs.shape[0], 2):
                self.image = cv2.rectangle(self.image,
                                           (self.tile_size * (other_car_obs[i] + 1), (self.map_size - other_car_obs[i+1]) * self.tile_size),
                                           (self.tile_size * (other_car_obs[i] + 2), (self.map_size - other_car_obs[i+1] + 1) * self.tile_size),
                                           (0, 0, 255), -1
                                           )
        else:
            for i in range(0, other_car_obs.shape[0], 2):
                if 0 <= other_car_obs[i+1] <= 2:
                    self.image = cv2.rectangle(self.image,
                                               (self.tile_size * (other_car_obs[i] + 1), (self.map_size - other_car_obs[i+1]) * self.tile_size),
                                               (self.tile_size * (other_car_obs[i] + 2), (self.map_size - other_car_obs[i+1] + 1) * self.tile_size),
                                               (0, 0, 255), -1
                                               )

    def render(self):
        cv2.imshow('simple carla', self.image)
        cv2.waitKey(50)

if __name__ == '__main__':
    vis = Visualizer(map_size=10, tile_size=30)
    vis.drawImage(np.array([0, 1]), np.array([0, 5, 1, 6, 0, -4, 1, -3]))
    vis.render()