import random

import torch
import numpy as np
# from simple_carla.maxent.maxent import *
from maxent.maxent import *

np.set_printoptions(precision=2)

def generate_cheating_obs_mask(other_car_dis):
    obs_mask_rsample = torch.zeros(size=other_car_dis.size()).type(torch.float).to(other_car_dis.device)
    obs_mask_rsample[..., 0:2] = (0 <= other_car_dis[..., 0::2]).type(torch.float).to(other_car_dis.device) * (
                other_car_dis[..., 0::2] <= 0.22).type(torch.float).to(other_car_dis.device)

    obs_mask_rsample = torch.cat([obs_mask_rsample[..., 0:1].repeat(2),
                                  obs_mask_rsample[..., 1:2].repeat(2),
                                  obs_mask_rsample[..., 2:3].repeat(2),
                                  obs_mask_rsample[..., 3:4].repeat(2)], dim=-1)
    return obs_mask_rsample
    
class FourCarEnv:
    def __init__(self):
        # ego car state: (lane_id: 0, 1; speed: 1); action (stay, left, right)
        self.ego_car = np.array([0., 1.])

        # other vehicles (lane_id: 0, 1; vertical distance: -10~10; speed: -2, -1, 0, 1, 2)
        self.first_car = np.array([0, 5, 1])
        self.first_car_pre = np.array([0, 5, 1])
        self.second_car = np.array([1, 5, 1])
        self.second_car_pre = np.array([1, 5, 1])
        self.third_car = np.array([0, -5, 1])
        self.third_car_pre = np.array([0, 5, 1])
        self.fourth_car = np.array([1, -5, 1])
        self.fourth_car_pre = np.array([1, 5, 1])

        self.map_size = 10.0
        self.speed_size = 3.0

        self.first_car_init = None
        self.second_car_init = None
        self.third_car_init = None
        self.fouth_car_init = None

    def normalizer(self, values, is_ego=False):
        """
        Normalize all the values in ego car or other cars
        """
        if is_ego:
            values[-1] /= self.speed_size
        else:
            values[-2] /= self.map_size
            values[-1] /= self.speed_size

        return values

    def reset(self, ego_car=None, other_car=None):
        # ego car (lane_id: 0, 1; speed: 1)
        if ego_car is None:
            self.ego_car = self.normalizer(np.array([float(random.randint(0, 1)), 1.]), is_ego=True)
        else:
            if ego_car == 'l':
                self.ego_car = self.normalizer(np.array([0, 1.0]), is_ego=True)
            else:
                self.ego_car = self.normalizer(np.array([1, 1.0]), is_ego=True)

        if self.first_car_init is None:
            # other vehicles (lane_id: 0, 1; vertical distance: -10~10; absolute speed: -3~3)
            # front-left, front-right, rear-left, rear-right
            self.first_car = self.normalizer(np.array([0., float(random.randint(6, 7)), 1.]))  # 5 10
            # self.first_car = self.normalizer(np.array([0., 5, 1.]))
            self.first_car_pre = self.first_car.copy()
            self.second_car = self.normalizer(np.array([1., float(random.randint(6, 7)), 1.]))
            # self.second_car = self.normalizer(np.array([1., 5, 1.]))
            self.second_car_pre = self.second_car.copy()
            self.third_car = self.normalizer(np.array([0., float(random.randint(-7, -6)), 1.]))
            # self.third_car = self.normalizer(np.array([0., -5, 1.]))
            self.third_car_pre = self.third_car.copy()
            self.fourth_car = self.normalizer(np.array([1., float(random.randint(-7, -6)), 1.]))
            # self.fourth_car = self.normalizer(np.array([1., -5, 1.]))
            self.fourth_car_pre = self.fourth_car.copy()
        else:
            # other vehicles (lane_id: 0, 1; vertical distance: -10~10; absolute speed: -3~3)
            # front-left, front-right, rear-left, rear-right
            self.first_car = self.first_car_init.copy()
            # self.first_car = self.normalizer(np.array([0., 5, 1.]))
            self.first_car_pre = self.first_car.copy()
            self.second_car = self.second_car_init.copy()
            # self.second_car = self.normalizer(np.array([1., 5, 1.]))
            self.second_car_pre = self.second_car.copy()
            self.third_car = self.third_car_init.copy()
            # self.third_car = self.normalizer(np.array([0., -5, 1.]))
            self.third_car_pre = self.third_car.copy()
            self.fourth_car = self.fouth_car_init.copy()
            # self.fourth_car = self.normalizer(np.array([1., -5, 1.]))
            self.fourth_car_pre = self.fourth_car.copy()

        # Randomly choose 1 of the 4 cars to slow down/speed up
        if other_car is None:
            car_id = random.choice([0, 1, 2, 3])  # 0: front-left, 1: front-right, 2: rear-left, 3:rear-right
        else:
            if other_car == 'lf':
                car_id = 0
            elif other_car == 'lr':
                car_id = 2
            elif other_car == 'rf':
                car_id = 1
            else:
                car_id = 3
        if car_id == 0:
            self.first_car[2] = 0 / self.speed_size  # random.randint(-1, 0) / self.speed_size
            self.ego_car[0] = 0
        elif car_id == 1:
            self.second_car[2] = 0 / self.speed_size  # random.randint(-1, 0) / self.speed_size
            self.ego_car[0] = 1
        elif car_id == 2:
            self.third_car[2] = 2 / self.speed_size  # random.randint(2, 3) / self.speed_size
            self.ego_car[0] = 0
        elif car_id == 3:
            self.fourth_car[2] = 2 / self.speed_size  # random.randint(2, 3) / self.speed_size
            self.ego_car[0] = 1

        ego_car_obs = self.ego_car.copy()
        other_car_obs = np.concatenate([self.first_car[:-1],
                                        self.second_car[:-1],
                                        self.third_car[:-1],
                                        self.fourth_car[:-1]])

        # -1 left, 0 same, 1 right
        h_dir_first = (self.first_car[0:1] - self.ego_car[0:1]).astype(np.float)

        h_dir_second = (self.second_car[0:1] - self.ego_car[0:1]).astype(np.float)

        h_dir_third = (self.third_car[0:1] - self.ego_car[0:1]).astype(np.float)

        h_dir_fourth = (self.fourth_car[0:1] - self.ego_car[0:1]).astype(np.float)

        other_car_text = np.concatenate([h_dir_first, self.first_car[1:],
                                         h_dir_second, self.second_car[1:],
                                         h_dir_third, self.third_car[1:],
                                         h_dir_fourth, self.fourth_car[1:]])

        done = False

        reward = 0.0
        # Give penalty and end the episode if ego car goes out of the road
        if 0 <= self.ego_car[0] <= 1:
            lane_reward = 0
        else:
            lane_reward = -5
            done = True

        collision_reward = 0.0

        speed_reward = self.ego_car[1] * 0.2

        reward = lane_reward + collision_reward + speed_reward

        return ego_car_obs, other_car_obs, other_car_text, reward, done

    def env_prediction(self, ego_car_obs, other_car_obs, other_car_text, action, maxent_theta):
        first_car = np.concatenate([other_car_obs[0:2], other_car_text[2:3]], axis=-1)
        second_car = np.concatenate([other_car_obs[2:4], other_car_text[5:6]], axis=-1)
        third_car = np.concatenate([other_car_obs[4:6], other_car_text[8:9]], axis=-1)
        fourth_car = np.concatenate([other_car_obs[6:8], other_car_text[11:12]], axis=-1)

        ego_car = ego_car_obs.copy()
        first_car_pre = first_car.copy()
        second_car_pre = second_car.copy()
        third_car_pre = third_car.copy()
        fourth_car_pre = fourth_car.copy()

        # move other vehicles
        # other vehicles (lane_id: 0, 1; vertical distance: -100~100; speed: -3~3)
        ego_car_speed = ego_car[1]

        first_car[1] = first_car[1] * self.map_size + (first_car[2] - ego_car_speed) * self.speed_size
        second_car[1] = second_car[1] * self.map_size + (second_car[2] - ego_car_speed) * self.speed_size
        third_car[1] = third_car[1] * self.map_size + (third_car[2] - ego_car_speed) * self.speed_size
        fourth_car[1] = fourth_car[1] * self.map_size + (fourth_car[2] - ego_car_speed) * self.speed_size

        first_car[1] = np.clip(first_car[1], a_min=-8, a_max=8) / self.map_size
        second_car[1] = np.clip(second_car[1], a_min=-8, a_max=8) / self.map_size
        third_car[1] = np.clip(third_car[1], a_min=-8, a_max=8) / self.map_size
        fourth_car[1] = np.clip(fourth_car[1], a_min=-8, a_max=8) / self.map_size

        # move ego vehicle
        # action (1,0,0):stay, (0,1,0):left, (0,0,1):right
        if action == 1.0:
            ego_car[0] -= 1.0
        elif action == 2.0:
            ego_car[0] += 1.0
        ego_car[0] = np.clip(ego_car[0], a_min=-1, a_max=2)

        ego_car_obs = ego_car.copy()
        other_car_obs = np.concatenate([first_car[:-1],
                                        second_car[:-1],
                                        third_car[:-1],
                                        fourth_car[:-1]])

        h_dir_first = (first_car[0:1] - ego_car[0:1]).astype(np.float)

        h_dir_second = (second_car[0:1] - ego_car[0:1]).astype(np.float)

        h_dir_third = (third_car[0:1] - ego_car[0:1]).astype(np.float)

        h_dir_fourth = (fourth_car[0:1] - ego_car[0:1]).astype(np.float)

        other_car_text = np.concatenate([h_dir_first, first_car[1:],
                                         h_dir_second, second_car[1:],
                                         h_dir_third, third_car[1:],
                                         h_dir_fourth, fourth_car[1:]])

        collision = False

        if ego_car[0] == first_car[0] and first_car[1] * first_car_pre[1] <= 0.0:
            collision = True
        if ego_car[0] == second_car[0] and second_car[1] * second_car_pre[1] <= 0.0:
            collision = True
        if ego_car[0] == third_car[0] and third_car[1] * third_car_pre[1] <= 0.0:
            collision = True
        if ego_car[0] == fourth_car[0] and fourth_car[1] * fourth_car_pre[1] <= 0.0:
            collision = True

        done = collision
        # print(action)

        if maxent_theta is None:
            reward = self.get_prescribed_reward(ego_car=ego_car,
                                                first_car=first_car,
                                                second_car=second_car,
                                                third_car=third_car,
                                                fourth_car=fourth_car,
                                                collision=collision,
                                                action=action)
        else:
            ego_obs = ego_car_obs[0:1]
            if 0 <= ego_obs <= 1:
                ego_obs = 0
            else:
                ego_obs = 1
            ego_obs = np.array([ego_obs])

            other_obs = other_car_obs[1::2].copy()
            other_lanes = other_car_obs[0::2].copy()
            if other_lanes[0] != ego_car_obs[0] and ego_obs == 0:
                other_obs[0] = 0.9
            if other_lanes[1] != ego_car_obs[0] and ego_obs == 0:
                other_obs[1] = 0.9
            if other_lanes[2] != ego_car_obs[0] and ego_obs == 0:
                other_obs[2] = -0.9
            if other_lanes[3] != ego_car_obs[0] and ego_obs == 0:
                other_obs[3] = -0.9

            if ego_obs == 1:
                other_obs[:] = 0

            action_feature = np.argmax(action)
            if action_feature == 0:
                action_feature = 0
            else:
                action_feature = 1

            action_feature = np.array([action_feature])

            feature = np.concatenate([ego_obs,
                                      other_obs,
                                      action_feature], axis=-1)

            feature_copy = feature.copy()
            feature_copy[1:1 + 4] *= 5
            reward = get_reward_compress(feature_copy, maxent_theta)

        return ego_car_obs, other_car_obs, other_car_text, reward, done

    def forward(self, action):
        self.first_car_pre = self.first_car.copy()
        self.second_car_pre = self.second_car.copy()
        self.third_car_pre = self.third_car.copy()
        self.fourth_car_pre = self.fourth_car.copy()

        # move other vehicles
        # other vehicles (lane_id: 0, 1; vertical distance: -100~100; speed: -3~3)
        ego_car_speed = self.ego_car[1]

        self.first_car[1] = self.first_car[1] * self.map_size + (self.first_car[2] - ego_car_speed) * self.speed_size
        self.second_car[1] = self.second_car[1] * self.map_size + (self.second_car[2] - ego_car_speed) * self.speed_size
        self.third_car[1] = self.third_car[1] * self.map_size + (self.third_car[2] - ego_car_speed) * self.speed_size
        self.fourth_car[1] = self.fourth_car[1] * self.map_size + (self.fourth_car[2] - ego_car_speed) * self.speed_size

        self.first_car[1] = np.clip(self.first_car[1], a_min=-8, a_max=8) / self.map_size
        self.second_car[1] = np.clip(self.second_car[1], a_min=-8, a_max=8) / self.map_size
        self.third_car[1] = np.clip(self.third_car[1], a_min=-8, a_max=8) / self.map_size
        self.fourth_car[1] = np.clip(self.fourth_car[1], a_min=-8, a_max=8) / self.map_size

        # move ego vehicle
        # action (1,0,0):stay, (0,1,0):left, (0,0,1):right
        if action[1] == 1.0:
            self.ego_car[0] -= 1.0
        elif action[2] == 1.0:
            self.ego_car[0] += 1.0
        self.ego_car[0] = np.clip(self.ego_car[0], a_min=-1, a_max=2)
        ego_car_obs = self.ego_car.copy()
        other_car_obs = np.concatenate([self.first_car[:-1],
                                        self.second_car[:-1],
                                        self.third_car[:-1],
                                        self.fourth_car[:-1]])

        h_dir_first = (self.first_car[0:1] - self.ego_car[0:1]).astype(np.float)

        h_dir_second = (self.second_car[0:1] - self.ego_car[0:1]).astype(np.float)

        h_dir_third = (self.third_car[0:1] - self.ego_car[0:1]).astype(np.float)

        h_dir_fourth = (self.fourth_car[0:1] - self.ego_car[0:1]).astype(np.float)

        other_car_text = np.concatenate([h_dir_first, self.first_car[1:],
                                         h_dir_second, self.second_car[1:],
                                         h_dir_third, self.third_car[1:],
                                         h_dir_fourth, self.fourth_car[1:]])

        collision = False

        if self.ego_car[0] == self.first_car[0] and self.first_car[1] * self.first_car_pre[1] <= 0.0:
            collision = True
        if self.ego_car[0] == self.second_car[0] and self.second_car[1] * self.second_car_pre[1] <= 0.0:
            collision = True
        if self.ego_car[0] == self.third_car[0] and self.third_car[1] * self.third_car_pre[1] <= 0.0:
            collision = True
        if self.ego_car[0] == self.fourth_car[0] and self.fourth_car[1] * self.fourth_car_pre[1] <= 0.0:
            collision = True

        done = collision
        # print(action)

        reward = 0.0
        if 0 <= self.ego_car[0] <= 1:
            lane_reward = 0
        else:
            lane_reward = -5
            # print('hit wall')
            done = True

        if action[1] == 1.0 or action[2] == 1.0:
            lane_reward -= 1.0

        if 0. < self.first_car[1] < 0.4 and self.ego_car[0] == 0:
            proximity_reward = -2
            # proximity_reward = -5.0 * (0.5 - self.first_car[1]) / 0.5
        elif 0. < self.second_car[1] < 0.4 and self.ego_car[0] == 1:
            proximity_reward = -2
            # proximity_reward = -5.0 * (0.5 - self.second_car[1]) / 0.5
        elif -0.4 < self.third_car[1] < 0 and self.ego_car[0] == 0:
            proximity_reward = -2
            # proximity_reward = -5.0 * (0.5 - self.third_car[1]) / 0.5
        elif -0.4 < self.fourth_car[1] < 0 and self.ego_car[0] == 1:
            proximity_reward = -2
            # proximity_reward = -5.0 * (0.5 - self.fourth_car[1]) / 0.5
        else:
            proximity_reward = 0.

        if collision:
            collision_reward = -5
            # print('hit car')
        else:
            collision_reward = 0.0

        speed_reward = self.ego_car[1] * 0.2

        reward = lane_reward + collision_reward + speed_reward + proximity_reward
        
        return ego_car_obs, other_car_obs, other_car_text, reward, done

    def get_prescribed_reward(self, ego_car, first_car, second_car, third_car, fourth_car, collision, action):
        reward = 0.0
        if 0 <= ego_car[0] <= 1:
            lane_reward = 0
        else:
            lane_reward = -5
            # print('hit wall')
            done = True

        if action == 1.0 or action == 2.0:
            lane_reward -= 1.0

        if 0. < first_car[1] < 0.4 and ego_car[0] == 0:
            proximity_reward = -2
        elif 0. < second_car[1] < 0.4 and ego_car[0] == 1:
            proximity_reward = -2
        elif -0.4 < third_car[1] < 0 and ego_car[0] == 0:
            proximity_reward = -2
        elif -0.4 < fourth_car[1] < 0 and ego_car[0] == 1:
            proximity_reward = -2
        else:
            proximity_reward = 0.

        if collision:
            collision_reward = -5
        else:
            collision_reward = 0.0

        speed_reward = ego_car[1] * 0.2

        reward = lane_reward + collision_reward + speed_reward + proximity_reward

        return reward

    def agent_planning(self, ego_car_obs, other_car_obs, other_car_text, maxent_theta=None):
        # print('plan')
        sample_size = 100
        horizon = 3
        discount = 0.9
        action_sample = np.random.randint(low=0, high=3, size=(horizon, sample_size, 1))

        init_ego_car_obs = ego_car_obs.copy()
        init_other_car_obs = other_car_obs.copy()
        init_other_car_text = other_car_text.copy()

        reward_list = []
        for i in range(sample_size):
            reward_all = 0
            ego_car_obs = init_ego_car_obs.copy()
            other_car_obs = init_other_car_obs.copy()
            other_car_text = init_other_car_text.copy()
            # print(other_car_obs[1::2])
            for t in range(horizon):
                ego_car_obs, other_car_obs, other_car_text, reward, done = self.env_prediction(ego_car_obs,
                                                                                               other_car_obs,
                                                                                               other_car_text,
                                                                                               action_sample[t, i],
                                                                                               maxent_theta)
                reward_all += reward * discount ** t
                # print(other_car_obs[1::2], reward)
            reward_list += [reward_all]
        reward_np = np.stack(reward_list, axis=0)
        opti_id = np.argmax(reward_np, axis=0)
        opti_action = action_sample[:, opti_id]
        return opti_action

    def expert_policy(self):
        # check if there is a car within specific range
        ego_car_speed = self.ego_car[1]

        # action (1,0,0):stay, (0,1,0):left, (0,0,1):right
        expert_control = np.zeros(shape=(3,))
        collision_flag = False
        if self.ego_car[0] == self.first_car[0] and self.first_car[1] < -(self.first_car[2] - ego_car_speed):
            collision_flag = True
        if self.ego_car[0] == self.second_car[0] and self.second_car[1] < -(self.second_car[2] - ego_car_speed):
            collision_flag = True
        if self.ego_car[0] == self.third_car[0] and -self.third_car[1] < (self.third_car[2] - ego_car_speed):
            collision_flag = True
        if self.ego_car[0] == self.fourth_car[0] and -self.fourth_car[1] < (self.fourth_car[2] - ego_car_speed):
            collision_flag = True

        if collision_flag:
            if self.ego_car[0] == 0:
                # if ego car is at the left lane, turn right
                expert_control[2] = 1
            else:
                # if ego car is at the right lane, turn left
                expert_control[1] = 1
        else:
            # if it is safe, stay
            expert_control[0] = 1

        return expert_control

    def expert_policy_norm(self):
        # check if there is a car within specific range
        ego_car_speed = self.ego_car[1]

        # action (1,0,0):stay, (0,1,0):left, (0,0,1):right
        expert_control = np.zeros(shape=(3,))
        collision_flag = False
        if self.ego_car[0] == self.first_car[0] and self.first_car[1] < -(self.first_car[2] - ego_car_speed):
            collision_flag = True
        if self.ego_car[0] == self.second_car[0] and self.second_car[1] < -(self.second_car[2] - ego_car_speed):
            collision_flag = True
        if self.ego_car[0] == self.third_car[0] and -self.third_car[1] < (self.third_car[2] - ego_car_speed):
            collision_flag = True
        if self.ego_car[0] == self.fourth_car[0] and -self.fourth_car[1] < (self.fourth_car[2] - ego_car_speed):
            collision_flag = True

        if collision_flag:
            if self.ego_car[0] == 0:
                # if ego car is at the left lane, turn right
                expert_control[2] = 1
            else:
                # if ego car is at the right lane, turn left
                expert_control[1] = 1
        else:
            # if it is at the left lane, turn right, otherwise, stay
            if self.ego_car[0] == 0:
                expert_control[2] = 1
            else:
                expert_control[0] = 1

        return expert_control

    def expert_policy_perceptual_bias(self, other_car_obs_mask, other_car_text_mask):
        # other_car_obs_mask/other_car_text_mask (first, second ,third, fourth)
        # check if there is a car within specific range
        ego_car_speed = self.ego_car[1]

        # action (1,0,0):stay, (0,1,0):left, (0,0,1):right
        expert_control = np.zeros(shape=(3,))
        collision_flag = False
        if self.ego_car[0] == self.first_car[0] and self.first_car[1] < -(self.first_car[2] - ego_car_speed):
            if other_car_text_mask[0] == 1 or other_car_obs_mask[0] == 1:
                collision_flag = True
            else:
                collision_flag = False
        if self.ego_car[0] == self.second_car[0] and self.second_car[1] < -(self.second_car[2] - ego_car_speed):
            if other_car_text_mask[1] == 1 or other_car_obs_mask[1] == 1:
                collision_flag = True
            else:
                collision_flag = False
        if self.ego_car[0] == self.third_car[0] and -self.third_car[1] < (self.third_car[2] - ego_car_speed):
            if other_car_text_mask[2] == 1:
                collision_flag = True
            else:
                collision_flag = False
        if self.ego_car[0] == self.fourth_car[0] and -self.fourth_car[1] < (self.fourth_car[2] - ego_car_speed):
            if other_car_text_mask[3] == 1:
                collision_flag = True
            else:
                collision_flag = False

        if collision_flag:
            if self.ego_car[0] == 0:
                # if ego car is at the left lane, turn right
                expert_control[2] = 1
            else:
                # if ego car is at the right lane, turn left
                expert_control[1] = 1
        else:
            # if it is at the left lane, turn right, otherwise, stay
            if self.ego_car[0] == 0:
                expert_control[2] = 1
            else:
                expert_control[0] = 1

        return expert_control

class FourCarCommEnv:
    def __init__(self, robot_model, human_model, perceptual_mask, policy_filter, method):
        self._state = None
        self._action = None
        self._robot_model = robot_model
        self._human_model = human_model

        self._human_obs = None

        self.model_device = 'cuda:1'

        self._env = FourCarEnv()

        self.visual_p = None
        self.text_p = None

        self.perceptual_mask = None
        self.policy_filter = None

        # self.maxent_theta = np.load('./maxent/irl_theta.npy')
        self.maxent_theta = np.array([-0.075, 0.529, 0.489, -0.535, -0.43, -0.072])

        self.method = method

    def reset(self):
        self._ego_car_obs, self._other_car_obs, self._other_car_text, reward, done = self._env.reset()

        ego_car_obs = torch.as_tensor(self._ego_car_obs).float().to(self.model_device)
        other_car_obs = torch.as_tensor(self._other_car_obs).float().to(self.model_device)
        other_car_text = torch.as_tensor(self._other_car_text).float().to(self.model_device)

        self._env_latent_state = self._robot_model.get_state_representation(ego_car_obs, other_car_obs, other_car_text,
                                                                            action=None, pre_state=None,
                                                                            ego_car_mask=None, other_car_mask=None)

        self._agent_latent_state = self._human_model.get_state_representation(ego_car_obs, other_car_obs,
                                                                              other_car_text,
                                                                              action=None, pre_state=None,
                                                                              ego_car_mask=None,
                                                                              other_car_mask=torch.zeros_like(
                                                                                  other_car_obs).to(self.model_device),
                                                                              other_car_text_mask=torch.zeros_like(
                                                                                  other_car_text).to(self.model_device))

        # self._action, _ = self._model.policy(self._agent_latent_state)
        first_car = self._env.normalizer(np.array([0., 7, 1.]))
        second_car = self._env.normalizer(np.array([1., 7, 1.]))
        third_car = self._env.normalizer(np.array([0., -7, 1.]))
        fourth_car = self._env.normalizer(np.array([1., -7, 1.]))

        self._human_obs = np.concatenate([first_car, second_car, third_car, fourth_car], axis=-1)

        human_ego_car_obs = ego_car_obs.detach().cpu().numpy().copy()
        human_other_car_obs, human_other_car_text = self.generate_obs_from_human_obs()
        opti_action = self._env.agent_planning(human_ego_car_obs,
                                               human_other_car_obs,
                                               human_other_car_text,
                                               self.maxent_theta)
        opti_action = opti_action[0]
        # print(opti_action)
        action = np.zeros(3)
        action[opti_action.item()] = 1.0
        self._action = action

        self._state = torch.cat([self._agent_latent_state, self._env_latent_state], dim=-1)

        return self._ego_car_obs, self._other_car_obs, self._other_car_text, self._action

    def update_human_obs(self, other_car_mask, other_car_text_mask, other_car_obs, other_car_text):
        # from visual obs update
        if other_car_mask[0] == 1:
            self._human_obs[0:2] = other_car_obs[0:2]
            self._human_obs[2] = other_car_text[2]
        if other_car_mask[2] == 1:
            self._human_obs[3:5] = other_car_obs[2:4]
            self._human_obs[5] = other_car_text[5]
        if other_car_mask[4] == 1:
            self._human_obs[6:8] = other_car_obs[4:6]
            self._human_obs[8] = other_car_text[8]
        if other_car_mask[6] == 1:
            self._human_obs[9:11] = other_car_obs[6:8]
            self._human_obs[11] = other_car_text[11]

        # from text obs update
        if other_car_text_mask[0] == 1:
            self._human_obs[0:2] = other_car_obs[0:2]
            self._human_obs[2] = other_car_text[2]
        if other_car_text_mask[3] == 1:
            self._human_obs[3:5] = other_car_obs[2:4]
            self._human_obs[5] = other_car_text[5]
        if other_car_text_mask[6] == 1:
            self._human_obs[6:8] = other_car_obs[4:6]
            self._human_obs[8] = other_car_text[8]
        if other_car_text_mask[9] == 1:
            self._human_obs[9:11] = other_car_obs[6:8]
            self._human_obs[11] = other_car_text[11]

        if 0 <= other_car_obs[1] <= 0.2:
            self._human_obs[0:2] = other_car_obs[0:2]
            self._human_obs[2] = other_car_text[2]
        if 0 <= other_car_obs[3] <= 0.2:
            self._human_obs[3:5] = other_car_obs[2:4]
            self._human_obs[5] = other_car_text[5]
        if 0 <= other_car_obs[5] <= 0.2:
            self._human_obs[6:8] = other_car_obs[4:6]
            self._human_obs[8] = other_car_text[8]
        if 0 <= other_car_obs[7] <= 0.2:
            self._human_obs[9:11] = other_car_obs[6:8]
            self._human_obs[11] = other_car_text[11]

    def generate_obs_from_human_obs(self):
        _human_obs = self._human_obs.copy()
        other_car_obs = np.concatenate([_human_obs[0:2], _human_obs[3:5], _human_obs[6:8], _human_obs[9:11]], axis=-1)
        other_car_text = _human_obs
        other_car_text[1::3] = np.abs(other_car_text[1::3])

        return other_car_obs, other_car_text

    def forward(self, action):
        other_car_mask = torch.as_tensor(action[..., :8]).to(self.model_device).to(torch.float)
        other_car_text_mask = torch.as_tensor(action[..., 8:]).to(self.model_device).to(torch.float)

        ego_car_obs = torch.as_tensor(self._ego_car_obs).float().to(self.model_device).to(torch.float)
        other_car_obs = torch.as_tensor(self._other_car_obs).float().to(self.model_device)
        other_car_text = torch.as_tensor(self._other_car_text).float().to(self.model_device)

        action_torch = torch.as_tensor(self._action).float().to(self.model_device)

        limited_view_mask = torch.ones_like(other_car_mask).to(other_car_mask.device)
        limited_view_mask[..., 4:] = 0.0

        if self.perceptual_mask is not None:
            visual_obs_mask = self.perceptual_mask.get_obs_mask(other_car_obs[..., 1::2])
            visual_obs_mask = [visual_obs_mask[..., k:k + 1].repeat(1, 1, 2) for k in range(4)]

            visual_obs_mask = torch.cat(visual_obs_mask, dim=-1)  # * 0.0

            other_car_mask = (visual_obs_mask + other_car_mask * limited_view_mask > 0.5).type(torch.float)
        else:
            if self.method == 'bc':
                other_car_mask = (generate_cheating_obs_mask(
                    other_car_obs[..., 1::2]) + other_car_mask * limited_view_mask > 0.5).type(torch.float)
            else:
                other_car_mask = (other_car_mask * limited_view_mask > 0.5).type(torch.float)

        self._env_latent_state = self._robot_model.get_state_representation(ego_car_obs, other_car_obs, other_car_text,
                                                                            action=action_torch,
                                                                            pre_state=self._env_latent_state,
                                                                            ego_car_mask=None, other_car_mask=None,
                                                                            other_car_text_mask=None)

        self._agent_latent_state = self._human_model.get_state_representation(ego_car_obs, other_car_obs,
                                                                              other_car_text,
                                                                              action=action_torch,
                                                                              pre_state=self._agent_latent_state,
                                                                              ego_car_mask=torch.ones_like(
                                                                                  ego_car_obs).to(
                                                                                  self.model_device),
                                                                              other_car_mask=other_car_mask,
                                                                              other_car_text_mask=other_car_text_mask)

        # print(other_car_mask, other_car_text_mask)
        self.update_human_obs(other_car_mask.detach().cpu().numpy(),
                              other_car_text_mask.detach().cpu().numpy(),
                              other_car_obs.detach().cpu().numpy(),
                              other_car_text.detach().cpu().numpy())

        human_ego_car_obs = ego_car_obs.detach().cpu().numpy().copy()
        human_other_car_obs, human_other_car_text = self.generate_obs_from_human_obs()
        opti_action = self._env.agent_planning(human_ego_car_obs,
                                               human_other_car_obs,
                                               human_other_car_text,
                                               self.maxent_theta)
        opti_action = opti_action[0]
        # print(opti_action)
        action = np.zeros(3)
        action[opti_action.item()] = 1.0
        self._action = action

        # self._action, _ = self._model.policy(self._agent_latent_state)

        self._ego_car_obs, self._other_car_obs, self._other_car_text, reward, done = self._env.forward(
            self._action)

        self._state = torch.cat([self._agent_latent_state, self._env_latent_state], dim=-1)

        return self._ego_car_obs, self._other_car_obs, self._other_car_text, self._action, reward, done

    def adaptive_random_shooting(self):
        with torch.no_grad():
            sample_size = 500
            max_itr = 4
            # lidar_p = np.array([0.25, 0.25, 0.25, 0.25, 0.25, 0.25])
            # text_p = np.array([0.25, 0.25, 0.25, 0.25, 0.25, 0.25])
            time_length = 4
            frame_skip = 1

            text_p_tau = 0.25
            visual_p_tau = 0.25

            mix_rate = 0.75
            visual_p = np.ones(shape=(time_length * frame_skip, 4)) * 0.35
            text_p = np.ones(shape=(time_length * frame_skip, 4)) * 0.35
            if self.visual_p is not None:
                visual_p[:-1] = (1 - mix_rate) * visual_p[:-1] + mix_rate * self.visual_p[1:]
            if self.text_p is not None:
                text_p[:-1] = (1 - mix_rate) * text_p[:-1] + mix_rate * self.text_p[1:]

            update_rate = 0.8

            for i in range(max_itr):
                _, _, _, accumulated_q_value, visual_action_mask, text_action_mask = self.random_shooting(visual_p,
                                                                                                          text_p,
                                                                                                          sample_size=sample_size,
                                                                                                          time_length=time_length,
                                                                                                          frame_skip=frame_skip)
                top_k_ids = torch.topk(accumulated_q_value, dim=0, k=int(100)).indices.squeeze(dim=-1).to(
                    self.model_device)
                visual_action_mask = torch.index_select(visual_action_mask, dim=1,
                                                        index=top_k_ids).detach().cpu().numpy()
                text_action_mask = torch.index_select(text_action_mask, dim=1, index=top_k_ids).detach().cpu().numpy()

                update_visual_p = visual_action_mask.mean(axis=1)
                update_text_p = text_action_mask.mean(axis=1)

                visual_p = (1 - update_rate) * visual_p + update_rate * update_visual_p[..., 0::2]
                text_p = (1 - update_rate) * text_p + update_rate * update_text_p[..., 0::3]

            # print(text_p[0])

            self.visual_p = visual_p.copy()
            self.text_p = text_p.copy()
            # print(max_id)
            visual_p_torch = torch.as_tensor(visual_p).to(self.model_device)
            text_p_torch = torch.as_tensor(text_p).to(self.model_device)

            opti_visual_action_mask = (visual_p_torch[0] > visual_p_tau).type(torch.float)
            opti_text_action_mask = (text_p_torch[0] > text_p_tau).type(torch.float)
            # print(opti_text_action_mask.detach().cpu().numpy(), opti_visual_action_mask.detach().cpu().numpy())

            opti_visual_action_mask = torch.cat([opti_visual_action_mask[0:1].repeat(2),
                                                 opti_visual_action_mask[1:2].repeat(2),
                                                 opti_visual_action_mask[2:3].repeat(2),
                                                 opti_visual_action_mask[3:4].repeat(2)])

            opti_text_action_mask = torch.cat([opti_text_action_mask[0:1].repeat(3),
                                               opti_text_action_mask[1:2].repeat(3),
                                               opti_text_action_mask[2:3].repeat(3),
                                               opti_text_action_mask[3:4].repeat(3)])

            opti_action_mask = torch.cat([opti_visual_action_mask, opti_text_action_mask], dim=-1)
            return opti_action_mask, opti_visual_action_mask, opti_text_action_mask, \
                   (visual_p.copy()[0] > visual_p_tau).astype(np.float), (text_p.copy()[0] > text_p_tau).astype(
                np.float)

    def random_shooting(self, visual_p=None, text_p=None, sample_size=12000, time_length=3, frame_skip=2):
        sample_size = sample_size
        time_length = time_length
        frame_skip = frame_skip

        if visual_p is None:
            visual_p = np.ones(shape=(time_length, 6)) * 0.5

        if text_p is None:
            text_p = np.ones(shape=(time_length, 6)) * 0.5
        # if self.lidar_action_mask_list is None:
        self.visual_action_mask_list = []
        for i in range(time_length):
            self.visual_action_mask = []
            self.visual_action_mask += [
                (torch.rand(size=(1, sample_size, 1)) < visual_p[i, 0].item()).type(torch.float).repeat(frame_skip, 1,
                                                                                                        2)]
            self.visual_action_mask += [
                (torch.rand(size=(1, sample_size, 1)) < visual_p[i, 1].item()).type(torch.float).repeat(frame_skip, 1,
                                                                                                        2)]
            self.visual_action_mask += [
                (torch.rand(size=(1, sample_size, 1)) < visual_p[i, 2].item()).type(torch.float).repeat(frame_skip, 1,
                                                                                                        2)]
            self.visual_action_mask += [
                (torch.rand(size=(1, sample_size, 1)) < visual_p[i, 3].item()).type(torch.float).repeat(frame_skip, 1,
                                                                                                        2)]
            self.visual_action_mask = torch.cat(self.visual_action_mask, dim=-1)
            self.visual_action_mask_list += [self.visual_action_mask]
        self.visual_action_mask_list = torch.cat(self.visual_action_mask_list, dim=0).to(self.model_device)

        # if self.text_action_mask_list is None:
        self.text_action_mask_list = []
        for i in range(time_length):
            self.text_action_mask = []
            self.text_action_mask += [
                (torch.rand(size=(1, sample_size, 1)) < text_p[i, 0].item()).type(torch.float).repeat(frame_skip, 1, 3)]
            self.text_action_mask += [
                (torch.rand(size=(1, sample_size, 1)) < text_p[i, 1].item()).type(torch.float).repeat(frame_skip, 1, 3)]
            self.text_action_mask += [
                (torch.rand(size=(1, sample_size, 1)) < text_p[i, 2].item()).type(torch.float).repeat(frame_skip, 1, 3)]
            self.text_action_mask += [
                (torch.rand(size=(1, sample_size, 1)) < text_p[i, 3].item()).type(torch.float).repeat(frame_skip, 1, 3)]

            self.text_action_mask = torch.cat(self.text_action_mask, dim=-1)
            self.text_action_mask_list += [self.text_action_mask]
        self.text_action_mask_list = torch.cat(self.text_action_mask_list, dim=0).to(self.model_device)

        current_state = torch.tensor(self._state).unsqueeze(dim=0).unsqueeze(dim=0).to(self.model_device)
        current_state = current_state.repeat(1, sample_size, 1)

        visual_action_mask = self.visual_action_mask_list
        text_action_mask = self.text_action_mask_list
        accumulated_q_value = self.prediction_mask(current_state, visual_action_mask, text_action_mask)
        # print(accumulated_q_value.size())
        max_id = torch.argmax(accumulated_q_value, dim=0).item()
        # print(max_id)

        opti_visual_action_mask = self.visual_action_mask_list[0, max_id]
        opti_text_action_mask = self.text_action_mask_list[0, max_id]
        opti_action_mask = torch.cat([opti_visual_action_mask, opti_text_action_mask], dim=-1)
        return opti_action_mask, opti_visual_action_mask, opti_text_action_mask, \
               accumulated_q_value, self.visual_action_mask_list, self.text_action_mask_list

    def prediction_mask(self, current_state, visual_action_mask, text_action_mask):
        # current_state (1, batch, 2 * latent_size)
        # action_mask (time, batch, action (19+5))
        current_agent_state = current_state[..., :self._human_model.latent_size]
        current_env_state = current_state[..., self._robot_model.latent_size:]

        time_length, batch_size, _ = visual_action_mask.size()

        next_env_state = current_env_state
        next_agent_state = current_agent_state

        accumulated_q_value = 0.0
        discount_factor = 0.9

        decoder_ego_car = self._robot_model.autoencoder.decoder_ego_car
        decoder_other_car = self._robot_model.autoencoder.decoder_other_car
        decoder_other_car_text = self._robot_model.autoencoder.decoder_other_car_text
        q_values = [[]] * time_length
        for t in range(time_length):
            # given z_t, a_m_t, predict z_t+1
            # predict next env state
            next_env_state, agent_action = self.prediction(next_env_state, next_agent_state)

            # update human state
            next_ego_obs = decoder_ego_car(next_env_state).mean
            next_other_obs = decoder_other_car(next_env_state).mean
            next_other_text_obs = decoder_other_car_text(next_env_state).mean

            limited_view_mask = torch.ones_like(visual_action_mask[t:t + 1]).to(visual_action_mask.device)
            limited_view_mask[..., 4:] = 0.0

            if self.perceptual_mask is not None:
                visual_obs_mask = self.perceptual_mask.get_obs_mask(next_other_obs[..., 1::2])
                visual_obs_mask = [visual_obs_mask[:, :, k:k + 1].repeat(1, 1, 2) for k in range(4)]

                visual_obs_mask = torch.cat(visual_obs_mask, dim=-1)  # * 0.0

                other_car_mask = (visual_obs_mask + visual_action_mask[t:t + 1] * limited_view_mask > 0.5).type(
                    torch.float)
            else:
                other_car_mask = visual_action_mask[t:t + 1] * limited_view_mask
            next_agent_state = self._human_model.get_state_representation(next_ego_obs, next_other_obs,
                                                                          next_other_text_obs,
                                                                          action=agent_action,
                                                                          pre_state=next_agent_state,
                                                                          ego_car_mask=torch.ones_like(
                                                                              next_ego_obs).to(
                                                                              self.model_device),
                                                                          other_car_mask=other_car_mask,
                                                                          other_car_text_mask=text_action_mask[t:t + 1])

            # get Q value
            mask_cost = (torch.sum(visual_action_mask[t:t + 1], dim=-1, keepdim=True) / 2.0).pow(2) * 0.01 \
                        + (torch.sum(text_action_mask[t:t + 1], dim=-1, keepdim=True) / 3.0).pow(2) * 0.03

            accumulated_q_value += discount_factor ** t * (self.get_reward(next_env_state) * 10.0 - mask_cost)
        return accumulated_q_value.squeeze(dim=0)  # (batch, 1)

    def get_q_value(self, latent_state: torch.Tensor):
        action, _ = self._human_model.policy(latent_state)
        combined_feat = torch.cat([latent_state, action], dim=-1)
        return self._robot_model.qf1_model(combined_feat).mean

    def get_reward(self, latent_state: torch.Tensor):
        return self._robot_model.reward_model(latent_state).mean

    def prediction(self, current_env_state, current_agent_state):
        # human take an action based current agent state
        # action, action_agent_dist = self._human_model.policy(current_agent_state)
        q = []
        T, B, _ = current_agent_state.size()
        a1 = torch.zeros(size=(T, B, 3)).to(self.model_device).type(torch.float)
        a1[..., 0] = 1
        a2 = torch.zeros(size=(T, B, 3)).to(self.model_device).type(torch.float)
        a2[..., 1] = 1
        a3 = torch.zeros(size=(T, B, 3)).to(self.model_device).type(torch.float)
        a3[..., 2] = 1

        q.append(self._human_model.qf1_model(torch.cat([current_agent_state, a1], dim=-1)).mean)
        q.append(self._human_model.qf1_model(torch.cat([current_agent_state, a2], dim=-1)).mean)
        q.append(self._human_model.qf1_model(torch.cat([current_agent_state, a3], dim=-1)).mean)

        if self.policy_filter is not None:
            q[0] += self.policy_filter(current_env_state)[..., 0:1] * 0.1
            q[1] += self.policy_filter(current_env_state)[..., 1:2] * 0.1
            q[2] += self.policy_filter(current_env_state)[..., 2:3] * 0.1

        policy_dist = torch.distributions.OneHotCategorical(logits=torch.cat([q[0], q[1], q[2]], dim=-1))
        action = policy_dist.sample()

        # forward environment based on current env state
        next_state_dist = self._robot_model.autoencoder.transition(torch.cat([current_env_state, action], dim=-1))
        return next_state_dist.mean, action

    @property
    def env(self):
        return self._env
        
if __name__ == '__main__':
    ego_car_obs = np.array([0, 1])
    first_car_obs = np.array([0, -2])
    second_car_obs = np.array([1, 5])
    third_car_obs = np.array([0, -3])
    fourth_car_obs = np.array([1, 5])
    action = np.array([0, 1, 0])

    state_id = encode(ego_car_obs,
                      np.concatenate([first_car_obs, second_car_obs, third_car_obs, fourth_car_obs], axis=-1),
                      action)

    ego_car_obs, first_car_obs, second_car_obs, third_car_obs, fourth_car_obs, action = decode(state_id)

    env = FourCarEnv()
    ego_car_obs, other_car_obs, other_car_text, reward, done = env.reset()
    print(ego_car_obs, other_car_obs, other_car_text, reward, done)
    ego_car_obs, other_car_obs, other_car_text, reward, done = env.forward([1, 0, 0])
    print(ego_car_obs, other_car_obs, other_car_text, reward, done)
