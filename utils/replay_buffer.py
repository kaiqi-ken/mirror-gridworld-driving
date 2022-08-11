import torch
import numpy as np
import random
from torch.utils.data import Dataset

import numpy as np
from four_car_env import encode


class DataRecorder:
    def __init__(self, path):
        self.path = path
        # allocate memory for buffers
        self.collisions = []

        self.human_actions = []

        self.obs = []
        self.text = []

        self.first_car = []
        self.second_car = []
        self.third_car = []
        self.fourth_car = []
        self.episode_num = []

        # add some raw data buffers

    def add_case_setup(self, first_car, second_car, third_car, fourth_car, episode_num):
        self.first_car += [first_car]
        self.second_car += [second_car]
        self.episode_num += [episode_num]
        self.third_car += [third_car]
        self.fourth_car += [fourth_car]

    def add_data(self, collision,
                 human_action,
                 obs,
                 text):
        # add traj data to buffer
        if collision is not None:
            self.collisions += [collision]
        if human_action is not None:
            self.human_actions += [human_action]
        if obs is not None:
            self.obs += [obs]
        if text is not None:
            self.text += [text]

    def save_data(self, case: str):
        # save traj of each episode
        first_car_np = np.array(self.first_car)
        second_car_np = np.array(self.second_car)
        episode_num_np = np.array(self.episode_num)
        third_car_np = np.array(self.third_car)
        fourth_car_np = np.array(self.fourth_car)

        obs_np = np.array(self.obs)
        text_np = np.array(self.text)
        collisions_np = np.array(self.collisions)

        human_actions_np = np.array(self.human_actions)
        file_name = f'traj_{case}.npz'
        np.savez(f'{self.path}/{file_name}',
                 first_car=first_car_np,
                 second_car=second_car_np,
                 third_car=third_car_np,
                 fourth_car=fourth_car_np,
                 episode_num=episode_num_np,
                 obs=obs_np,
                 text=text_np,
                 collisions=collisions_np,
                 human_actions=human_actions_np, )

        print(f'Save file {file_name} to {self.path}')

    def clear_buffer(self):
        self.first_car.clear()
        self.second_car.clear()
        self.third_car.clear()
        self.fourth_car.clear()
        self.episode_num.clear()

        self.obs.clear()
        self.text.clear()
        self.collisions.clear()
        self.human_actions.clear()


class HumanDataSetMaxEnt():
    def __init__(self, device='cuda:0'):
        self.obs_train = None
        self.text_train = None
        self.human_action_train = None
        self.done_train = None

        self.device = device
        self.train = True

        # self.full_traj = True

    def load_train_data(self, path_list):
        for path in path_list:
            data_npz = np.load(f'{path}')
            _obs_train = data_npz['obs']
            _text_train = data_npz['text']
            _human_action_train = data_npz['human_actions']

            while True:
                if _obs_train.shape[0] >= 10:
                    break
                else:
                    _obs_train = np.concatenate([_obs_train, _obs_train[-1:]], axis=0)
                    _text_train = np.concatenate([_text_train, _text_train[-1:]], axis=0)
                    _human_action_train = np.concatenate([_human_action_train, _human_action_train[-1:]], axis=0)

            if self.obs_train is None:
                self.obs_train = _obs_train
                self.text_train = _text_train
                self.human_action_train = _human_action_train
                # self.done_train = data_npz['done']
            else:
                self.obs_train = np.concatenate([self.obs_train, _obs_train], axis=0)
                self.text_train = np.concatenate([self.text_train, _text_train], axis=0)
                self.human_action_train = np.concatenate([self.human_action_train, _human_action_train], axis=0)
                # self.done_train = np.concatenate([self.done_train, data_npz['done']], axis=0)

            # (batch, time, id)
            self.state_idxs = []
            for i in range(int(self.obs_train.shape[0])):
                ego_obs = self.obs_train[i, :2]
                other_obs = self.obs_train[i, 2:] * 10
                action = self.human_action_train[i]
                state_idx = encode(ego_obs, other_obs, action)
                self.state_idxs += [state_idx]
            self.state_idxs = np.stack(self.state_idxs, axis=0).reshape(-1, 10, 1)

class HumanDataSetMaxEntCompress():
    def __init__(self, device='cuda:0'):
        self.obs_train = None
        self.text_train = None
        self.human_action_train = None
        self.done_train = None

        self.device = device
        self.train = True

        # self.full_traj = True

    def load_train_data(self, path_list):
        for path in path_list:
            data_npz = np.load(f'{path}')
            _obs_train = data_npz['obs']
            _text_train = data_npz['text']
            _human_action_train = data_npz['human_actions']

            while True:
                if _obs_train.shape[0] >= 10:
                    break
                else:
                    _obs_train = np.concatenate([_obs_train, _obs_train[-1:]], axis=0)
                    _text_train = np.concatenate([_text_train, _text_train[-1:]], axis=0)
                    _human_action_train = np.concatenate([_human_action_train, _human_action_train[-1:]], axis=0)

            if self.obs_train is None:
                self.obs_train = _obs_train
                self.text_train = _text_train
                self.human_action_train = _human_action_train
                # self.done_train = data_npz['done']
            else:
                self.obs_train = np.concatenate([self.obs_train, _obs_train], axis=0)
                self.text_train = np.concatenate([self.text_train, _text_train], axis=0)
                self.human_action_train = np.concatenate([self.human_action_train, _human_action_train], axis=0)
                # self.done_train = np.concatenate([self.done_train, data_npz['done']], axis=0)

            # (batch, time, id)
            self.feature = []
            for i in range(int(self.obs_train.shape[0])):
                ego_obs = self.obs_train[i, 0:1]
                if 0 <= ego_obs <= 1:
                    ego_obs = 0
                else:
                    ego_obs = 1
                ego_obs = np.array([ego_obs])

                other_obs = self.obs_train[i, 3::2]
                other_lanes = self.obs_train[i, 2::2]
                if other_lanes[0] != self.obs_train[i, 0] and ego_obs == 0:
                    other_obs[0] = 0.9
                if other_lanes[1] != self.obs_train[i, 0] and ego_obs == 0:
                    other_obs[1] = 0.9
                if other_lanes[2] != self.obs_train[i, 0] and ego_obs == 0:
                    other_obs[2] = -0.9
                if other_lanes[3] != self.obs_train[i, 0] and ego_obs == 0:
                    other_obs[3] = -0.9

                if ego_obs == 1:
                    other_obs[:] = 0


                action = np.argmax(self.human_action_train[i])
                if action == 0:
                    action = 0
                else:
                    action = 1

                action = np.array([action])
                self.feature += [np.concatenate([ego_obs, other_obs, action])]

            self.feature = np.stack(self.feature, axis=0).reshape(-1, 10, 6)




class HumanDataSet(Dataset):
    def __init__(self, device='cuda:0'):
        self.obs_train = None
        self.text_train = None
        self.human_action_train = None
        self.done_train = None
        # self.reward_buffer = None

        self.obs_test = None
        self.text_test = None
        self.human_action_test = None
        self.done_test = None
        # self.reward_buffer = None

        self.device = device
        self.train = True

        # self.full_traj = True

    def load_train_data(self, path_list):
        for path in path_list:
            data_npz = np.load(f'{path}')
            _obs_train = data_npz['obs']
            _text_train = data_npz['text']
            _human_action_train = data_npz['human_actions']

            while True:
                if _obs_train.shape[0] >= 10:
                    break
                else:
                    _obs_train = np.concatenate([_obs_train, _obs_train[-1:]], axis=0)
                    _text_train = np.concatenate([_text_train, _text_train[-1:]], axis=0)
                    _human_action_train = np.concatenate([_human_action_train, _human_action_train[-1:]], axis=0)

            if self.obs_train is None:
                self.obs_train = _obs_train
                self.text_train = _text_train
                self.human_action_train = _human_action_train
                # self.done_train = data_npz['done']
            else:
                self.obs_train = np.concatenate([self.obs_train, _obs_train], axis=0)
                self.text_train = np.concatenate([self.text_train, _text_train], axis=0)
                self.human_action_train = np.concatenate([self.human_action_train, _human_action_train], axis=0)
                # self.done_train = np.concatenate([self.done_train, data_npz['done']], axis=0)

    def load_test_data(self, path_list):
        for path in path_list:
            data_npz = np.load(f'{path}')
            data_npz = np.load(f'{path}')
            _obs_test = data_npz['obs']
            _text_test = data_npz['text']
            _human_action_test = data_npz['human_actions']

            while True:
                if _obs_test.shape[0] >= 10:
                    break
                else:
                    _obs_test = np.concatenate([_obs_test, _obs_test[-1:]], axis=0)
                    _text_test = np.concatenate([_text_test, _text_test[-1:]], axis=0)
                    _human_action_test = np.concatenate([_human_action_test, _human_action_test[-1:]], axis=0)
            if self.obs_test is None:
                self.obs_test = _obs_test
                self.text_test = _text_test
                self.human_action_test = _human_action_test
                # self.done_test = data_npz['done']
            else:
                self.obs_test = np.concatenate([self.obs_test, _obs_test], axis=0)
                self.text_test = np.concatenate([self.text_test, _text_test], axis=0)
                self.human_action_test = np.concatenate([self.human_action_test, _human_action_test], axis=0)
                # self.done_test = np.concatenate([self.done_test, data_npz['done']], axis=0)

    def __len__(self):
        return int(self.obs_train.shape[0] / 10)

    def __getitem__(self, i):
        if self.train:
            return {"obs": torch.as_tensor(self.obs_train[i * 10:(i + 1) * 10]),
                    "text": torch.as_tensor(self.text_train[i * 10:(i + 1) * 10]),
                    "human_action": torch.as_tensor(self.human_action_train[i * 10:(i + 1) * 10]),
                    # "done": torch.as_tensor(self.done_train[i])
                    }


class ReplayBuffer(object):
    """Buffer to store and replay environment transitions."""

    def __init__(self, obs_shape, text_shape, info_mask_obs_shape, infor_mask_text_shape,
                 action_shape, reward_shape, capacity, batch_size, device):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device
        # Initialize all the buffers
        self.obs_buffer = np.empty(shape=(capacity, *obs_shape), dtype=np.float32)
        self.text_buffer = np.empty(shape=(capacity, *text_shape), dtype=np.float32)
        self.info_mask_obs_buffer = np.empty(shape=(capacity, *info_mask_obs_shape), dtype=np.float32)
        self.info_mask_text_buffer = np.empty(shape=(capacity, *infor_mask_text_shape), dtype=np.float32)
        self.action_buffer = np.empty(shape=(capacity, *action_shape), dtype=np.float32)
        self.expert_buffer = np.empty(shape=(capacity, *action_shape), dtype=np.float32)
        self.reward_buffer = np.empty(shape=(capacity, *reward_shape), dtype=np.float32)
        self.done_buffer = np.empty(shape=(capacity, *reward_shape), dtype=np.float32)
        self.idx = 0

    def add(self, obs, text, infor_mask_obs, infor_mask_text, action, expert_action, reward, done):
        if self.idx < self.capacity:
            self.obs_buffer[self.idx] = obs
            self.text_buffer[self.idx] = text
            self.info_mask_obs_buffer[self.idx] = infor_mask_obs
            self.info_mask_text_buffer[self.idx] = infor_mask_text
            self.action_buffer[self.idx] = action
            self.expert_buffer[self.idx] = expert_action
            self.reward_buffer[self.idx] = reward
            self.done_buffer[self.idx] = done
            self.idx += 1
        else:
            self.obs_buffer = self.obs_buffer[1:]
            self.obs_buffer = np.append(self.obs_buffer,
                                        obs.reshape((1, obs.shape[0])),
                                        axis=0)
            self.text_buffer = self.text_buffer[1:]
            self.text_buffer = np.append(self.text_buffer,
                                         text.reshape((1, text.shape[0])),
                                         axis=0)
            self.info_mask_obs_buffer = self.info_mask_obs_buffer[1:]
            self.info_mask_obs_buffer = np.append(self.info_mask_obs_buffer,
                                                  infor_mask_obs.reshape((1, infor_mask_obs.shape[0])),
                                                  axis=0)
            self.info_mask_text_buffer = self.info_mask_text_buffer[1:]
            self.info_mask_text_buffer = np.append(self.info_mask_text_buffer,
                                                   infor_mask_text.reshape((1, infor_mask_text.shape[0])),
                                                   axis=0)
            self.action_buffer = self.action_buffer[1:]
            self.action_buffer = np.append(self.action_buffer,
                                           action.reshape((1, action.shape[0])),
                                           axis=0)
            self.expert_buffer = self.expert_buffer[1:]
            self.expert_buffer = np.append(self.expert_buffer,
                                           expert_action.reshape((1, expert_action.shape[0])),
                                           axis=0)
            self.reward_buffer = self.reward_buffer[1:]
            self.reward_buffer = np.append(self.reward_buffer,
                                           reward.reshape((1, reward.shape[0])),
                                           axis=0)
            self.done_buffer = self.done_buffer[1:]
            self.done_buffer = np.append(self.done_buffer,
                                         done.reshape((1, done.shape[0])),
                                         axis=0)

    def sample(self, time=30):
        idxs = np.random.randint(
            0, self.capacity - time + 1 if self.idx == self.capacity else self.idx - time + 1, size=self.batch_size)
        obses = torch.as_tensor(self.obs_buffer[idxs], device=self.device).unsqueeze(1)
        texts = torch.as_tensor(self.text_buffer[idxs], device=self.device).unsqueeze(1)
        info_masks_obs = torch.as_tensor(self.info_mask_obs_buffer[idxs], device=self.device).unsqueeze(1)
        info_masks_text = torch.as_tensor(self.info_mask_text_buffer[idxs], device=self.device).unsqueeze(1)
        actions = torch.as_tensor(self.action_buffer[idxs], device=self.device).unsqueeze(1)
        expert_actions = torch.as_tensor(self.expert_buffer[idxs], device=self.device).unsqueeze(1)
        rewards = torch.as_tensor(self.reward_buffer[idxs], device=self.device).unsqueeze(1)
        dones = torch.as_tensor(self.done_buffer[idxs], device=self.device).unsqueeze(1)

        for i in range(1, time):
            next_obses = torch.as_tensor(self.obs_buffer[idxs + i], device=self.device).unsqueeze(1)
            next_texts = torch.as_tensor(self.text_buffer[idxs + i], device=self.device).unsqueeze(1)
            next_info_masks_obs = torch.as_tensor(self.info_mask_obs_buffer[idxs + i], device=self.device).unsqueeze(1)
            next_info_masks_text = torch.as_tensor(self.info_mask_text_buffer[idxs + i], device=self.device).unsqueeze(
                1)
            next_actions = torch.as_tensor(self.action_buffer[idxs + i], device=self.device).unsqueeze(1)
            next_expert_actions = torch.as_tensor(self.expert_buffer[idxs + i], device=self.device).unsqueeze(1)
            next_rewards = torch.as_tensor(self.reward_buffer[idxs + i], device=self.device).unsqueeze(1)
            next_dones = torch.as_tensor(self.done_buffer[idxs + i], device=self.device).unsqueeze(1)
            obses = torch.cat((obses, next_obses), 1)
            texts = torch.cat((texts, next_texts), 1)
            info_masks_obs = torch.cat((info_masks_obs, next_info_masks_obs), 1)
            info_masks_text = torch.cat((info_masks_text, next_info_masks_text), 1)
            actions = torch.cat((actions, next_actions), 1)
            expert_actions = torch.cat((expert_actions, next_expert_actions), 1)
            rewards = torch.cat((rewards, next_rewards), 1)
            dones = torch.cat((dones, next_dones), 1)

        return obses, texts, info_masks_obs, info_masks_text, actions, expert_actions, rewards, dones

    def save(self, path=None, level=None):
        if path is None:
            raise NotImplementedError
        else:
            np.save(f'{path}/obs_buffer_{level}.npy', self.obs_buffer)
            np.save(f'{path}/text_buffer_{level}.npy', self.text_buffer)
            np.save(f'{path}/info_mask_obs_buffer_{level}.npy', self.info_mask_obs_buffer)
            np.save(f'{path}/info_mask_text_buffer_{level}.npy', self.info_mask_text_buffer)
            np.save(f'{path}/action_buffer_{level}.npy', self.action_buffer)
            np.save(f'{path}/expert_buffer_{level}.npy', self.expert_buffer)
            np.save(f'{path}/reward_buffer_{level}.npy', self.reward_buffer)
            np.save(f'{path}/done_buffer_{level}.npy', self.done_buffer)

    def load(self, path=None, level=None):
        self.obs_buffer = np.load(f'{path}/obs_buffer_{level}.npy')
        self.text_buffer = np.load(f'{path}/text_buffer_{level}.npy')
        self.info_mask_obs_buffer = np.load(f'{path}/info_mask_obs_buffer_{level}.npy')
        self.info_mask_text_buffer = np.load(f'{path}/info_mask_text_buffer_{level}.npy')
        self.action_buffer = np.load(f'{path}/action_buffer_{level}.npy')
        self.expert_buffer = np.load(f'{path}/expert_buffer_{level}.npy')
        self.reward_buffer = np.load(f'{path}/reward_buffer_{level}.npy')
        self.done_buffer = np.load(f'{path}/done_buffer_{level}.npy')


class ReplayBufferText(object):
    """Buffer to store and replay environment transitions."""

    def __init__(self, obs_shape, text_shape, action_shape, reward_shape, capacity, batch_size, device):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device
        # Initialize all the buffers
        self.obs_buffer = np.empty(shape=(capacity, *obs_shape), dtype=np.float32)
        self.text_buffer = np.empty(shape=(capacity, *text_shape), dtype=np.float32)
        self.action_buffer = np.empty(shape=(capacity, *action_shape), dtype=np.float32)
        self.expert_buffer = np.empty(shape=(capacity, *action_shape), dtype=np.float32)
        self.reward_buffer = np.empty(shape=(capacity, *reward_shape), dtype=np.float32)
        self.done_buffer = np.empty(shape=(capacity, *reward_shape), dtype=np.float32)
        self.idx = 0

    def add(self, obs, text, action, expert_action, reward, done):
        if self.idx < self.capacity:
            self.obs_buffer[self.idx] = obs
            self.text_buffer[self.idx] = text
            self.action_buffer[self.idx] = action
            self.expert_buffer[self.idx] = expert_action
            self.reward_buffer[self.idx] = reward
            self.done_buffer[self.idx] = done
            self.idx += 1
        else:
            self.obs_buffer = self.obs_buffer[1:]
            self.obs_buffer = np.append(self.obs_buffer,
                                        obs.reshape((1, obs.shape[0])),
                                        axis=0)
            self.text_buffer = self.text_buffer[1:]
            self.text_buffer = np.append(self.text_buffer,
                                         text.reshape((1, text.shape[0])),
                                         axis=0)
            self.action_buffer = self.action_buffer[1:]
            self.action_buffer = np.append(self.action_buffer,
                                           action.reshape((1, action.shape[0])),
                                           axis=0)
            self.expert_buffer = self.expert_buffer[1:]
            self.expert_buffer = np.append(self.expert_buffer,
                                           expert_action.reshape((1, expert_action.shape[0])),
                                           axis=0)
            self.reward_buffer = self.reward_buffer[1:]
            self.reward_buffer = np.append(self.reward_buffer,
                                           reward.reshape((1, reward.shape[0])),
                                           axis=0)
            self.done_buffer = self.done_buffer[1:]
            self.done_buffer = np.append(self.done_buffer,
                                         done.reshape((1, done.shape[0])),
                                         axis=0)

    def sample(self, time=10):
        idxs = np.random.randint(
            0, self.capacity - time + 1 if self.idx == self.capacity else self.idx - time + 1, size=self.batch_size)
        obses = torch.as_tensor(self.obs_buffer[idxs], device=self.device).unsqueeze(1)
        texts = torch.as_tensor(self.text_buffer[idxs], device=self.device).unsqueeze(1)
        actions = torch.as_tensor(self.action_buffer[idxs], device=self.device).unsqueeze(1)
        expert_actions = torch.as_tensor(self.expert_buffer[idxs], device=self.device).unsqueeze(1)
        rewards = torch.as_tensor(self.reward_buffer[idxs], device=self.device).unsqueeze(1)
        dones = torch.as_tensor(self.done_buffer[idxs], device=self.device).unsqueeze(1)

        for i in range(1, time):
            next_obses = torch.as_tensor(self.obs_buffer[idxs + i], device=self.device).unsqueeze(1)
            next_texts = torch.as_tensor(self.text_buffer[idxs + i], device=self.device).unsqueeze(1)
            next_actions = torch.as_tensor(self.action_buffer[idxs + i], device=self.device).unsqueeze(1)
            next_expert_actions = torch.as_tensor(self.expert_buffer[idxs + i], device=self.device).unsqueeze(1)
            next_rewards = torch.as_tensor(self.reward_buffer[idxs + i], device=self.device).unsqueeze(1)
            next_dones = torch.as_tensor(self.done_buffer[idxs + i], device=self.device).unsqueeze(1)
            obses = torch.cat((obses, next_obses), 1)
            texts = torch.cat((texts, next_texts), 1)
            actions = torch.cat((actions, next_actions), 1)
            expert_actions = torch.cat((expert_actions, next_expert_actions), 1)
            rewards = torch.cat((rewards, next_rewards), 1)
            dones = torch.cat((dones, next_dones), 1)

        return obses, texts, actions, expert_actions, rewards, dones

    def save(self):
        np.save('./model_save/obs_buffer.npy', self.obs_buffer)
        np.save('./model_save/text_buffer.npy', self.text_buffer)
        np.save('./model_save/action_buffer.npy', self.action_buffer)
        np.save('./model_save/expert_buffer.npy', self.expert_buffer)
        np.save('./model_save/reward_buffer.npy', self.reward_buffer)
        np.save('./model_save/done_buffer.npy', self.done_buffer)

    def load(self, dir):
        self.obs_buffer = np.load(dir + 'obs_buffer.npy')
        self.text_buffer = np.load(dir + 'text_buffer.npy')
        self.action_buffer = np.load(dir + 'action_buffer.npy')
        self.expert_buffer = np.load(dir + 'expert_buffer.npy')
        self.reward_buffer = np.load(dir + 'reward_buffer.npy')
        self.done_buffer = np.load(dir + 'done_buffer.npy')
