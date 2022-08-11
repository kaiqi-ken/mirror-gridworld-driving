import numpy as np
import torch
import torch.nn as nn
from rlpyt.utils.buffer import buffer_func
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims, to_onehot, from_onehot

from models.dense import DenseModel, DenseModelNormal
from models.action import ActionDecoder

ModelReturnSpec = namedarraytuple('ModelReturnSpec', ['action', 'state'])


def soft_update_from_to(source, target, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )


class PolicyFilter(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(PolicyFilter, self).__init__()
        # self.policy_filter = torch.nn.Sequential(nn.Linear(input_size, hidden_size),
        #                                          nn.ReLU(),
        #                                          nn.Linear(hidden_size, output_size))
        self.policy_filter = torch.nn.Sequential(nn.Linear(input_size, output_size))

    def forward(self, input):
        output = self.policy_filter(input)
        return output


class PerceptualMaskDis(torch.nn.Module):
    def __init__(self):
        super(PerceptualMaskDis, self).__init__()
        self.obs_mask = torch.nn.Parameter((torch.rand(size=(4,)) - 0.5) * (1.0) - 2.0, requires_grad=True)
        # self.obs_mask = torch.nn.Parameter(torch.tensor([-5.0, -5.0, -0.5, -0.5]), requires_grad=True)
        self.text_mask = torch.nn.Parameter((torch.rand(size=(4,)) - 0.5) * (1.0) - 0.0, requires_grad=True)

    def get_obs_mask(self, obs):
        obs_mask = (obs.abs() < torch.sigmoid(self.obs_mask)).to(torch.float) \
                   + 1.0 * torch.sigmoid(self.obs_mask) \
                   - (1.0 * torch.sigmoid(
            self.obs_mask)).detach()
        return obs_mask

    def get_text_mask(self):
        return None


class PerceptualMask(nn.Module):
    def __init__(self):
        super(PerceptualMask, self).__init__()
        self.obs_mask = torch.nn.Parameter(torch.randn(4, ) + 0.0, requires_grad=True)
        self.text_mask = torch.nn.Parameter(torch.randn(4, ) + 0.0, requires_grad=True)

    def rsample_obs_mask(self):
        obs_mask_dist = torch.distributions.Bernoulli(logits=self.obs_mask)
        obs_mask_rsample = obs_mask_dist.sample() + obs_mask_dist.mean - obs_mask_dist.mean.detach()
        return obs_mask_rsample

    def rsample_text_mask(self):
        text_mask_dist = torch.distributions.Bernoulli(logits=self.text_mask)
        text_mask_rsample = text_mask_dist.sample() + text_mask_dist.mean - text_mask_dist.mean.detach()
        return text_mask_rsample


class AutoEncoder(nn.Module):
    def __init__(self, latent_state_size=16, ego_car_obs_size=2, other_car_obs_size=4 * 3, text_obs_size=4 * 3,
                 action_size=3):
        super(AutoEncoder, self).__init__()

        self.encoder = DenseModelNormal(feature_size=2 * ego_car_obs_size + 2 * other_car_obs_size + 2 * text_obs_size,
                                        output_shape=(latent_state_size,),
                                        hidden_size=64,
                                        layers=3)

        self.encoder_seq = DenseModelNormal(
            feature_size=2 * ego_car_obs_size + 2 * other_car_obs_size + 2 * text_obs_size + latent_state_size + action_size,
            output_shape=(latent_state_size,),
            hidden_size=64,
            layers=3)

        self.transition = DenseModelNormal(feature_size=latent_state_size + action_size,
                                           output_shape=(latent_state_size,),
                                           layers=3,
                                           hidden_size=32)

        self.decoder_ego_car = DenseModel(feature_size=latent_state_size,
                                          output_shape=(ego_car_obs_size,),
                                          layers=3,
                                          hidden_size=32,
                                          dist='normal')

        self.decoder_other_car = DenseModel(feature_size=latent_state_size,
                                            output_shape=(other_car_obs_size,),
                                            layers=3,
                                            hidden_size=32,
                                            dist='normal')

        self.decoder_other_car_text = DenseModel(feature_size=latent_state_size,
                                                 output_shape=(text_obs_size,),
                                                 layers=3,
                                                 hidden_size=32,
                                                 dist='normal')

        self.ego_car_obs_size = ego_car_obs_size
        self.other_car_obs_size = other_car_obs_size
        self.text_obs_size = text_obs_size
        self.latent_state_size = latent_state_size
        # self.transition = None

    def get_latent_state_dist(self, ego_car_obs, other_car_obs, other_car_text,
                              pre_state=None, action=None,
                              ego_car_mask=None, other_car_mask=None, other_car_text_mask=None):
        # input (time, batch, feature)
        # T, B, _ = observation.shape

        if pre_state is not None and action is not None:
            latent_state_dist = self.encoder_seq(torch.cat([ego_car_obs * ego_car_mask,
                                                            other_car_obs * other_car_mask,
                                                            other_car_text * other_car_text_mask,
                                                            ego_car_mask, other_car_mask, other_car_text_mask,
                                                            pre_state, action,
                                                            ], dim=-1))
        else:
            latent_state_dist = self.encoder(torch.cat([ego_car_obs * ego_car_mask,
                                                        other_car_obs * other_car_mask,
                                                        other_car_text * other_car_text_mask,
                                                        ego_car_mask, other_car_mask, other_car_text_mask], dim=-1))
        return latent_state_dist


class BisimModel(nn.Module):
    def __init__(
            self,
            action_shape,
            latent_size=48,
            action_dist='one_hot',
            reward_shape=(1,),
            reward_layers=3,
            reward_hidden=128,
            qf_shape=(1,),
            qf_layers=3,
            qf_hidden=128,
            actor_layers=3,
            actor_hidden=128,
            init_temp=0.2,
            dtype=torch.float,
            q_update_tau=0.005,
            encoder_update_tau=0.005,
            **kwargs,
    ):
        super().__init__()
        self.autoencoder = AutoEncoder(latent_state_size=latent_size,
                                       ego_car_obs_size=2,
                                       other_car_obs_size=4 * 2,
                                       text_obs_size=4 * 3,
                                       action_size=3)

        self.action_shape = action_shape
        output_size = np.prod(action_shape)
        action_size = output_size
        self.action_dist = action_dist

        self.actor_model = ActionDecoder(latent_size, action_size, actor_layers, actor_hidden,
                                         dist='one_hot')
        self.log_alpha = torch.tensor(np.log(init_temp))
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -torch.prod(torch.Tensor(action_shape)).item()

        self.reward_model = DenseModel(latent_size, reward_shape, reward_layers, reward_hidden)

        self.qf1_model = DenseModel(latent_size + action_size, qf_shape, qf_layers, qf_hidden)
        self.qf2_model = DenseModel(latent_size + action_size, qf_shape, qf_layers, qf_hidden)
        self.target_qf1_model = DenseModel(latent_size + action_size, qf_shape, qf_layers, qf_hidden)
        self.target_qf2_model = DenseModel(latent_size + action_size, qf_shape, qf_layers, qf_hidden)

        self.dtype = dtype
        self.q_update_tau = q_update_tau
        self.encoder_update_tau = encoder_update_tau
        self.step_count = 0
        self.init_random_steps = 1000
        self.latent_size = latent_size

    def update_target_networks(self):
        soft_update_from_to(target=self.target_qf1_model, source=self.qf1_model, tau=self.q_update_tau)
        soft_update_from_to(target=self.target_qf2_model, source=self.qf2_model, tau=self.q_update_tau)

    def policy(self, state: torch.Tensor):
        all_actions = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        q_values = []

        # Finding the action with highest Q-value (DQN)
        # size = feat.size()[:-1] + (3,)
        # q = list()
        # for i in range(3):
        #     action_list = torch.zeros(3)
        #     action_list[i] = 1.0
        #     action_list = action_list + torch.zeros(size=size)
        #     features = torch.cat([feat, action_list], dim=-1)
        #     q += [self.qf1_model(features).mean]
        # q = torch.stack(q, dim=0)
        # index = torch.argmax(q, dim=0)
        # action = torch.zeros(size=size)
        # lead_dim = feat.dim() - 1
        # assert lead_dim in (0, 1, 2)
        # if lead_dim == 0:
        #     action[index] = 1.0
        # elif lead_dim == 1:
        #     for i in range(feat.size(0)):
        #         action[i, index[i].item()] = 1.0
        # elif lead_dim == 2:
        #     for i in range(feat.size(0)):
        #         for j in range(feat.size(1)):
        #             action[i, j, index[i, j].item()] = 1.0

        size = state.size()[:-1] + (3,)
        q = list()
        for action in all_actions:
            action = torch.tensor(action).to(state.device).type(self.dtype)
            action = torch.zeros(size=size).to(state.device) + action
            combined_feat = torch.cat([state, action], dim=-1)
            q += [self.qf1_model(combined_feat).mean]
        q = torch.stack(q, dim=0)
        index = torch.argmax(q, dim=0)
        action = torch.zeros(size=size).to(action.device)
        lead_dim = state.dim() - 1
        assert lead_dim in (0, 1, 2)
        if lead_dim == 0:
            action[index] = 1.0
        elif lead_dim == 1:
            for i in range(state.size(0)):
                action[i, index[i].item()] = 1.0
        elif lead_dim == 2:
            for i in range(state.size(0)):
                for j in range(state.size(1)):
                    action[i, j, index[i, j].item()] = 1.0
        action_dist = None

        # action_dist = self.actor_model(state)
        # if self.action_dist == 'tanh_normal':
        #     action = action_dist.rsample()
        # elif self.action_dist == 'one_hot':
        #     action = action_dist.sample()
        #     # This doesn't change the value, but gives us straight-through gradients
        #     action = action + action_dist.probs - action_dist.probs.detach()

        return action, action_dist

    def get_state_representation(self, ego_car_obs, other_car_obs, other_car_text,
                                 action=None, pre_state=None,
                                 ego_car_mask=None, other_car_mask=None, other_car_text_mask=None):
        if ego_car_mask is None:
            ego_car_mask = torch.ones_like(ego_car_obs).to(ego_car_obs.device)
        if other_car_mask is None:
            other_car_mask = torch.ones_like(other_car_obs).to(other_car_obs.device)
        if other_car_text_mask is None:
            other_car_text_mask = torch.ones_like(other_car_text).to(other_car_text.device)
        latent_state_dist = self.autoencoder.get_latent_state_dist(ego_car_obs,
                                                                   other_car_obs,
                                                                   other_car_text,
                                                                   pre_state, action,
                                                                   ego_car_mask, other_car_mask, other_car_text_mask)
        if self.training:
            return latent_state_dist.rsample()
        else:
            return latent_state_dist.sample()


class BisimModelBC(nn.Module):
    def __init__(
            self,
            action_shape,
            latent_size=48,
            action_dist='one_hot',
            reward_shape=(1,),
            reward_layers=3,
            reward_hidden=128,
            qf_shape=(1,),
            qf_layers=3,
            qf_hidden=32,
            actor_layers=1,  # shalow 1, deep 3
            actor_hidden=32,  # shalow 32, deep 64
            init_temp=0.2,
            dtype=torch.float,
            q_update_tau=0.005,
            encoder_update_tau=0.005,
            **kwargs,
    ):
        super().__init__()
        self.encoder_seq = torch.nn.GRU(input_size=2 * (2 + 4 * 2 + 4 * 3) + 3, hidden_size=latent_size, num_layers=1)
        self.encoder = DenseModelNormal(feature_size=2 * (2 + 4 * 2 + 4 * 3),
                                        output_shape=(latent_size,),
                                        hidden_size=32,
                                        layers=3)

        self.action_shape = action_shape
        output_size = np.prod(action_shape)
        action_size = output_size
        self.action_dist = action_dist

        self.actor_model = ActionDecoder(latent_size, action_size, actor_layers, actor_hidden,
                                         dist='one_hot')
        self.log_alpha = torch.tensor(np.log(init_temp))
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -torch.prod(torch.Tensor(action_shape)).item()

        self.reward_model = DenseModel(latent_size, reward_shape, reward_layers, reward_hidden)

        self.qf1_model = DenseModel(latent_size + action_size, qf_shape, qf_layers, qf_hidden)
        self.qf2_model = DenseModel(latent_size + action_size, qf_shape, qf_layers, qf_hidden)
        self.target_qf1_model = DenseModel(latent_size + action_size, qf_shape, qf_layers, qf_hidden)
        self.target_qf2_model = DenseModel(latent_size + action_size, qf_shape, qf_layers, qf_hidden)

        self.dtype = dtype
        self.q_update_tau = q_update_tau
        self.encoder_update_tau = encoder_update_tau
        self.step_count = 0
        self.init_random_steps = 1000
        self.latent_size = latent_size

    def update_target_networks(self):
        soft_update_from_to(target=self.target_qf1_model, source=self.qf1_model, tau=self.q_update_tau)
        soft_update_from_to(target=self.target_qf2_model, source=self.qf2_model, tau=self.q_update_tau)

    def policy(self, state: torch.Tensor):
        all_actions = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        q_values = []

        # Finding the action with highest Q-value (DQN)
        # size = feat.size()[:-1] + (3,)
        # q = list()
        # for i in range(3):
        #     action_list = torch.zeros(3)
        #     action_list[i] = 1.0
        #     action_list = action_list + torch.zeros(size=size)
        #     features = torch.cat([feat, action_list], dim=-1)
        #     q += [self.qf1_model(features).mean]
        # q = torch.stack(q, dim=0)
        # index = torch.argmax(q, dim=0)
        # action = torch.zeros(size=size)
        # lead_dim = feat.dim() - 1
        # assert lead_dim in (0, 1, 2)
        # if lead_dim == 0:
        #     action[index] = 1.0
        # elif lead_dim == 1:
        #     for i in range(feat.size(0)):
        #         action[i, index[i].item()] = 1.0
        # elif lead_dim == 2:
        #     for i in range(feat.size(0)):
        #         for j in range(feat.size(1)):
        #             action[i, j, index[i, j].item()] = 1.0

        size = state.size()[:-1] + (3,)
        q = list()
        for action in all_actions:
            action = torch.tensor(action).to(state.device).type(self.dtype)
            action = torch.zeros(size=size).to(state.device) + action
            combined_feat = torch.cat([state, action], dim=-1)
            q += [self.qf1_model(combined_feat).mean]
        q = torch.stack(q, dim=0)
        index = torch.argmax(q, dim=0)
        action = torch.zeros(size=size).to(action.device)
        lead_dim = state.dim() - 1
        assert lead_dim in (0, 1, 2)
        if lead_dim == 0:
            action[index] = 1.0
        elif lead_dim == 1:
            for i in range(state.size(0)):
                action[i, index[i].item()] = 1.0
        elif lead_dim == 2:
            for i in range(state.size(0)):
                for j in range(state.size(1)):
                    action[i, j, index[i, j].item()] = 1.0
        action_dist = None

        # action_dist = self.actor_model(state)
        # if self.action_dist == 'tanh_normal':
        #     action = action_dist.rsample()
        # elif self.action_dist == 'one_hot':
        #     action = action_dist.sample()
        #     # This doesn't change the value, but gives us straight-through gradients
        #     action = action + action_dist.probs - action_dist.probs.detach()

        return action, action_dist

    def get_state_representation(self, ego_car_obs, other_car_obs, other_car_text,
                                 action=None, pre_state=None,
                                 ego_car_mask=None, other_car_mask=None, other_car_text_mask=None):
        if ego_car_mask is None:
            ego_car_mask = torch.ones_like(ego_car_obs).to(ego_car_obs.device)
        if other_car_mask is None:
            other_car_mask = torch.ones_like(other_car_obs).to(other_car_obs.device)
        if other_car_text_mask is None:
            other_car_text_mask = torch.ones_like(other_car_text).to(other_car_text.device)

        if pre_state is not None and action is not None:
            rnn_input = torch.cat([ego_car_obs * ego_car_mask,
                                   other_car_obs * other_car_mask,
                                   other_car_text * other_car_text_mask,
                                   ego_car_mask, other_car_mask, other_car_text_mask,
                                   action,
                                   ], dim=-1).unsqueeze(0)
            h_pre = pre_state.unsqueeze(0).contiguous()

            if len(rnn_input.size()) == 2:
                rnn_input = rnn_input.unsqueeze(0)
                h_pre = h_pre.unsqueeze(0)
                latent_state, _ = self.encoder_seq(rnn_input, h_pre)
                latent_state = latent_state.squeeze(0).squeeze(0)
            elif len(rnn_input.size()) == 4:
                rnn_input = rnn_input.squeeze(0)
                h_pre = h_pre.squeeze(0)
                latent_state, _ = self.encoder_seq(rnn_input, h_pre)
                latent_state = latent_state
            else:
                latent_state, _ = self.encoder_seq(rnn_input, h_pre)
                latent_state = latent_state.squeeze(0)
        else:
            latent_state = self.encoder(torch.cat([ego_car_obs * ego_car_mask,
                                                   other_car_obs * other_car_mask,
                                                   other_car_text * other_car_text_mask,
                                                   ego_car_mask, other_car_mask, other_car_text_mask], dim=-1)).mean
        return latent_state


class BisimModelGAIL(nn.Module):
    def __init__(
            self,
            action_shape,
            latent_size=48,
            action_dist='one_hot',
            reward_shape=(1,),
            reward_layers=3,
            reward_hidden=128,
            qf_shape=(1,),
            qf_layers=2,
            qf_hidden=128,
            actor_layers=3,  # shalow 1, deep 3
            actor_hidden=64,  # shalow 32, deep 64
            init_temp=0.2,
            dtype=torch.float,
            q_update_tau=0.1,
            encoder_update_tau=0.005,
            **kwargs,
    ):
        super().__init__()
        self.encoder_seq = torch.nn.GRU(input_size=2 * (2 + 4 * 2 + 4 * 3) + 3, hidden_size=latent_size, num_layers=1)
        self.encoder = DenseModelNormal(feature_size=2 * (2 + 4 * 2 + 4 * 3),
                                        output_shape=(latent_size,),
                                        hidden_size=32,
                                        layers=3)

        self.action_shape = action_shape
        output_size = np.prod(action_shape)
        action_size = output_size
        self.action_dist = action_dist

        self.actor_model = ActionDecoder(latent_size, action_size, actor_layers, actor_hidden,
                                         dist='one_hot')
        self.log_alpha = torch.tensor(np.log(init_temp))
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -torch.prod(torch.Tensor(action_shape)).item()

        self.reward_model = DenseModel(latent_size, reward_shape, reward_layers, reward_hidden)

        self.qf1_model = DenseModel(latent_size, qf_shape, qf_layers, qf_hidden)
        self.qf2_model = DenseModel(latent_size, qf_shape, qf_layers, qf_hidden)

        self.dtype = dtype
        self.q_update_tau = q_update_tau
        self.encoder_update_tau = encoder_update_tau
        self.step_count = 0
        self.init_random_steps = 1000
        self.latent_size = latent_size

    def update_target_networks(self):
        soft_update_from_to(target=self.target_qf1_model, source=self.qf1_model, tau=self.q_update_tau)
        soft_update_from_to(target=self.target_qf2_model, source=self.qf2_model, tau=self.q_update_tau)

    def policy(self, state: torch.Tensor):
        all_actions = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        q_values = []

        # Finding the action with highest Q-value (DQN)
        # size = feat.size()[:-1] + (3,)
        # q = list()
        # for i in range(3):
        #     action_list = torch.zeros(3)
        #     action_list[i] = 1.0
        #     action_list = action_list + torch.zeros(size=size)
        #     features = torch.cat([feat, action_list], dim=-1)
        #     q += [self.qf1_model(features).mean]
        # q = torch.stack(q, dim=0)
        # index = torch.argmax(q, dim=0)
        # action = torch.zeros(size=size)
        # lead_dim = feat.dim() - 1
        # assert lead_dim in (0, 1, 2)
        # if lead_dim == 0:
        #     action[index] = 1.0
        # elif lead_dim == 1:
        #     for i in range(feat.size(0)):
        #         action[i, index[i].item()] = 1.0
        # elif lead_dim == 2:
        #     for i in range(feat.size(0)):
        #         for j in range(feat.size(1)):
        #             action[i, j, index[i, j].item()] = 1.0

        size = state.size()[:-1] + (3,)
        q = list()
        for action in all_actions:
            action = torch.tensor(action).to('cuda:1').type(self.dtype)
            action = torch.zeros(size=size).to('cuda:1') + action
            combined_feat = torch.cat([state, action], dim=-1)
            q += [self.qf1_model(combined_feat).mean]
        q = torch.stack(q, dim=0)
        index = torch.argmax(q, dim=0)
        action = torch.zeros(size=size).to(action.device)
        lead_dim = state.dim() - 1
        assert lead_dim in (0, 1, 2)
        if lead_dim == 0:
            action[index] = 1.0
        elif lead_dim == 1:
            for i in range(state.size(0)):
                action[i, index[i].item()] = 1.0
        elif lead_dim == 2:
            for i in range(state.size(0)):
                for j in range(state.size(1)):
                    action[i, j, index[i, j].item()] = 1.0
        action_dist = None

        # action_dist = self.actor_model(state)
        # if self.action_dist == 'tanh_normal':
        #     action = action_dist.rsample()
        # elif self.action_dist == 'one_hot':
        #     action = action_dist.sample()
        #     # This doesn't change the value, but gives us straight-through gradients
        #     action = action + action_dist.probs - action_dist.probs.detach()

        return action, action_dist

    def get_state_representation(self, ego_car_obs, other_car_obs, other_car_text,
                                 action=None, pre_state=None,
                                 ego_car_mask=None, other_car_mask=None, other_car_text_mask=None):
        if ego_car_mask is None:
            ego_car_mask = torch.ones_like(ego_car_obs).to(ego_car_obs.device)
        if other_car_mask is None:
            other_car_mask = torch.ones_like(other_car_obs).to(other_car_obs.device)
        if other_car_text_mask is None:
            other_car_text_mask = torch.ones_like(other_car_text).to(other_car_text.device)

        if pre_state is not None and action is not None:
            rnn_input = torch.cat([ego_car_obs * ego_car_mask,
                                   other_car_obs * other_car_mask,
                                   other_car_text * other_car_text_mask,
                                   ego_car_mask, other_car_mask, other_car_text_mask,
                                   action,
                                   ], dim=-1).unsqueeze(0)
            h_pre = pre_state.unsqueeze(0).contiguous()

            if len(rnn_input.size()) == 2:
                rnn_input = rnn_input.unsqueeze(0)
                h_pre = h_pre.unsqueeze(0)
                latent_state, _ = self.encoder_seq(rnn_input, h_pre)
                latent_state = latent_state.squeeze(0).squeeze(0)
            else:
                latent_state, _ = self.encoder_seq(rnn_input, h_pre)
                latent_state = latent_state.squeeze(0)
        else:
            latent_state = self.encoder(torch.cat([ego_car_obs * ego_car_mask,
                                                   other_car_obs * other_car_mask,
                                                   other_car_text * other_car_text_mask,
                                                   ego_car_mask, other_car_mask, other_car_text_mask], dim=-1)).mean
        return latent_state


class BisimModelSQIL(nn.Module):
    def __init__(
            self,
            action_shape,
            latent_size=48,
            action_dist='one_hot',
            reward_shape=(1,),
            reward_layers=3,
            reward_hidden=128,
            qf_shape=(1,),
            qf_layers=3,
            qf_hidden=32,
            actor_layers=1,  # shalow 1, deep 3
            actor_hidden=32,  # shalow 32, deep 64
            init_temp=0.2,
            dtype=torch.float,
            q_update_tau=0.005,
            encoder_update_tau=0.005,
            **kwargs,
    ):
        super().__init__()
        self.encoder_seq = torch.nn.GRU(input_size=2 * (2 + 4 * 2 + 4 * 3) + 3, hidden_size=latent_size, num_layers=1)
        self.encoder = DenseModelNormal(feature_size=2 * (2 + 4 * 2 + 4 * 3),
                                        output_shape=(latent_size,),
                                        hidden_size=32,
                                        layers=3)

        self.action_shape = action_shape
        output_size = np.prod(action_shape)
        action_size = output_size
        self.action_dist = action_dist

        self.actor_model = ActionDecoder(latent_size, action_size, actor_layers, actor_hidden,
                                         dist='one_hot')
        self.log_alpha = torch.tensor(np.log(init_temp))
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -torch.prod(torch.Tensor(action_shape)).item()

        self.reward_model = DenseModel(latent_size, reward_shape, reward_layers, reward_hidden)

        self.qf1_model = DenseModel(latent_size + action_size, qf_shape, qf_layers, qf_hidden)
        self.qf2_model = DenseModel(latent_size + action_size, qf_shape, qf_layers, qf_hidden)
        self.target_qf1_model = DenseModel(latent_size + action_size, qf_shape, qf_layers, qf_hidden)
        self.target_qf2_model = DenseModel(latent_size + action_size, qf_shape, qf_layers, qf_hidden)

        self.dtype = dtype
        self.q_update_tau = q_update_tau
        self.encoder_update_tau = encoder_update_tau
        self.step_count = 0
        self.init_random_steps = 1000
        self.latent_size = latent_size

    def update_target_networks(self):
        soft_update_from_to(target=self.target_qf1_model, source=self.qf1_model, tau=self.q_update_tau)
        soft_update_from_to(target=self.target_qf2_model, source=self.qf2_model, tau=self.q_update_tau)

    def policy(self, state: torch.Tensor):
        all_actions = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        q_values = []

        # Finding the action with highest Q-value (DQN)
        # size = feat.size()[:-1] + (3,)
        # q = list()
        # for i in range(3):
        #     action_list = torch.zeros(3)
        #     action_list[i] = 1.0
        #     action_list = action_list + torch.zeros(size=size)
        #     features = torch.cat([feat, action_list], dim=-1)
        #     q += [self.qf1_model(features).mean]
        # q = torch.stack(q, dim=0)
        # index = torch.argmax(q, dim=0)
        # action = torch.zeros(size=size)
        # lead_dim = feat.dim() - 1
        # assert lead_dim in (0, 1, 2)
        # if lead_dim == 0:
        #     action[index] = 1.0
        # elif lead_dim == 1:
        #     for i in range(feat.size(0)):
        #         action[i, index[i].item()] = 1.0
        # elif lead_dim == 2:
        #     for i in range(feat.size(0)):
        #         for j in range(feat.size(1)):
        #             action[i, j, index[i, j].item()] = 1.0

        size = state.size()[:-1] + (3,)
        q = list()
        for action in all_actions:
            action = torch.tensor(action).to(state.device).type(self.dtype)
            action = torch.zeros(size=size).to(state.device) + action
            combined_feat = torch.cat([state, action], dim=-1)
            q += [self.qf1_model(combined_feat).mean]
        q = torch.stack(q, dim=0)
        index = torch.argmax(q, dim=0)
        action = torch.zeros(size=size).to(action.device)
        lead_dim = state.dim() - 1
        assert lead_dim in (0, 1, 2)
        if lead_dim == 0:
            action[index] = 1.0
        elif lead_dim == 1:
            for i in range(state.size(0)):
                action[i, index[i].item()] = 1.0
        elif lead_dim == 2:
            for i in range(state.size(0)):
                for j in range(state.size(1)):
                    action[i, j, index[i, j].item()] = 1.0
        action_dist = None

        # action_dist = self.actor_model(state)
        # if self.action_dist == 'tanh_normal':
        #     action = action_dist.rsample()
        # elif self.action_dist == 'one_hot':
        #     action = action_dist.sample()
        #     # This doesn't change the value, but gives us straight-through gradients
        #     action = action + action_dist.probs - action_dist.probs.detach()

        return action, action_dist

    def get_state_representation(self, ego_car_obs, other_car_obs, other_car_text,
                                 action=None, pre_state=None,
                                 ego_car_mask=None, other_car_mask=None, other_car_text_mask=None):
        if ego_car_mask is None:
            ego_car_mask = torch.ones_like(ego_car_obs).to(ego_car_obs.device)
        if other_car_mask is None:
            other_car_mask = torch.ones_like(other_car_obs).to(other_car_obs.device)
        if other_car_text_mask is None:
            other_car_text_mask = torch.ones_like(other_car_text).to(other_car_text.device)

        if pre_state is not None and action is not None:
            rnn_input = torch.cat([ego_car_obs * ego_car_mask,
                                   other_car_obs * other_car_mask,
                                   other_car_text * other_car_text_mask,
                                   ego_car_mask, other_car_mask, other_car_text_mask,
                                   action,
                                   ], dim=-1).unsqueeze(0)
            h_pre = pre_state.unsqueeze(0).contiguous()

            if len(rnn_input.size()) == 2:
                rnn_input = rnn_input.unsqueeze(0)
                h_pre = h_pre.unsqueeze(0)
                latent_state, _ = self.encoder_seq(rnn_input, h_pre)
                latent_state = latent_state.squeeze(0).squeeze(0)
            elif len(rnn_input.size()) == 4:
                rnn_input = rnn_input.squeeze(0)
                h_pre = h_pre.squeeze(0)
                latent_state, _ = self.encoder_seq(rnn_input, h_pre)
                latent_state = latent_state
            else:
                latent_state, _ = self.encoder_seq(rnn_input, h_pre)
                latent_state = latent_state.squeeze(0)
        else:
            latent_state = self.encoder(torch.cat([ego_car_obs * ego_car_mask,
                                                   other_car_obs * other_car_mask,
                                                   other_car_text * other_car_text_mask,
                                                   ego_car_mask, other_car_mask, other_car_text_mask], dim=-1)).mean
        return latent_state


class BisimModelMaxEnt(nn.Module):
    def __init__(
            self,
            action_shape,
            latent_size=48,
            action_dist='one_hot',
            reward_shape=(1,),
            reward_layers=3,
            reward_hidden=128,
            qf_shape=(1,),
            qf_layers=3,
            qf_hidden=128,
            actor_layers=1,  # shalow 1, deep 3
            actor_hidden=32,  # shalow 32, deep 64
            init_temp=0.2,
            dtype=torch.float,
            q_update_tau=0.005,
            encoder_update_tau=0.005,
            **kwargs,
    ):
        super().__init__()
        self.encoder_seq = torch.nn.GRU(input_size=2 * (2 + 4 * 2 + 4 * 3) + 3, hidden_size=latent_size, num_layers=1)
        self.encoder = DenseModelNormal(feature_size=2 * (2 + 4 * 2 + 4 * 3),
                                        output_shape=(latent_size,),
                                        hidden_size=32,
                                        layers=3)

        self.action_shape = action_shape
        output_size = np.prod(action_shape)
        action_size = output_size
        self.action_dist = action_dist

        self.actor_model = ActionDecoder(latent_size, action_size, actor_layers, actor_hidden,
                                         dist='one_hot')
        self.log_alpha = torch.tensor(np.log(init_temp))
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -torch.prod(torch.Tensor(action_shape)).item()

        self.reward_model = DenseModel(latent_size, reward_shape, reward_layers, reward_hidden)

        self.qf1_model = DenseModel(latent_size + action_size, qf_shape, qf_layers, qf_hidden)
        self.qf2_model = DenseModel(latent_size + action_size, qf_shape, qf_layers, qf_hidden)
        self.target_qf1_model = DenseModel(latent_size + action_size, qf_shape, qf_layers, qf_hidden)
        self.target_qf2_model = DenseModel(latent_size + action_size, qf_shape, qf_layers, qf_hidden)

        self.dtype = dtype
        self.q_update_tau = q_update_tau
        self.encoder_update_tau = encoder_update_tau
        self.step_count = 0
        self.init_random_steps = 1000
        self.latent_size = latent_size

    def update_target_networks(self):
        soft_update_from_to(target=self.target_qf1_model, source=self.qf1_model, tau=self.q_update_tau)
        soft_update_from_to(target=self.target_qf2_model, source=self.qf2_model, tau=self.q_update_tau)

    def policy(self, state: torch.Tensor):
        all_actions = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        q_values = []

        # Finding the action with highest Q-value (DQN)
        # size = feat.size()[:-1] + (3,)
        # q = list()
        # for i in range(3):
        #     action_list = torch.zeros(3)
        #     action_list[i] = 1.0
        #     action_list = action_list + torch.zeros(size=size)
        #     features = torch.cat([feat, action_list], dim=-1)
        #     q += [self.qf1_model(features).mean]
        # q = torch.stack(q, dim=0)
        # index = torch.argmax(q, dim=0)
        # action = torch.zeros(size=size)
        # lead_dim = feat.dim() - 1
        # assert lead_dim in (0, 1, 2)
        # if lead_dim == 0:
        #     action[index] = 1.0
        # elif lead_dim == 1:
        #     for i in range(feat.size(0)):
        #         action[i, index[i].item()] = 1.0
        # elif lead_dim == 2:
        #     for i in range(feat.size(0)):
        #         for j in range(feat.size(1)):
        #             action[i, j, index[i, j].item()] = 1.0

        size = state.size()[:-1] + (3,)
        q = list()
        for action in all_actions:
            action = torch.tensor(action).to(state.device).type(self.dtype)
            action = torch.zeros(size=size).to(state.device) + action
            combined_feat = torch.cat([state, action], dim=-1)
            q += [self.qf1_model(combined_feat).mean]
        q = torch.stack(q, dim=0)
        index = torch.argmax(q, dim=0)
        action = torch.zeros(size=size).to(action.device)
        lead_dim = state.dim() - 1
        assert lead_dim in (0, 1, 2)
        if lead_dim == 0:
            action[index] = 1.0
        elif lead_dim == 1:
            for i in range(state.size(0)):
                action[i, index[i].item()] = 1.0
        elif lead_dim == 2:
            for i in range(state.size(0)):
                for j in range(state.size(1)):
                    action[i, j, index[i, j].item()] = 1.0
        action_dist = None

        # action_dist = self.actor_model(state)
        # if self.action_dist == 'tanh_normal':
        #     action = action_dist.rsample()
        # elif self.action_dist == 'one_hot':
        #     action = action_dist.sample()
        #     # This doesn't change the value, but gives us straight-through gradients
        #     action = action + action_dist.probs - action_dist.probs.detach()

        return action, action_dist

    def get_state_representation(self, ego_car_obs, other_car_obs, other_car_text,
                                 action=None, pre_state=None,
                                 ego_car_mask=None, other_car_mask=None, other_car_text_mask=None):
        if ego_car_mask is None:
            ego_car_mask = torch.ones_like(ego_car_obs).to(ego_car_obs.device)
        if other_car_mask is None:
            other_car_mask = torch.ones_like(other_car_obs).to(other_car_obs.device)
        if other_car_text_mask is None:
            other_car_text_mask = torch.ones_like(other_car_text).to(other_car_text.device)

        if pre_state is not None and action is not None:
            rnn_input = torch.cat([ego_car_obs * ego_car_mask,
                                   other_car_obs * other_car_mask,
                                   other_car_text * other_car_text_mask,
                                   ego_car_mask, other_car_mask, other_car_text_mask,
                                   action,
                                   ], dim=-1).unsqueeze(0)
            h_pre = pre_state.unsqueeze(0).contiguous()

            if len(rnn_input.size()) == 2:
                rnn_input = rnn_input.unsqueeze(0)
                h_pre = h_pre.unsqueeze(0)
                latent_state, _ = self.encoder_seq(rnn_input, h_pre)
                latent_state = latent_state.squeeze(0).squeeze(0)
            else:
                latent_state, _ = self.encoder_seq(rnn_input, h_pre)
                latent_state = latent_state.squeeze(0)
        else:
            latent_state = self.encoder(torch.cat([ego_car_obs * ego_car_mask,
                                                   other_car_obs * other_car_mask,
                                                   other_car_text * other_car_text_mask,
                                                   ego_car_mask, other_car_mask, other_car_text_mask], dim=-1)).mean
        return latent_state