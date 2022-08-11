#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import numpy as np
import os
import torch
import torch.optim as optim
from cat_model import BisimModel, PerceptualMaskDis, PolicyFilter, BisimModelBC
import random
from utils.replay_buffer import HumanDataSet
from utils.module import get_parameters
import argparse

batch_size = 4
batch_length = 10
device = 'cuda:0'


def test_forward(obs_test, text_test, human_action_test, method='om', perceptual_mask=None, policy_filter=None,
                 model=None, env_type='clear'):
    obs = obs_test.reshape(-1, batch_length, 10).transpose(0, 1)
    text = text_test.reshape(-1, batch_length, 12).transpose(0, 1)
    human_action = human_action_test.reshape(-1, batch_length, 3).transpose(0, 1)

    ego_car_obs = obs[..., :2]
    other_car_obs = obs[..., 2:]
    other_car_text = text

    batch_t = ego_car_obs.shape[0]
    batch_b = ego_car_obs.shape[1]

    if method == 'om' or method == 'om-p':
        obs_mask_rsample = perceptual_mask.get_obs_mask(other_car_obs[..., 1::2])
        obs_mask_rsample = torch.cat([obs_mask_rsample[..., 0:1].repeat(1, 1, 2),
                                      obs_mask_rsample[..., 1:2].repeat(1, 1, 2),
                                      obs_mask_rsample[..., 2:3].repeat(1, 1, 2),
                                      obs_mask_rsample[..., 3:4].repeat(1, 1, 2)], dim=-1)

        text_mask_rsample = torch.zeros_like(other_car_text).to(device).type(torch.float)
        ego_car_mask = torch.ones_like(ego_car_obs).to(device).type(torch.float)
    else:
        obs_mask_rsample = (torch.rand(size=other_car_obs.size()) < 1.5).type(torch.float).to(device)
        text_mask_rsample = (torch.rand(size=other_car_text.size()) < -1.5).type(torch.float).to(device)
        ego_car_mask = torch.ones_like(ego_car_obs).to(device).type(torch.float)

    latent_state_rsample = [[]] * batch_t
    latent_state_mean = [[]] * batch_t
    latent_state_std = [[]] * batch_t
    for t in range(batch_t):
        if t == 0:
            if method == 'om' or method == 'om-p':
                latent_state_dist = basic_model.autoencoder.get_latent_state_dist(
                    ego_car_obs[t], other_car_obs[t], other_car_text[t],
                    pre_state=None, action=None,
                    ego_car_mask=ego_car_mask[t], other_car_mask=obs_mask_rsample[t],
                    other_car_text_mask=text_mask_rsample[t])
                latent_state_rsample[t] = latent_state_dist.rsample()
                latent_state_mean[t] = latent_state_dist.mean
                latent_state_std[t] = latent_state_dist.stddev
            else:
                latent_state = model.get_state_representation(
                    ego_car_obs[t], other_car_obs[t], other_car_text[t],
                    pre_state=None, action=None,
                    ego_car_mask=ego_car_mask[t], other_car_mask=obs_mask_rsample[t],
                    other_car_text_mask=text_mask_rsample[t])
                latent_state_mean[t] = latent_state
        else:
            if method == 'om' or method == 'om-p':
                latent_state_dist = basic_model.autoencoder.get_latent_state_dist(
                    ego_car_obs[t], other_car_obs[t], other_car_text[t],
                    pre_state=latent_state_rsample[t - 1], action=human_action[t],
                    ego_car_mask=ego_car_mask[t], other_car_mask=obs_mask_rsample[t],
                    other_car_text_mask=text_mask_rsample[t])
                latent_state_rsample[t] = latent_state_dist.rsample()
                latent_state_mean[t] = latent_state_dist.mean
                latent_state_std[t] = latent_state_dist.stddev
            else:
                latent_state = model.get_state_representation(
                    ego_car_obs[t], other_car_obs[t], other_car_text[t],
                    pre_state=latent_state_mean[t - 1], action=human_action[t],
                    ego_car_mask=ego_car_mask[t], other_car_mask=obs_mask_rsample[t],
                    other_car_text_mask=text_mask_rsample[t])
                latent_state_mean[t] = latent_state

    # latent_state_rsample = torch.stack(latent_state_rsample, dim=0)
    latent_state_mean = torch.stack(latent_state_mean, dim=0)
    # latent_state_std = torch.stack(latent_state_std, dim=0)

    a_1 = torch.zeros(size=(batch_t, batch_b, 3)).to(device).type(torch.float)
    a_1[..., 0] = 1.0
    a_2 = torch.zeros(size=(batch_t, batch_b, 3)).to(device).type(torch.float)
    a_2[..., 1] = 1.0
    a_3 = torch.zeros(size=(batch_t, batch_b, 3)).to(device).type(torch.float)
    a_3[..., 2] = 1.0

    q_1 = basic_model.qf1_model(torch.cat([latent_state_mean, a_1], dim=-1)).mean
    q_2 = basic_model.qf1_model(torch.cat([latent_state_mean, a_2], dim=-1)).mean
    q_3 = basic_model.qf1_model(torch.cat([latent_state_mean, a_3], dim=-1)).mean

    # p_1 = torch.exp(q_1) / (torch.exp(q_1) + torch.exp(q_2) + torch.exp(q_3))
    # p_2 = torch.exp(q_2) / (torch.exp(q_1) + torch.exp(q_2) + torch.exp(q_3))
    # p_3 = torch.exp(q_3) / (torch.exp(q_1) + torch.exp(q_2) + torch.exp(q_3))

    if method == 'om':
        policy_residule = policy_filter(latent_state_mean)
        policy_dist = torch.distributions.OneHotCategorical(logits=torch.cat([q_1 + 0.1 * policy_residule[..., 0:1],
                                                                              q_2 + 0.1 * policy_residule[..., 1:2],
                                                                              q_3 + 0.1 * policy_residule[..., 2:3]], dim=-1)[
                                                                   :-1])
    elif method == 'om-p':
        policy_dist = torch.distributions.OneHotCategorical(
            logits=torch.cat([q_1,
                              q_2,
                              q_3], dim=-1)[:-1])
    else:
        # policy_dist = torch.distributions.OneHotCategorical(logits=torch.cat([q_1, q_2, q_3], dim=-1)[:-1])
        policy_dist = model.actor_model(latent_state_mean[:-1])
    loss = -policy_dist.log_prob(human_action[1:]).mean()
    return loss


def train(basic_model, train_path_list=None, test_path_list=None, test_only=False, method='om', model_save_path='.',
             model_id=0, env_type='clear'):
    if method == 'om':
        max_epoch = 150
        perceptual_mask = PerceptualMaskDis().to(device)
        policy_filter = PolicyFilter(input_size=basic_model.latent_size, output_size=3, hidden_size=32).to(device)

        # perceptual_mask = PerceptualMask().to(device)
    elif method == 'om-p':
        max_epoch = 150
        perceptual_mask = PerceptualMaskDis().to(device)
    human_dataset = HumanDataSet()

    human_dataset.load_train_data(train_path_list)
    human_dataset.load_test_data(test_path_list)

    if not test_only:
        train_data_loader = torch.utils.data.DataLoader(human_dataset, batch_size=batch_size * batch_length,
                                                        shuffle=True,
                                                        num_workers=4)

        if method == 'om-p':
            optimizer = optim.Adam(get_parameters([perceptual_mask]), lr=0.05)
        elif method == 'om':
            optimizer = optim.Adam(get_parameters([perceptual_mask, policy_filter]), lr=0.05)
        min_val_loss = 1000.0
        pre_loss = 1000
        val_count = 0
        for epoch in range(max_epoch):
            loss_avg = 0.0
            count = 0.0
            for data in train_data_loader:
                obs = data['obs'].to(device)
                text = data['text'].to(device)
                human_action = data['human_action'].to(device)

                obs = torch.as_tensor(human_dataset.obs_train).to(device).type(torch.float)
                text = torch.as_tensor(human_dataset.text_train).to(device).type(torch.float)
                human_action = torch.as_tensor(human_dataset.human_action_train).to(device).type(torch.float)

                if obs.size(0) % batch_length == 0:
                    obs = obs.reshape(-1, batch_length, 10).type(torch.float).transpose(0, 1)
                    text = text.reshape(-1, batch_length, 12).type(torch.float).transpose(0, 1)
                    human_action = human_action.reshape(-1, batch_length, 3).type(torch.float).transpose(0, 1)

                    ego_car_obs = obs[..., :2]
                    other_car_obs = obs[..., 2:]
                    other_car_text = text

                    batch_t = ego_car_obs.shape[0]
                    batch_b = ego_car_obs.shape[1]

                    if method == 'om' or method == 'om-p':
                        obs_mask_rsample = perceptual_mask.get_obs_mask(other_car_obs[..., 1::2])
                        obs_mask_rsample = torch.cat([obs_mask_rsample[..., 0:1].repeat(1, 1, 2),
                                                      obs_mask_rsample[..., 1:2].repeat(1, 1, 2),
                                                      obs_mask_rsample[..., 2:3].repeat(1, 1, 2),
                                                      obs_mask_rsample[..., 3:4].repeat(1, 1, 2)],
                                                     dim=-1)  # * 0.0 + 1.0
                        # obs_mask_rsample = perceptual_mask.rsample_obs_mask().unsqueeze(0).unsqueeze(0)
                        # obs_mask_rsample = torch.cat([obs_mask_rsample[..., 0:1].repeat(batch_t, batch_b, 2),
                        #                               obs_mask_rsample[..., 1:2].repeat(batch_t, batch_b, 2),
                        #                               obs_mask_rsample[..., 2:3].repeat(batch_t, batch_b, 2),
                        #                               obs_mask_rsample[..., 3:4].repeat(batch_t, batch_b, 2)], dim=-1)

                        text_mask_rsample = torch.zeros_like(other_car_text).to(device).type(torch.float)
                        ego_car_mask = torch.ones_like(ego_car_obs).to(device).type(torch.float)
             

                    latent_state_rsample = [[]] * batch_t
                    latent_state_mean = [[]] * batch_t
                    latent_state_std = [[]] * batch_t
                    for t in range(batch_t):
                        if t == 0:
                            if method == 'om' or method == 'om-p':
                                latent_state_dist = basic_model.autoencoder.get_latent_state_dist(
                                    ego_car_obs[t], other_car_obs[t], other_car_text[t],
                                    pre_state=None, action=None,
                                    ego_car_mask=ego_car_mask[t], other_car_mask=obs_mask_rsample[t],
                                    other_car_text_mask=text_mask_rsample[t])
                                latent_state_rsample[t] = latent_state_dist.rsample()
                                latent_state_mean[t] = latent_state_dist.mean
                                latent_state_std[t] = latent_state_dist.stddev
                            
                        else:
                            if method == 'om' or method == 'om-p':
                                latent_state_dist = basic_model.autoencoder.get_latent_state_dist(
                                    ego_car_obs[t], other_car_obs[t], other_car_text[t],
                                    pre_state=latent_state_rsample[t - 1], action=human_action[t],
                                    ego_car_mask=ego_car_mask[t], other_car_mask=obs_mask_rsample[t],
                                    other_car_text_mask=text_mask_rsample[t])
                                latent_state_rsample[t] = latent_state_dist.rsample()
                                latent_state_mean[t] = latent_state_dist.mean
                                latent_state_std[t] = latent_state_dist.stddev
                            

                    # latent_state_rsample = torch.stack(latent_state_rsample, dim=0)
                    latent_state_mean = torch.stack(latent_state_mean, dim=0)
                    # latent_state_std = torch.stack(latent_state_std, dim=0)

                    a_1 = torch.zeros(size=(batch_t, batch_b, 3)).to(device).type(torch.float)
                    a_1[..., 0] = 1.0
                    a_2 = torch.zeros(size=(batch_t, batch_b, 3)).to(device).type(torch.float)
                    a_2[..., 1] = 1.0
                    a_3 = torch.zeros(size=(batch_t, batch_b, 3)).to(device).type(torch.float)
                    a_3[..., 2] = 1.0

                    if method == 'om':
                        q_1 = basic_model.qf1_model(torch.cat([latent_state_mean, a_1], dim=-1)).mean
                        q_2 = basic_model.qf1_model(torch.cat([latent_state_mean, a_2], dim=-1)).mean
                        q_3 = basic_model.qf1_model(torch.cat([latent_state_mean, a_3], dim=-1)).mean
                        policy_residule = policy_filter(latent_state_mean)
                        policy_dist = torch.distributions.OneHotCategorical(
                            logits=torch.cat([q_1 + 0.1 * policy_residule[..., 0:1],
                                              q_2 + 0.1 * policy_residule[..., 1:2],
                                              q_3 + 0.1 * policy_residule[..., 2:3]], dim=-1)[
                                   :-1])
                        loss = -policy_dist.log_prob(human_action[1:]).sum()
                        loss += policy_residule.pow(2).mean() * 5.0
                        if env_type == 'clear':
                            # loss += (torch.sigmoid(perceptual_mask.obs_mask) - 1.0).pow(2).sum() * 0.5
                            loss += (obs_mask_rsample - 1.0).pow(2).sum() * 0.5
                        elif env_type == 'fog':
                            # loss += (torch.sigmoid(perceptual_mask.obs_mask) - 0.0).pow(2).sum() * 0.5
                            loss += (obs_mask_rsample - 0.0).pow(2).sum() * 0.5
                    elif method == 'om-p':
                        q_1 = basic_model.qf1_model(torch.cat([latent_state_mean, a_1], dim=-1)).mean
                        q_2 = basic_model.qf1_model(torch.cat([latent_state_mean, a_2], dim=-1)).mean
                        q_3 = basic_model.qf1_model(torch.cat([latent_state_mean, a_3], dim=-1)).mean
                        policy_dist = torch.distributions.OneHotCategorical(
                            logits=torch.cat([q_1,
                                              q_2,
                                              q_3], dim=-1)[:-1])
                        loss = -policy_dist.log_prob(human_action[1:]).sum()
                        if env_type == 'clear':
                            # loss += (torch.sigmoid(perceptual_mask.obs_mask) - 1.0).pow(2).sum() * 0.5
                            loss += (obs_mask_rsample - 1.0).pow(2).sum() * 0.5
                        elif env_type == 'fog':
                            # loss += (torch.sigmoid(perceptual_mask.obs_mask) - 0.0).pow(2).sum() * 0.5
                            loss += (obs_mask_rsample - 0.0).pow(2).sum() * 0.5
                    else:
                        policy_dist = model.actor_model(latent_state_mean[:-1])
                        loss = -policy_dist.log_prob(human_action[1:]).sum()

                    # regularize policy residule

                    # loss = latent_state_mean.pow(2).sum()
                    # print(q_1, q_2)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    loss_avg += loss  # -policy_dist.log_prob(human_action[1:]).sum()
                    count += 1
                    # print(loss)
            

            # save model
            if os.path.isdir(model_save_path) is not True:
                os.mkdir(f'{model_save_path}/')
            if method == 'om':
                torch.save(perceptual_mask.state_dict(), f'{model_save_path}/{env_type}_pm_model_{model_id}.pt')
                torch.save(policy_filter.state_dict(), f'{model_save_path}/{env_type}_pf_model_{model_id}.pt')
            elif method == 'om-p':
                torch.save(perceptual_mask.state_dict(), f'{model_save_path}/{env_type}_pm-p_model_{model_id}.pt')
            else:
                torch.save(model.state_dict(), f'{model_save_path}/{env_type}_bc_model_{model_id}.pt')

            # load model
            # if method == 'om':
            #     perceptual_mask.load_state_dict(torch.load(f'{model_save_path}/pm_model_{model_id}.pt'))
            #     policy_filter.load_state_dict(torch.load(f'{model_save_path}/pf_model_{model_id}.pt'))
            # else:
            #     model.load_state_dict(torch.load(f'{model_save_path}/bc_model_{model_id}.pt'))
            # test on test data
            test_data_size = human_dataset.obs_test.shape[0]
            validation_size = int(test_data_size / 2)
            obs_test = torch.as_tensor(human_dataset.obs_test).to(device).type(torch.float)
            text_test = torch.as_tensor(human_dataset.text_test).to(device).type(torch.float)
            human_action_test = torch.as_tensor(human_dataset.human_action_test).to(device).type(torch.float)
            if method == 'om':
                loss = test_forward(obs_test[:validation_size],
                                    text_test[:validation_size],
                                    human_action_test[:validation_size],
                                    method=method,
                                    perceptual_mask=perceptual_mask, policy_filter=policy_filter, model=None)
            elif method == 'om-p':
                loss = test_forward(obs_test[:validation_size],
                                    text_test[:validation_size],
                                    human_action_test[:validation_size],
                                    method=method,
                                    perceptual_mask=perceptual_mask, policy_filter=None, model=None)
                                    

            if loss < min_val_loss and epoch > 0:
                min_val_loss = loss.clone()
                if os.path.isdir(model_save_path) is not True:
                    os.mkdir(f'{model_save_path}/')
                if method == 'om':
                    torch.save(perceptual_mask.state_dict(), f'{model_save_path}/{env_type}_pm_model_best_{model_id}.pt')
                    torch.save(policy_filter.state_dict(), f'{model_save_path}/{env_type}_pf_model_best_{model_id}.pt')
                elif method == 'om-p':
                    torch.save(perceptual_mask.state_dict(), f'{model_save_path}/{env_type}_pm-p_model_best_{model_id}.pt')
                    
                val_count = 0.0
            else:
                if min_val_loss < loss:
                    val_count += 1

                if val_count > 10:
                    break
            # pre_loss = loss.clone()
            # print(val_count, loss.item(), min_val_loss.item())
            # else:
            # break
            # print('stop')

    if method == 'om':
        perceptual_mask.load_state_dict(torch.load(f'{model_save_path}/{env_type}_pm_model_best_{model_id}.pt'))
        policy_filter.load_state_dict(torch.load(f'{model_save_path}/{env_type}_pf_model_best_{model_id}.pt'))
        print(torch.sigmoid(perceptual_mask.obs_mask).cpu().detach().numpy())
    elif method == 'om-p':
        perceptual_mask.load_state_dict(torch.load(f'{model_save_path}/{env_type}_pm-p_model_best_{model_id}.pt'))
        print(torch.sigmoid(perceptual_mask.obs_mask).cpu().detach().numpy())
        
    if method == 'om':
        loss = test_forward(obs_test[validation_size:],
                            text_test[validation_size:],
                            human_action_test[validation_size:],
                            method=method,
                            perceptual_mask=perceptual_mask, policy_filter=policy_filter, model=None)
    elif method == 'om-p':
        loss = test_forward(obs_test[validation_size:],
                            text_test[validation_size:],
                            human_action_test[validation_size:],
                            method=method,
                            perceptual_mask=perceptual_mask, policy_filter=None, model=None)
    print(f'{loss.item()}')
    return loss.item()
    # model.load_state_dict(torch.load(f'./exp_data/'))


# ==============================================================================
# -- main() --------------------------------------------------------------
# ==============================================================================


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        description='simple carla')
    argparser.add_argument(
        '--train_level',
        help='train_level',
        default=1.0,
        type=float)
    argparser.add_argument(
        '--method',
        help='method',
        default='om',
        type=str)
    argparser.add_argument(
        '--test_id',
        help='test_id for between subjects test',
        default=23,
        type=int)
    argparser.add_argument(
        '--type',
        help='type',
        default='within',
        type=str)
    argparser.add_argument(
        '--round',
        help='seed round',
        default=1,
        type=int)
    argparser.add_argument(
        '--env_type',
        help='type',
        default='clear',
        type=str)
    args = argparser.parse_args()

    test_case = args.type

    # generate_data()
    basic_model = BisimModel(action_shape=(3,)).to(device)
    basic_model.load_state_dict(torch.load(f'./model_save/used_model/model_latest.pt'))
    # basic_model.load_state_dict(torch.load(f'./best_models/rand_initpos_rand_speed_0.6discount/2model_latest.pt'))
    test_loss_list = []
    if test_case == 'within':
        # within same person
        for id in range(20, 25):
            save_path_train = f'./exp_data/{args.env_type}/train/participant_{id}'
            save_path_test = f'./exp_data/{args.env_type}/test/participant_{id}'

            train_level = args.train_level
            exp_cases_template = [('l', 'lf', '0'),
                                  ('l', 'lr', '0'),
                                  # ('l', 'rf', '0'),
                                  # ('r', 'lr', '0'),
                                  ('r', 'rf', '0'),
                                  ('r', 'rr', '0'),
                                  ('l', 'lf', '1'),
                                  ('l', 'lr', '1'),
                                  # ('l', 'rf', '1'),
                                  # ('r', 'lr', '1'),
                                  ('r', 'rf', '1'),
                                  ('r', 'rr', '1'),
                                  ('l', 'lf', '2'),
                                  # ('l', 'lr', '2'),
                                  # ('l', 'rf', '2'),
                                  ('r', 'lr', '2'),
                                  ('r', 'rf', '2'),
                                  ('r', 'rr', '2')
                                  ]

            train_path_list = []
            test_path_list = []

            exp_cases = exp_cases_template.copy()
            random.shuffle(exp_cases)

            exp_cases_test = exp_cases_template.copy()
            random.shuffle(exp_cases_test)

            for i in range(int(len(exp_cases_template) * train_level)):
                case = exp_cases.pop()
                train_path_list += [f'{save_path_train}/traj_{case[0]}-{case[1]}-{case[2]}.npz']
            for i in range(len(exp_cases_template)):
                case = exp_cases_test[i]
                test_path_list += [f'{save_path_test}/traj_{case[0]}-{case[1]}-{case[2]}.npz']

            if args.method == 'sqil':
                test_loss = game_loop(train_path_list, test_path_list,
                                      model_save_path=f'./model_save/same_people/{args.method}_round_{args.round}/{id}',
                                      model_id=f'{args.id}_{train_level}')
            else:
                test_loss = train_om(basic_model=basic_model, train_path_list=train_path_list,
                                     test_path_list=test_path_list,
                                     method=args.method,
                                     model_save_path=f'./model_save/same_people/{args.method}_round_{args.round}/{id}', model_id=f'{id}_{train_level}',
                                     env_type=args.env_type)
            test_loss_np = np.array([test_loss])
            if os.path.isdir(f'./model_save/same_people/{args.method}_round_{args.round}/') is not True:
                os.mkdir(f'./model_save/same_people/{args.method}_round_{args.round}/')
            np.save(f'./model_save/same_people/{args.method}_round_{args.round}/{id}/test_loss_{train_level}_{args.method}_{args.env_type}', test_loss_np)
   
