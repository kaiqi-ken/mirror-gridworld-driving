#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Example of automatic vehicle control from client side."""

# Import CARLA modules
import random

import numpy as np

import argparse
import logging
import torch
import os

from algo.basic_algo import Bisimulation
from four_car_env import FourCarCommEnv
from cat_model import BisimModel, BisimModelBC, BisimModelSQIL, PerceptualMaskDis, PolicyFilter
from utils.replay_buffer import ReplayBufferText
from utils.visualizer import Visualizer
import time
from maxent.maxent import *

def game_loop(model_save_path, model_id, method, args):
    """ Main loop for agent"""
    robot_model = BisimModel(action_shape=(3,)).to(args.device)
    robot_model.load_state_dict(torch.load('./model_save/model_latest.pt'))
    robot_model.eval()
    perceptual_mask = None
    policy_filter = None
    if method == 'om':
        model = BisimModel(action_shape=(3,)).to(args.device)
        model.load_state_dict(torch.load('./model_save/model_latest.pt'))
        # model.load_state_dict(torch.load('./model_save/reward_model_0.9_constcost/model_latest.pt'))
        # model.load_state_dict(torch.load('./model_save/discout_9_const_cost/model_latest.pt'))
        model.eval()

        perceptual_mask = PerceptualMaskDis().to(args.device)

        policy_filter = PolicyFilter(input_size=model.latent_size, output_size=3, hidden_size=32).to(args.device)

        perceptual_mask.load_state_dict(torch.load(f'{model_save_path}/pm_fog_model_best_{model_id}.pt'))
        # print(torch.sigmoid(perceptual_mask.obs_mask).cpu().detach().numpy())
        policy_filter.load_state_dict(torch.load(f'{model_save_path}/pf_fog_model_best_{model_id}.pt'))
        perceptual_mask.eval()
        policy_filter.eval()
    elif method[0:2] == 'bc':
        model = BisimModelBC(action_shape=(3,)).to(args.device)
        model.load_state_dict(torch.load(f'{model_save_path}/bc_model_best_{model_id}.pt'))
        # model.load_state_dict(torch.load('./model_save/discout_9_const_cost/model_latest.pt'))
        model.eval()
    elif method == 'sqil':
        model = BisimModelSQIL(action_shape=(3,)).to(args.device)
        model.load_state_dict(torch.load(f'{model_save_path}/cf_sqil_fog_model_best_{model_id}.pt'))
        # model.load_state_dict(torch.load('./model_save/discout_9_const_cost/model_latest.pt'))
        model.eval()

    env = FourCarCommEnv(robot_model=robot_model, human_model=model, perceptual_mask=perceptual_mask, policy_filter=policy_filter, method=method)
    visualizer = Visualizer(map_size=env._env.map_size, tile_size=30)

    ego_car_obs, other_car_obs, other_car_text, action = env.reset()

    # visualizer.drawImage(ego_car_obs=ego_car_obs.copy(), other_car_obs=other_car_obs.copy())
    # visualizer.render()

    collect_count = 0
    itr = 0
    episode_count = 0.0

    max_episode = 20
    episode_irl_rewards = []
    episode_irl_reward = 0

    episode_rewards = []
    episode_reward = 0

    episode_collisions = []
    episode_collision = 0

    episode_visual_masks = []
    episode_visual_mask = 0
    episode_text_masks = []
    episode_text_mask = 0
    while True:
        opti_action_mask, opti_visual_action_mask, opti_text_action_mask, \
        visual_action_mask_np, text_action_mask_np = env.adaptive_random_shooting()

        # write simulated agent inside forward
        # opti_action_mask[:] = 0.0
        # opti_action_mask[:] = 1.0

        ego_car_obs, other_car_obs, other_car_text, action, reward, done = env.forward(opti_action_mask)

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
        irl_reward = get_reward_compress(feature_copy, env.maxent_theta)
        # visualizer.drawImage(ego_car_obs=ego_car_obs.copy(), other_car_obs=other_car_obs.copy())
        # visualizer.render()
        # time.sleep(0.4)

        irl_reward, reward, done = np.array([irl_reward]), np.array([reward]), np.array([done])

        episode_reward += reward
        episode_irl_reward += irl_reward
        episode_collision += done
        episode_visual_mask += opti_visual_action_mask[::2]
        episode_text_mask += opti_text_action_mask[::3]
        collect_count += 1
        # # print(collect_count)
        # if done:
        #     print('test')
        if collect_count % args.max_episode_count == 0 or done:
            itr += 1
            episode_count += 1
            episode_rewards += [episode_reward/collect_count]
            episode_irl_rewards += [episode_irl_reward/collect_count]
            episode_visual_masks += [episode_visual_mask.detach().cpu().numpy()/collect_count]
            episode_text_masks += [episode_text_mask.detach().cpu().numpy()/collect_count]
            episode_collisions += [episode_collision]
            # print(episode_collision)

            episode_reward = 0
            episode_irl_reward = 0
            episode_collision = 0
            episode_visual_mask = 0
            episode_text_mask = 0
            ego_car_obs, other_car_obs, other_car_text, action = env.reset()
            task_reward = 0
            collect_count = 0
            if episode_count >= max_episode:
                break

    print('game ends')
    episode_collisions = np.concatenate(episode_collisions)
    episode_rewards = np.concatenate(episode_rewards)
    episode_irl_rewards = np.concatenate(episode_irl_rewards)
    episode_text_masks = np.stack(episode_text_masks, axis=0)
    episode_visual_masks = np.stack(episode_visual_masks, axis=0)
    print(np.sum(episode_collisions))
    print(np.mean(episode_text_masks), np.mean(episode_visual_masks))
    print(f'{model_save_path}/')

    if os.path.isdir(f'{model_save_path}/') is not True:
        os.mkdir(f'{model_save_path}/')

    pre_fix = ''
    np.save(f'{model_save_path}/{pre_fix}{method}_{id}_{train_level}_rewards',episode_rewards)
    np.save(f'{model_save_path}/{pre_fix}{method}_{id}_{train_level}_irl_rewards', episode_irl_rewards)
    np.save(f'{model_save_path}/{pre_fix}{method}_{id}_{train_level}_collisions', episode_collisions)
    np.save(f'{model_save_path}/{pre_fix}{method}_{id}_{train_level}_visual_masks', episode_visual_masks)
    np.save(f'{model_save_path}/{pre_fix}{method}_{id}_{train_level}_text_masks', episode_text_masks)


# ==============================================================================
# -- main() --------------------------------------------------------------
# ==============================================================================


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        description='CARLA Automatic Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='Print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--tm-port',
        metavar='P',
        default=8000,
        type=int,
        help='port to communicate with TM (default: 8000)')
    argparser.add_argument(
        '--render',
        default=True,
        type=bool,
        help='Render display of 3rd person view (default: True)'
    )
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='Window resolution (default: 1280x720)')
    argparser.add_argument(
        '-l', '--loop',
        action='store_true',
        dest='loop',
        help='Sets a new random destination upon reaching the previous one (default: False)',
        default=True)
    argparser.add_argument(
        '-b', '--behavior', type=str,
        choices=["cautious", "normal", "aggressive"],
        help='Choose one of the possible agent behaviors (default: normal) ',
        default='normal')
    argparser.add_argument("-a", "--agent", type=str,
                           choices=["Behavior", "Roaming", "Basic"],
                           help="select which agent to run",
                           default="Behavior")
    argparser.add_argument(
        '-s', '--seed',
        help='Set seed for repeating executions (default: None)',
        default=None,
        type=int)
    argparser.add_argument(
        '--max_episode_count',
        help='Max count for each data collection phase',
        default=10,
        type=int)
    argparser.add_argument(
        '--device',
        help='GPU device',
        default=1,
        type=int)
    argparser.add_argument(
        '--gpt_device',
        help='GPU device for gpt',
        default=1,
        type=int)
    argparser.add_argument(
        '--train_level',
        help='train_level',
        default=0.2,
        type=float)
    argparser.add_argument(
        '--method',
        help='method',
        default='om',
        type=str)
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
    args = argparser.parse_args()

    test_case = args.type

    args.width, args.height = [int(x) for x in args.res.split('x')]
    args.device = torch.device('cuda', args.device)

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)
    if test_case == 'within':
        for id in range(20, 25):
            train_level = args.train_level
            model_id = f'{id}_{train_level}'
            model_save_path = f'./model_save/same_people/{args.method}_round_{args.round}/{id}'

            try:
                game_loop(model_save_path=model_save_path, model_id=model_id, method=args.method, args=args)
            except KeyboardInterrupt:
                print('\nCancelled by user. Bye!')
    else:
        for id in range(20, 25):
            train_level = args.train_level
            model_id = f'{id}_{int(train_level)}'
            model_save_path = f'./model_save/between_people/{args.method}_round_{args.round}/{id}'

            try:
                game_loop(model_save_path=model_save_path, model_id=model_id, method=args.method, args=args)
            except KeyboardInterrupt:
                print('\nCancelled by user. Bye!')
