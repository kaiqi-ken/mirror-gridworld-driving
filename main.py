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

from algo.basic_algo import Bisimulation
from four_car_env import FourCarEnv
from cat_model import BisimModel
from utils.replay_buffer import ReplayBufferText
from utils.visualizer import Visualizer


def game_loop(args):
    """ Main loop for agent"""

    # Initializations
    env = None
    replay_buffer = ReplayBufferText(obs_shape=(2 + 4 * 2,),
                                     text_shape=(4 * 3,),
                                     action_shape=(3,),
                                     reward_shape=(1,),
                                     capacity=100000,
                                     batch_size=50,
                                     device=torch.device('cuda', 1))

    try:
        model = BisimModel(action_shape=(3,)).to(args.device)
        model.eval()
        algo = Bisimulation(model=model, device=args.device)
        env = FourCarEnv()
        visualizer = Visualizer(map_size=env.map_size, tile_size=30)

        ego_car_obs, other_car_obs, other_car_text, reward, done = env.reset()
        collect_count = 0
        itr = 0
        task_reward = 0.0
        episode_count = 0.0
        beta, beta_decay = 0.5, 1.0
        explore_rate = 0.5
        explore_decay = 1.0
        expert_flag = False
        is_init = True

        state = None
        control = None
        while True:
            ego_car_obs = torch.tensor(ego_car_obs).float().to(args.device)
            other_car_obs = torch.tensor(other_car_obs).float().to(args.device)
            other_car_text = torch.tensor(other_car_text).float().to(args.device)
            expert_control = env.expert_policy()

            if expert_flag:
                control = expert_control
                control = np.zeros(shape=(3,))
                control[0] = 1.0
            else:
                if control is not None:
                    control_torch = torch.tensor(control).float().to(args.device)
                    state = model.get_state_representation(ego_car_obs, other_car_obs, other_car_text,
                                                           action=control_torch, pre_state=state,
                                                           ego_car_mask=None, other_car_mask=None,
                                                           other_car_text_mask=None)
                else:
                    state = model.get_state_representation(ego_car_obs, other_car_obs, other_car_text,
                                                           action=None, pre_state=None,
                                                           ego_car_mask=None, other_car_mask=None,
                                                           other_car_text_mask=None)

                control, _ = model.policy(state)
                control = control.detach().cpu().numpy()

                if random.uniform(0, 1) < explore_rate:
                    control = np.zeros(shape=(3,))
                    # control[random.choice([0, 1, 2])] = 1.0
                    control[0] = 1.0

                q = []
                q.append(model.qf1_model(
                    torch.cat([state, torch.tensor([1., 0., 0.]).to(args.device).type(torch.float)],
                              dim=-1)).mean.item())
                q.append(model.qf1_model(
                    torch.cat([state, torch.tensor([0., 1., 0.]).to(args.device).type(torch.float)],
                              dim=-1)).mean.item())
                q.append(model.qf1_model(
                    torch.cat([state, torch.tensor([0., 0., 1.]).to(args.device).type(torch.float)],
                              dim=-1)).mean.item())
                              
            next_ego_car_obs, next_other_car_obs, next_other_car_text, reward, done = env.forward(control)

            task_reward += reward

            reward, done = np.array([reward]), np.array([done])
            replay_buffer.add(np.concatenate([next_ego_car_obs, next_other_car_obs]), next_other_car_text,
                              control, expert_control, reward, done)  # obs_t, a_{t-1}, r_t, d_t
            ego_car_obs = next_ego_car_obs
            other_car_obs = next_other_car_obs
            other_car_text = next_other_car_text
            collect_count += 1
            if collect_count % args.max_episode_count == 0 or done:
                itr += 1
                episode_count += 1
                print(collect_count)

                if itr >= 100 and episode_count > 30:
                    episode_count = 0
                    model.train()
                    model = algo.optimize_agent(model, replay_buffer, itr - 3)
                    model.eval()
                    if beta > 0.2:
                        beta *= beta_decay
                    if explore_rate > 0.2:
                        explore_rate *= explore_decay
                    expert_flag = True if random.uniform(0, 1) < beta else False
                ego_car_obs, other_car_obs, other_car_text, reward, done = env.reset()
                algo.writer.add_scalar('Loss/task_reward', task_reward / (1 + collect_count), itr)
                task_reward = 0
                collect_count = 0
            # Initial reset
            if is_init and collect_count == 50:
                collect_count = 0
                ego_car_obs, other_car_obs, other_car_text, reward, done = env.reset()
                is_init = False

    finally:
        print('game ends')


# ==============================================================================
# -- main() --------------------------------------------------------------
# ==============================================================================


def main():
    """Main method"""

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

    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]
    args.device = torch.device('cuda', args.device)

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    try:
        game_loop(args)
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':
    main()
