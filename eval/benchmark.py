'''
Author: Brandon
Date: March 6, 2024
Description: Evaluates the performance of a copilot, surrogate
human pilot, and intervention function in a Gymnasium environment

'''
from envs.utils import make_env
from datetime import datetime
import numpy as np  
import tqdm
import pygame
from gym.utils.play import play
import torch
from copilots.diffusion import Diffusion
from copilots.modules import SharedAutonomy
import torch.distributions as distributions
import torch.nn as nn
from pfrl.nn.lmbda import Lambda
import pfrl
from pfrl import replay_buffers
from experts.agents.sac import make_SAC
import cv2
from intervention.functions import make_intervetion_function, make_trajectory_intervetion_function

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



class SurrogatePilot(object):
    def __init__(self, 
                 expert, 
                 env,
                 corruption_type,
                 corruption_prob = 0.1,
                 restore_prob = 0.1):
        self._expert = expert
        self._corruption_type = corruption_type
        self._corruption_prob = corruption_prob
        self._restore_prob = restore_prob
        self._env = env

        self._last_action = None
        self._corruption_on = False

    def reset(self):
        self._prev_action = None
        self._corruption_on = False

    def _is_corrupted(self):
        # return True and possibly turn corruption off
        if self._corruption_on:
            if np.random.rand() < self._restore_prob:
                self._corruption_on = False
            return True
        
        # return False and possibly turn corruption on
        else:
            if np.random.rand() < self._corruption_prob:
                self._corruption_on = True
            return False

    def act(self, observation):
        with self._expert.eval_mode():

            # default expert action
            action = self._expert.act(observation)
            if self._corruption_type=='none':
                self._prev_action = action
                return action, False
            # no-op and random corruption are performed always if specified
            if self._corruption_type=="no-op":
                action = np.zeros_like(self._expert.act(observation))
                return action, False
            if self._corruption_type=="random":
                action = self._env.action_space.sample()
                return action, False
        
            # corrupt with noise or lag if corruption is on
            is_corrupted = self._is_corrupted()
            if is_corrupted:
                if self._corruption_type=='noise':
                    action = self._env.action_space.sample()
                if self._corruption_type=='lag':
                    if self._prev_action is None:
                        action = self._expert.act(observation)
                    else:
                        action = self._prev_action

        # return action and store previous
        self._prev_action = action
        return action, is_corrupted


def test_IDA(agent, env, advantage_fn, corruption_type='noise', gamma=0.2, num_episodes=100, render=False, margin=0.5):
    #margin= 0.5 #0.1  # 0.5
    timeouts = 0
    successes = 0
    crashes = 0
    num_episodes = num_episodes
    observation, info = env.reset()
    if render:
        rgb_frame = env.render()
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(output_path) + "/" + str(datetime.now()) + "video.mp4", fourcc, 24, (rgb_frame.shape[1], rgb_frame.shape[0]))
    corr_vals = []
    exp_vals = []
    for _ in range(num_episodes):
        action_hist = []
        partial_state_hist = []
        observation, _ = env.reset()
        agent.reset()
        
        r = 0
        for t_step in range(1000):
            ######################################################################
            # get action from copilot
            ######################################################################
            # first concetanate state and isotropic gausian noise for action
            state = torch.from_numpy(observation[env.env.goal_mask])
            action, corrupted = agent.act(observation)

            state_conditioned_action = torch.unsqueeze(torch.hstack([state, torch.from_numpy(action).float()]),  axis=0)
            state_conditioned_action = diffusion.sample(copilot, state_conditioned_action, gamma=gamma)
            copilot_action = state_conditioned_action[0,-2:].numpy()
            
            if advantage_fn is not None:
                copilot_adv = advantage_fn(partial_state_hist + [state.numpy()], action_hist + [copilot_action])
                expert_adv = advantage_fn(partial_state_hist + [state.numpy()], action_hist + [action])
                adv = (torch.sum(torch.sign(copilot_adv - expert_adv)) / len(copilot_adv)).item()
            else:
                adv = np.inf
            #adv=0
            if corrupted:
                corr_vals.append(adv)
            else:
                exp_vals.append(adv)
            
            if adv > margin:
                observation, reward, done, terminated, info = env.step(copilot_action)
                partial_state_hist.append(state.numpy())
                action_hist.append(copilot_action)
            else:
                observation, reward, done, terminated, info = env.step(action)
                partial_state_hist.append(state.numpy())
                action_hist.append(action)
            r += reward
            if render:
                rgb_frame = env.render()
                if corrupted:
                    rgb_frame[-20:,:] = np.array([255, 0, 0])
                if adv > margin:
                    rgb_frame[-40:-20, :] = np.array([0,0,255])
                video_writer.write(rgb_frame)

            if (done or terminated):
                if r > 200: #-10:
                    successes += 1
                if reward <= -100:
                    crashes += 1
                break
            if t_step == 999:
                timeouts += 1
    if render:
        video_writer.release()
    return successes, crashes, timeouts, corr_vals, exp_vals
    

if __name__ == "__main__":
    import argparse
    import yaml
    import os
    from pathlib import Path
    import colored_traceback
    colored_traceback.add_hook(always=True)

    parser = argparse.ArgumentParser(
        prog="Benchmark IDA",
        description="benchmarks IDA with surrogate pilot"
    )

    parser.add_argument('config_path')
    args = parser.parse_args()
    with open(args.config_path) as config_file:
        config = yaml.safe_load(config_file.read())
    

    output_path = Path(config['output_dir'])
    os.makedirs(output_path, exist_ok=True)

    # load the environment
    if config['render']:
        env = make_env(config['env']['env_name'], render_mode='rgb_array', N=config['env']['num_goals'])
    else:
        env = make_env(config['env']['env_name'], N=config['env']['num_goals'])
    
    # loads the copilot
    copilot = SharedAutonomy(obs_size=config['copilot']['state_conditioned_action_dim'])
    copilot.load_state_dict(torch.load(config['copilot']['copilot_path']))
    diffusion = Diffusion(action_dim=config['copilot']['action_dim'], img_size=64, device='cpu')

    # load the intervention function
    agent = make_SAC(env.observation_space.low.size, 
                        env.action_space.low.size, 
                        env.action_space.low, 
                        env.action_space.high)

    agent.load(config['expert_path'])
    Q_intervention = agent.q_func1
    if config['advantage_fn']:
        advantage_fn = make_trajectory_intervetion_function(Q_intervention, env, discount=config['advantage_gamma'])
    else:
        advantage_fn = None

    # generates the surrogate pilot
    pilot = SurrogatePilot(agent, env, config['corruption_type'])

    hist = []
    for _ in tqdm.tqdm(range(config['num_evaluations'])):
        env.env.initialize_goal_space()     # required to reset goals for each evaluation
        successes, crashes, timeouts, corr_vals, exp_vals = test_IDA(pilot,
                                                env, 
                                                advantage_fn, 
                                                gamma=config['gamma'], 
                                                num_episodes=config['num_episodes'],
                                                render=config['render'],
                                                margin=config['margin'])
        hist.append(successes)

    # write the config file
    with open(output_path / 'config.yaml', 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    plt.hist(corr_vals, alpha=0.33, bins=np.linspace(-1, 1, 10), density=True)
    plt.hist(exp_vals, alpha=0.33, bins=np.linspace(-1, 1, 10), density=True)
    plt.ylabel("Frequency")
    plt.xlabel("Advantage Value")
    plt.legend(["Corrupted Expert" , "Expert"])
    plt.title("I(copilot, expert)")
    plt.savefig(output_path / 'advantages.png')

    # write results.txt
    mean = np.mean(hist)
    std = np.std(hist)
    num_samples = len(hist)
    f = open(output_path / 'results.txt', 'w')
    f.write('mean success: ' + str(mean) + '\n')
    f.write('standard deviation: ' +  str(std) + '\n')
    f.write('number of evaluations: ' + str(num_samples) + '\n')

    f.close()

