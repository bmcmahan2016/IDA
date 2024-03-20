'''
Author: Brandon
Date: March 6, 2024
Description: Evaluates the performance of a copilot, surrogate
human pilot, and intervention function in a Gymnasium environment

'''
from envs.utils import make_env
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

def test_IDA(agent, env, advantage_fn, gamma=0.2, num_episodes=100, render=False, margin=0.5):
    #margin= 0.5 #0.1  # 0.5
    timeouts = 0
    successes = 0
    crashes = 0
    num_episodes = num_episodes
    with agent.eval_mode():
        observation, info = env.reset()
        if render:
            rgb_frame = env.render()
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter("debug" + ".mp4", fourcc, 24, (rgb_frame.shape[1], rgb_frame.shape[0]))
        corr_vals = []
        exp_vals = []
        for _ in range(num_episodes):
            action_hist = []
            partial_state_hist = []
            observation, _ = env.reset()
            
            action = agent.act(observation)
            r = 0
            for t_step in range(1000):
                corrupted=False
                ######################################################################
                # get action from copilot
                ######################################################################
                # first concetanate state and isotropic gausian noise for action
                state = torch.from_numpy(observation[env.env.goal_mask])
                action = agent.act(observation)
                #breakpoint()
                if np.random.rand() < 0.3:
                    corrupted=True
                    noise_action = 2*np.random.rand(2) - 1
                    action = noise_action
                #action = 0.65*action + 0.35*noise_action

                #laggy surrogate pilot -- only a 15% chance of updating action from current observation
                # if np.random.rand() > 0.85:
                #     action = agent.act(observation)
                state_conditioned_action = torch.unsqueeze(torch.hstack([state, torch.from_numpy(action).float()]),  axis=0)
                state_conditioned_action = diffusion.sample(copilot, state_conditioned_action, gamma=gamma)
                copilot_action = state_conditioned_action[0,-2:].numpy()
                # state_conditioned_action1 = state_conditioned_action4#diffusion.sample(copilot, state_conditioned_action, gamma=gamma)diffusion.sample(copilot, state_conditioned_action, gamma=gamma)
                
                # state_conditioned_action2 = state_conditioned_action4#diffusion.sample(copilot, state_conditioned_action, gamma=gamma)
                # state_conditioned_action3 = state_conditioned_action4#diffusion.sample(copilot, state_conditioned_action, gamma=gamma)
                
                # state_conditioned_action5 = state_conditioned_action4#diffusion.sample(copilot, state_conditioned_action, gamma=gamma)
                # state_conditioned_action6 = state_conditioned_action4#diffusion.sample(copilot, state_conditioned_action, gamma=gamma)
                # state_conditioned_action7 = state_conditioned_action4#diffusion.sample(copilot, state_conditioned_action, gamma=gamma)
                # state_conditioned_action8 = state_conditioned_action4#diffusion.sample(copilot, state_conditioned_action, gamma=gamma)

                # copilot_action1 = state_conditioned_action1[0,-2:].numpy()
                # copilot_action2= state_conditioned_action2[0,-2:].numpy()
                # copilot_action3 = state_conditioned_action3[0,-2:].numpy()
                # copilot_action4 = state_conditioned_action4[0,-2:].numpy()
                # copilot_action5 = state_conditioned_action5[0,-2:].numpy()
                # copilot_action6= state_conditioned_action6[0,-2:].numpy()
                # copilot_action7 = state_conditioned_action7[0,-2:].numpy()
                # copilot_action8 = state_conditioned_action8[0,-2:].numpy()

                if render:
                    rgb_frame = env.render()
                    video_writer.write(rgb_frame)
                if advantage_fn is not None:
                    copilot_adv = advantage_fn(partial_state_hist + [state.numpy()], action_hist + [copilot_action])
                    expert_adv = advantage_fn(partial_state_hist + [state.numpy()], action_hist + [action])
                    adv = (torch.sum(torch.sign(copilot_adv - expert_adv)) / len(copilot_adv)).item()
                    # adv_2 = advantage_fn(partial_state_hist + [state.numpy()], action_hist + [copilot_action2]) - advantage_fn(partial_state_hist + [state.numpy()], action_hist + [action])
                    # adv_3 = advantage_fn(partial_state_hist + [state.numpy()], action_hist + [copilot_action3]) - advantage_fn(partial_state_hist + [state.numpy()], action_hist + [action])
                    # adv_4 = advantage_fn(partial_state_hist + [state.numpy()], action_hist + [copilot_action4]) - advantage_fn(partial_state_hist + [state.numpy()], action_hist + [action])
                    # adv_5 = advantage_fn(partial_state_hist + [state.numpy()], action_hist + [copilot_action5]) - advantage_fn(partial_state_hist + [state.numpy()], action_hist + [action])
                    # adv_6 = advantage_fn(partial_state_hist + [state.numpy()], action_hist + [copilot_action6]) - advantage_fn(partial_state_hist + [state.numpy()], action_hist + [action])
                    # adv_7 = advantage_fn(partial_state_hist + [state.numpy()], action_hist + [copilot_action7]) - advantage_fn(partial_state_hist + [state.numpy()], action_hist + [action])
                    # adv_8 = advantage_fn(partial_state_hist + [state.numpy()], action_hist + [copilot_action8]) - advantage_fn(partial_state_hist + [state.numpy()], action_hist + [action])
                    # adv_ix = np.argmax([adv_1, adv_2, adv_3, adv_4])
                    # adv = [adv_1, adv_2, adv_3, adv_4, adv_5, adv_6, adv_7, adv_8][adv_ix]
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
    candidate_goals = 0.2 * np.array([
                [1, 0],
                [np.sqrt(2)/2, np.sqrt(2)/2],
                [0, 1],
                [-np.sqrt(2)/2, np.sqrt(2)/2],
                [-1,0],
                [-np.sqrt(2)/2, -np.sqrt(2)/2],
                [0,-1],
                [np.sqrt(2)/2, -np.sqrt(2)/2]
            ])
    if config['advantage_fn']:
        advantage_fn = make_trajectory_intervetion_function(Q_intervention, env, discount=config['advantage_gamma'])
    else:
        advantage_fn = None

    hist = []
    for _ in tqdm.tqdm(range(config['num_evaluations'])):
        env.env.initialize_goal_space()     # required to reset goals for each evaluation
        successes, crashes, timeouts, corr_vals, exp_vals = test_IDA(agent,
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

