
import numpy as np
import torch
import argparse
from copilots.diffusion import Diffusion
from copilots.modules import SharedAutonomy
import yaml
import os
from pathlib import Path
import colored_traceback
from envs.utils import make_env
import multiprocessing
from functools import partial
import tqdm
from pilots import SurrogatePilot
from experts.agents.sac import make_SAC
from intervention.functions import make_intervetion_function, make_trajectory_intervetion_function, InterventionFunction
import dill
import concurrent.futures
from multiprocessing import Manager, Pool


colored_traceback.add_hook(always=True)


def test_IDA(agent, env, copilot, diffusion, advantage_fn, corruption_type='noise', gamma=0.2, num_episodes=100, render=False, margin=0.5):
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
        advantage_fn.reset()
        
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


            behavior_action, adv = advantage_fn.behavior_policy(state.numpy(), action, copilot_action) 
            if corrupted:
                corr_vals.append(adv)
            else:
                exp_vals.append(adv)
            
            observation, reward, done, terminated, info = env.step(behavior_action)
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
    
def evaluate_IDA(agent, env, copilot, diffusion, advantage_fn, output_path, num_evaluations=10, gamma=0.2, num_episodes=100, render=False, margin=0.5):
    sr, cr, to = [], [], []
    for _ in tqdm.tqdm(range(num_evaluations)):
        env.env.initialize_goal_space()
        successes, crashes, timeouts, corr_vals, exp_vals = test_IDA(agent, 
                                                                     env, 
                                                                     copilot,
                                                                     diffusion,
                                                                     advantage_fn, 
                                                                     gamma=gamma, 
                                                                     num_episodes=num_episodes, 
                                                                     render=render, 
                                                                     margin=margin
                                                                     )
        sr.append(successes)
        cr.append(crashes)
        to.append(timeouts)
    # write results.txt
    sr_mean = np.mean(sr) / num_episodes
    sr_sem = np.std(sr) / np.sqrt(num_evaluations)
    
    crash_mean = np.mean(cr) / num_episodes
    crash_sem = np.std(cr) / np.sqrt(num_evaluations)
    
    to_mean = np.mean(to) / num_episodes
    to_sem = np.std(to) / np.sqrt(num_evaluations)
    num_goals = advantage_fn._NUM_GOALS
    f = open(output_path / 'results.txt', 'a')
    f.write("NUMBER OF GOALS:   " + str(num_goals) + "\n")
    f.write('success rate: ' + str(sr_mean) + ' +/-' + str(sr_sem) + '\n')
    f.write('crash rate: ' + str(crash_mean) + ' +/-' + str(crash_sem) + '\n')
    f.write('timeout rate: ' + str(to_mean) + ' +/-' + str(to_sem) + '\n')
    f.write('number of evaluations: ' + str(num_evaluations) + '\n')
    f.write('episodes per eval: ' + str(num_episodes) + '\n')
    f.write("\n")

    f.close()

def load_config():
    parser = argparse.ArgumentParser(
        prog="Benchmark IDA",
        description="benchmarks IDA with surrogate pilot"
    )
    parser.add_argument('config_path')
    args = parser.parse_args()
    with open(args.config_path) as config_file:
        config = yaml.safe_load(config_file.read())

    return config

def create_envs(num_goals, env_name, render):
    envs = []
    for N in num_goals:
        if render:
            envs.append(make_env(env_name, render_mode='rgb_array', N=N))
        else:
            envs.append(make_env(env_name, N=N))
    return envs

def parse_num_goals(config):
    if isinstance(config['env']['num_goals'], str):
        num_goals = config['env']['num_goals'].split(',')
        # required to convert from str to int
        num_goals = [int(x) for x in num_goals]
    else:
        num_goals = [config['env']['num_goals']]
    return num_goals

def create_intervention_fns(Q_intervention, envs, num_goals, config):
    if config['use_intervention']:
        disable=False
    else:
        disable=True
    intervention_fns = []
    for N, env in zip(num_goals, envs):
        intervention_fns.append(InterventionFunction(Q_intervention, 
                                                  env, 
                                                  num_goals=N, 
                                                  discount=config['advantage_gamma'], 
                                                  margin=config['margin'], 
                                                  disable=disable))
    return intervention_fns


def launch_evaluations():
    
    config = load_config()
    output_path = Path(config['output_dir'])
    os.makedirs(output_path, exist_ok=True)
    
    num_goals = parse_num_goals(config)
    envs = create_envs(num_goals, config['env']['env_name'], config['render'])

    # loads the copilot
    copilot = SharedAutonomy(obs_size=config['copilot']['state_conditioned_action_dim'])
    copilot.load_state_dict(torch.load(config['copilot']['copilot_path']))
    diffusion = Diffusion(action_dim=config['copilot']['action_dim'], img_size=64, device='cpu')

    # load the intervention function
    agent = make_SAC(envs[0].observation_space.low.size, 
                        envs[0].action_space.low.size, 
                        envs[0].action_space.low, 
                        envs[0].action_space.high)

    agent.load(config['expert_path'])
    pilot = SurrogatePilot(agent, envs[0], config['corruption_type'])
    Q_intervention = agent.q_func1
    intervention_fns = create_intervention_fns(Q_intervention, envs, num_goals, config)

    for env, intervention_fn in zip(envs, intervention_fns):
        evaluate_IDA(pilot, 
                     env, 
                     copilot,
                     diffusion,
                     intervention_fn, 
                     output_path, 
                     num_evaluations=config['num_evaluations'], 
                     gamma=config['gamma'], 
                     num_episodes=config['num_episodes'],
                     render=config['render'])

    
    
if __name__=='__main__':
    launch_evaluations()