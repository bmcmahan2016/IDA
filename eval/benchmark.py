
import numpy as np
import torch
import argparse
from copilots.diffusion import Diffusion
from copilots.modules import SharedAutonomy
import yaml
import os
from pathlib import Path
from envs.utils import make_env
from experts.agents.sac import make_SAC
import colored_traceback
import tqdm
from pilots import SurrogatePilot
from intervention.functions import InterventionFunction
from multiprocessing import Manager, Pool
from datetime import datetime
import matplotlib.pyplot as plt
import cv2

colored_traceback.add_hook(always=True)


def test_IDA(agent, 
             env, 
             copilot, 
             diffusion, 
             advantage_fn, 
             corruption_type='noise', 
             output_path = None,
             gamma=0.2, 
             num_episodes=100, 
             render=False, 
             margin=0.5, 
             discard_reacher_control_reward=False
             ):
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
    corruption_flag = []
    copilot_advs = []
    intervened_flag = []
    episode_returns = []
    episode_lengths = []
    # history of all past positions
    x_pos_history = []  
    y_pos_history = []
    for _ in range(num_episodes):
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
            x_pos_history.append(state[0])
            y_pos_history.append(state[1])
            action, corrupted = agent.act(observation)
            corruption_flag.append(corrupted)

            state_conditioned_action = torch.unsqueeze(torch.hstack([state, torch.from_numpy(action).float()]),  axis=0)
            state_conditioned_action = diffusion.sample(copilot, state_conditioned_action, gamma=gamma)
            copilot_action = state_conditioned_action[0,-2:].numpy()


            behavior_action, adv = advantage_fn.behavior_policy(state.numpy(), action, copilot_action) 
            if np.all(behavior_action != action):
                intervened_flag.append(True)
            else:
                intervened_flag.append(False)
            copilot_advs.append(adv)
            
            observation, reward, done, terminated, info = env.step(behavior_action)
            if discard_reacher_control_reward:
                reward = info['reward_dist']
            r += reward
            if render:
                rgb_frame = env.render()
                if corrupted:
                    cv2.putText(rgb_frame, "C", (400, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_4)
                if adv > margin:
                    cv2.putText(rgb_frame, "I", (400, 375), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_4)
                cv2.putText(rgb_frame, "Pilot: " + str(action), (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_4)
                cv2.putText(rgb_frame, "Copilot: " + str(copilot_action), (50, 375), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_4)
                frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                cv2.imwrite("frame%d.jpg"%t_step, frame)
                video_writer.write(frame)

            if (done or terminated):
                episode_returns.append(r)
                episode_lengths.append(t_step)
                break
            elif t_step == 999:
                episode_lengths.append(t_step)
            #     if r > 200: #-10:
            #         successes += 1
            #     if reward <= -100:
            #         crashes += 1
            #     break
            # if t_step == 999:
            #     timeouts += 1
    if render:
        video_writer.release()
    return episode_returns,  np.array(corruption_flag), np.array(copilot_advs), np.array(intervened_flag), np.array(x_pos_history), np.array(y_pos_history), np.array(episode_lengths) #successes, crashes, timeouts, np.array(corruption_flag), np.array(copilot_advs), np.array(intervened_flag)
    
def write_reacher_results(eval_returns, output_path, num_goals):

    # if lunar lander use this code

    avg_returns = []
    for episode_returns in eval_returns:
        successes = 0
        crashes = 0
        avg_returns.append(np.mean(episode_returns))

    num_episodes = len(eval_returns[0])  # number of episodes in first evaluation
    num_evaluations = len(eval_returns)
    
    # write results.txt
    mean_return = np.mean(avg_returns)
    return_sem = np.std(avg_returns) / np.sqrt(num_evaluations)
    
    f = open(output_path / 'results.txt', 'a')
    f.write("NUMBER OF GOALS:   " + str(num_goals) + "\n")
    f.write('average return: ' + str(mean_return) + ' +/-' + str(return_sem) + '\n')
    f.write('number of evaluations: ' + str(num_evaluations) + '\n')
    f.write('episodes per eval: ' + str(num_episodes) + '\n')
    f.write("\n")

    f.close()

def write_lunar_lander_results(eval_returns, output_path, num_goals):

    # if lunar lander use this code
    success_rate = []
    crash_rate = []
    for episode_returns in eval_returns:
        successes = 0
        crashes = 0
        for r in episode_returns:
            if r > 200:
                successes += 1
            elif r <= -100:
                crashes += 1
        success_rate.append(successes)
        crash_rate.append(crashes)

    num_episodes = len(eval_returns[0])  # number of episodes in first evaluation
    num_evaluations = len(eval_returns)
    
    # write results.txt
    sr_mean = np.mean(success_rate) / num_episodes
    sr_sem = np.std(success_rate) / np.sqrt(num_evaluations)
    
    crash_mean = np.mean(crash_rate) / num_episodes
    crash_sem = np.std(crash_rate) / np.sqrt(num_evaluations)
    
    f = open(output_path / 'results.txt', 'a')
    f.write("NUMBER OF GOALS:   " + str(num_goals) + "\n")
    f.write('success rate: ' + str(sr_mean) + ' +/-' + str(sr_sem) + '\n')
    f.write('crash rate: ' + str(crash_mean) + ' +/-' + str(crash_sem) + '\n')
    f.write('number of evaluations: ' + str(num_evaluations) + '\n')
    f.write('episodes per eval: ' + str(num_episodes) + '\n')
    f.write("\n")

    f.close()

def evaluate_IDA(agent, env, copilot, diffusion, advantage_fn, output_path, num_evaluations=10, gamma=0.2, num_episodes=100, render=False, margin=0.5):
    sr, cr, to = [], [], []
    # determines if reacher should discard the control return
    drcr = ('distance_reward' in str(output_path))
    eval_returns = []  # list of episode returns (list of list)
    unintervened_corrupt_policy_states = []
    unintervened_expert_policy_states = []
    intervened_corrupt_policy_states = []
    intervened_expert_policy_states = []
    position_hist = []
    episode_durations = []
    intervened_flag_hist = []
    for _ in tqdm.tqdm(range(num_evaluations)):
        env.env.initialize_goal_space()
        #successes, crashes, timeouts, corrupted_flag, copilot_advs, intervened_flag 
        episode_returns, corrupted_flag, copilot_advs, intervened_flag, x_pos, y_pos, episode_lengths  = test_IDA(agent, 
                                                                                    env, 
                                                                                    copilot,
                                                                                    diffusion,
                                                                                    advantage_fn, 
                                                                                    gamma=gamma, 
                                                                                    num_episodes=num_episodes, 
                                                                                    render=render, 
                                                                                    margin=margin,
                                                                                    output_path=output_path,
                                                                                    discard_reacher_control_reward=drcr,
                                                                                    )
        eval_returns.append(episode_returns)
        unintervened_corrupt_policy_states.append(np.sum(corrupted_flag[intervened_flag==False]==True))
        unintervened_expert_policy_states.append(np.sum(corrupted_flag[intervened_flag==False]==False))
        intervened_corrupt_policy_states.append(np.sum(corrupted_flag[intervened_flag==True]==True))
        intervened_expert_policy_states.append(np.sum(corrupted_flag[intervened_flag==True]==False))
        position_hist.append(np.hstack([x_pos.reshape(-1,1), y_pos.reshape(-1,1)]))
        episode_durations.append(episode_lengths)
        intervened_flag_hist.extend(intervened_flag)


    # call a lunar lander specific writing function here
    if 'lunar' in str(output_path):
        write_lunar_lander_results(eval_returns, output_path, advantage_fn._NUM_GOALS)
    # or call a reacher specific writing function here
    if 'reacher' in str(output_path):
        write_reacher_results(eval_returns, output_path, advantage_fn._NUM_GOALS)
    #    sr.append(successes)
    #    cr.append(crashes)
    #    to.append(timeouts)
    # write results.txt
    # sr_mean = np.mean(sr) / num_episodes
    # sr_sem = np.std(sr) / np.sqrt(num_evaluations)
    
    # crash_mean = np.mean(cr) / num_episodes
    # crash_sem = np.std(cr) / np.sqrt(num_evaluations)
    
    # to_mean = np.mean(to) / num_episodes
    # to_sem = np.std(to) / np.sqrt(num_evaluations)
    # num_goals = advantage_fn._NUM_GOALS
    # f = open(output_path / 'results.txt', 'a')
    # f.write("NUMBER OF GOALS:   " + str(num_goals) + "\n")
    # f.write('success rate: ' + str(sr_mean) + ' +/-' + str(sr_sem) + '\n')
    # f.write('crash rate: ' + str(crash_mean) + ' +/-' + str(crash_sem) + '\n')
    # f.write('timeout rate: ' + str(to_mean) + ' +/-' + str(to_sem) + '\n')
    # f.write('number of evaluations: ' + str(num_evaluations) + '\n')
    # f.write('episodes per eval: ' + str(num_episodes) + '\n')
    # f.write("\n")

    # f.close()

    # save information about the intervention function
    if advantage_fn._disable == False:
        num_goals = advantage_fn._NUM_GOALS
        plt.hist(copilot_advs[corrupted_flag==False], alpha=0.5, bins=20)
        plt.hist(copilot_advs[corrupted_flag], alpha=0.5, bins=20)
        plt.legend(['Actions Drawn from Expert Policy', 'Actions Drawn from Corrupted Policy'])
        plt.title("Distribution of Copilot Advantages")
        plt.ylabel("Frequency")
        plt.xlabel("Copilot Advantage Score")
        hist_name = str(num_goals) + "_goals_copilot_advantage_distribution.eps"
        plt.savefig(output_path / hist_name)
        plt.close()

        fname = str(num_goals) + "_goals_copilot_advs.npy"
        np.save(output_path / fname, copilot_advs)
        fname = str(num_goals) + "_goals_corrupted_flag.npy"
        np.save(output_path / fname, corrupted_flag)
        fname = str(num_goals) + "_goals_intervened_flag.npy"
        np.save(output_path / fname, intervened_flag_hist)
        fname = str(num_goals) + "_position_history.npy"
        breakpoint()
        np.save(output_path / fname, np.vstack(position_hist))
        fname = str(num_goals) + "_episode_lengths.npy"
        np.save(output_path / fname, np.array(episode_durations))

        # distribution for un-intervned states
        #corrupt_policy_states = np.sum(corrupted_flag[intervened_flag==False]==True)
        #expert_policy_states = np.sum(corrupted_flag[intervened_flag==False]==False)
        plt.bar([-0.25,0.75], [np.mean(unintervened_expert_policy_states), np.mean(intervened_expert_policy_states)], 
                yerr=[np.std(unintervened_expert_policy_states)/np.sqrt(len(unintervened_expert_policy_states)), np.std(intervened_expert_policy_states)/np.sqrt(len(intervened_expert_policy_states))], width=0.5, capsize=5)
        #plt.title("Unintervened States")
        #fig_name = str(num_goals) + "_goals_unintervened_pie.eps"
        #plt.savefig(output_path / fig_name)
        #plt.close()

        # distribution for intervened states
        #corrupt_policy_states = np.sum(corrupted_flag[intervened_flag==True]==True)
        #expert_policy_states = np.sum(corrupted_flag[intervened_flag==True]==False)
        plt.bar([0.25,1.25], [np.mean(unintervened_corrupt_policy_states), np.mean(intervened_corrupt_policy_states)], 
                yerr=[np.std(unintervened_corrupt_policy_states)/np.sqrt(len(unintervened_corrupt_policy_states)), np.std(intervened_corrupt_policy_states)/np.sqrt(len(intervened_corrupt_policy_states))], width=0.5, capsize=5)
        plt.xticks([0,1], ["unintervend states", "intervened states"])
        plt.title("Intervened States")
        fig_name = str(num_goals) + "_goals_intervened_pie.eps"
        #plt.legend(["Unintervened States", "Intervened States"])
        plt.savefig(output_path / fig_name)
        fig_name = str(num_goals) + "_goals_intervened_pie.png"
        #plt.legend(["Unintervened States", "Intervened States"])
        plt.savefig(output_path / fig_name)
        plt.close()

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
    if config['use_intervention']:
        output_path = Path(config['output_dir']) / Path(config['env']['env_name']) / Path(config['corruption_type']) / "IDA" / str(datetime.now())
    elif config['gamma'] != 0:
        output_path = Path(config['output_dir']) / Path(config['env']['env_name']) / Path(config['corruption_type']) / "Copilot" / str(datetime.now())
    else:
        output_path = Path(config['output_dir']) / Path(config['env']['env_name']) / Path(config['corruption_type']) / "Pilot" / str(datetime.now())
    # prevent over-writting of previous results
    os.makedirs(output_path, exist_ok=False)
    
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
    with open(output_path / 'config.yaml', 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

if __name__=='__main__':
    launch_evaluations()