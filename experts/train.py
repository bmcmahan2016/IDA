'''
Train an expert on the CartPole Environment and save trajectories

Author: Brandon McMahan
Date: February 8, 2024
'''

import gymnasium as gym
import os
from envs.utils import make_env
from agents.sac import make_SAC
import tqdm
import numpy as np
import matplotlib.pyplot as plt




n_episodes = 100_000
max_episode_len = 1000
reward_hist = []


def train_expert(env, expert, output_path):
    for _ in tqdm.tqdm(range(1, n_episodes + 1)):

        observation, info = env.reset() 
        R = 0 # sum of episode rewards
        t = 0 # episode timestep
        while True:
            action = expert.act(observation) #env.action_space.sample()
            observation, reward, done, terminated, info = env.step(action)
            R += reward
            t += 1
            reset = t == max_episode_len

            expert.observe(observation, reward, done, reset)
            if done or reset or terminated:
                reward_hist.append(R)
                break

        if _%100 == 1:
            agent_name = output_path / ( str(_) + "_reward_" + str(R))
            expert.save(agent_name)
            plt.figure()
            plt.plot(reward_hist)
            plt.xlabel("Episode")
            plt.ylabel("Total Episode Return")
            plt.savefig(output_path / "expert_training_loss.png")
            plt.close()


            
# do policy rollouts and record the trajectories 
def collect_trajectories(env, expert, output_path, num_rollouts, solved_reward):       
    trajectories = []
    with expert.eval_mode():
        for _ in tqdm.tqdm(range(num_rollouts)):
            episode_reward = 0
            state_actions = []
            observation, info = env.reset()
            while True:
                action = expert.act(observation) 
                state_actions.append(np.concatenate([observation[env.env.goal_mask], action]))
                observation, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                done = (terminated or truncated)
                #env.render()
                if done:
                    # lander gets a reward of 1,000 on succesful land
                    if episode_reward >= solved_reward:  
                        state_actions = np.array(state_actions)
                        trajectories.append(state_actions)
                    break
    trajectories = np.vstack(trajectories)
    save_path = output_path / "demonstrations.npy"
    np.save(save_path, trajectories)


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(
        prog="Expert Training",
        description="trains an expert and uses this expert to collect demonstrations"
    )

    parser.add_argument('-env', '--environment', default='lunarlander')
    parser.add_argument('-agent', '--agent_type', default='sac')
    parser.add_argument('-out', '--output_dir')
    parser.add_argument('-collect_rollouts', action='store_true')
    parser.add_argument('-agent_path', default=None)
    parser.add_argument('-lr', default=3e-4)
    parser.add_argument('--rollouts', default=100000, type=int)
    parser.add_argument('--solved_reward', default=200, type=int)

    args = parser.parse_args()

    # initialize environment
    env = make_env(args.environment, exploring_starts=True)

    # creates expert agent
    if args.agent_type=='sac':
        expert = make_SAC(env.observation_space.low.size, 
                        env.action_space.low.size, 
                        env.action_space.low, 
                        env.action_space.high,
                        lr=args.lr)

    if args.agent_path is None:
        train_expert(env, expert, Path(args.output_dir))
    else:
        expert.load(args.agent_path)
        os.mkdir(args.output_dir)
    if args.collect_rollouts:
        collect_trajectories(env, expert, Path(args.output_dir), args.rollouts, args.solved_reward)
