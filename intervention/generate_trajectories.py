''' 
Generates trajectories for training a Q function
that will eventually be used to compute intervention
scores

Author: Brandon McMahan
Date: March 20, 2024
'''
import numpy as np
from envs.utils import make_env
from experts.agents.sac import make_SAC
import cv2

def corrupt_action(action, last_action, env, p=0.15):
    if np.random.rand() > p:
        return action
    corrupt_type = np.random.choice(["noise"])
    if corrupt_type == "noise":
        action = env.action_space.sample()
    elif corrupt_type == "lag":
        action=last_action
    elif corrupt_type == "no-op":
        action = np.zeros(env.action_space.low.shape)
    elif corrupt_type == "convex":
        random_action = env.action_space.sample()
        blend_factor = np.random.rand()
        action = (1-blend_factor)*action + blend_factor*random_action
    elif corrupt_type == "blend":
        noise_action = env.action_space.sample()
        alpha = np.random.rand()
        beta= np.random.rand()
        corrupting_action = (1-alpha)*noise_action + alpha*last_action
        action = (1-beta)*action + beta*corrupting_action
    return action



# create the environment
env = make_env('lunarlander', N=10, render_mode='rgb_array', exploring_starts=True)
observation, info = env.reset()
rgb_frame = env.render()
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter("lunar_lander_rollouts.mp4", fourcc, 24, (rgb_frame.shape[1], rgb_frame.shape[0]))
# load in the agents
expert = make_SAC(env.observation_space.low.size,
                  env.action_space.low.size,
                  env.action_space.low,
                  env.action_space.high
                  )
expert.load('/home/necl/code/IDA/experts/runs/lunar_lander/101_reward_-33.35258773585455')

corruption_on = False
CORRUPT_PROB = 0.01
RESTORE_PROB = 0.1

# perform N policy rollouts
NUM_TUPLES = 1_000_000
NUM_EPISODES = 50
MAX_ALLOWED_T_STEPS = 1_000
tuple_ix = 0
tuples = np.zeros((NUM_TUPLES, 2*env.action_space.low.shape[0] + 2*env.observation_space.low.shape[0] + 2))
with expert.eval_mode():
    for episode_num in range(NUM_EPISODES):
        episode_done = False
        action_trajectory = np.zeros((MAX_ALLOWED_T_STEPS, env.action_space.low.shape[0]))
        state_trajectory = np.zeros((MAX_ALLOWED_T_STEPS, env.observation_space.low.shape[0]))
        reward_trajectory = np.zeros((MAX_ALLOWED_T_STEPS, 1))
        done_trajectory = np.zeros((MAX_ALLOWED_T_STEPS, 1))
        t_step = 0
        last_action = env.action_space.sample()
        observation, info = env.reset()
        while not episode_done:
            action = expert.act(observation)
            state_trajectory[t_step] = observation
            # corrupt action with probability p`
            action = corrupt_action(action, last_action, env)
            
            observation, reward, terminated, truncated, info = env.step(action)

            rgb_frame = env.render()
            if corruption_on:
                rgb_frame[-20:, :, :] = np.array([0, 0, 255])
            video_writer.write(rgb_frame)

            # record (S, A, R, S', A')
            action_trajectory[t_step] = action
            reward_trajectory[t_step] = reward
            # increment the t_step
            t_step += 1
            if t_step >= MAX_ALLOWED_T_STEPS-1 or terminated or truncated:
                break
        done_trajectory[t_step] = 1
        state_trajectory[t_step] = observation
        # last action doesn't matter so we will use a random action
        action_trajectory[t_step] = env.action_space.sample()  

        # save the trajectory 
        for t_ix in range(t_step):
            tuples[tuple_ix] = np.hstack([state_trajectory[t_ix], 
                                    action_trajectory[t_ix], 
                                    reward_trajectory[t_ix], 
                                    done_trajectory[t_ix+1],
                                    state_trajectory[t_ix+1], 
                                    action_trajectory[t_ix+1]
                                    ])
            tuple_ix += 1
    video_writer.release()
