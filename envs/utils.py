import gym
import gymnasium
from envs.lunar_lander import LunarLander
from envs.reacher import ReacherEnv
import numpy as np

def make_env(name, render_mode=None, N=0):
    if name.lower()=="cartpole":
        env = gym.make('CartPole-v1', render_mode=render_mode)
        return gymnasium.wrappers.TimeLimit(ContinuousCartPole(env), max_episode_steps=500)
    if name.lower() == "lunarlander":
        env = LunarLander(continuous=True, randomize_helipad=True, render_mode=render_mode)
        # LunarLander is a gym environment and has a different API structure
        # it is necesary to wrap LunarLander for compatability
        env = gymnasium.wrappers.EnvCompatibility(env)
        return env
    if name.lower() == "reacher":
        env = ReacherEnv(render_mode=render_mode)
        return gymnasium.wrappers.TimeLimit(env, max_episode_steps=50)
    if name.lower() == "reacher_continuous":
        env = ReacherEnv(continuous=True, render_mode=render_mode, N=N)
        return gymnasium.wrappers.TimeLimit(env, max_episode_steps=50)

# creates environment
class ContinuousCartPole(gym.ActionWrapper):
    def __init__(self, env):
        super(ContinuousCartPole, self).__init__(env)
        self.action_space = gym.spaces.Box(-1,1) # 1D continuous variable
        # BM: added to allow masking of goal info when collecting expert demonstrations
        self.goal_mask = np.array([True,
                                   True,
                                   True,
                                   False])
        # BM: used to determine if trial was succesful
        self.solved_reward = 400

    def action(self, action):
        if action[0] < 0:
            action = 0
        else: 
            action = 1
        return action
