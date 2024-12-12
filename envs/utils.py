import gym
import gymnasium
from envs.lunar_lander import LunarLander
from envs.reacher import ReacherEnv
import numpy as np

def make_env(name, render_mode=None, N=0, exploring_starts=False, Y_min=-0.2, Y_max=0.2, X_min=-0.2, X_max=0.2):
    if name.lower()=="cartpole":
        env = gym.make('CartPole-v1', render_mode=render_mode)
        return gymnasium.wrappers.TimeLimit(ContinuousCartPole(env), max_episode_steps=500)
    if name.lower() == "lunar_lander":
        if N==0:
            N=10
        env = LunarLander(continuous=True, randomize_helipad=True, N=N)
        # LunarLander is a gym environment and has a different API structure
        # it is necesary to wrap LunarLander for compatability
        # render_mode is a Gymnasium env kwarg
        return env
    if name.lower() == "lunarlander":
        if N==0:
            N=10
        env = LunarLander(continuous=True, randomize_helipad=True, N=N, exploring_starts=exploring_starts)
        # LunarLander is a gym environment and has a different API structure
        # it is necesary to wrap LunarLander for compatability
        # render_mode is a Gymnasium env kwarg
        env = gymnasium.wrappers.EnvCompatibility(env, render_mode=render_mode)
        return env
    if name.lower() == "reacher":
        env = ReacherEnv(render_mode=render_mode)
        return gymnasium.wrappers.TimeLimit(env, max_episode_steps=50)
    if name.lower() == "reacher_continuous":
        env = ReacherEnv(continuous=True, render_mode=render_mode, N=N, X_min=X_min, X_max=X_max, Y_min=Y_min, Y_max=Y_max)
        return gymnasium.wrappers.TimeLimit(env, max_episode_steps=50)
    if name.lower() == "reacher_discrete":
        env = ReacherEnv(continuous=True, render_mode=render_mode, N=N)
        return gymnasium.wrappers.TimeLimit(env, max_episode_steps=50)
    if name.lower() == "reacher_linear":
        env = ReacherEnv(continuous=True, render_mode=render_mode, N=N, Y_max=Y_max, Y_min=Y_min)
        return gymnasium.wrappers.TimeLimit(env, max_episode_steps=50)
    if name.lower() == "reacher_subspace":
        env = ReacherEnv(continuous=True, render_mode=render_mode, N=N, X_min=X_min, Y_min=Y_min, X_max=X_max, Y_max=Y_max)
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
