from metadrive.envs.multigoal_intersection import MultiGoalIntersectionEnv
import gymnasium as gym
from collections import defaultdict
from agents.sac import make_SAC
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

config = dict(
        use_render=False,
        manual_control=False,
        vehicle_config=dict(show_lidar=False, show_navi_mark=True, show_line_to_navi_mark=True),
        accident_prob=0.0,
        decision_repeat=5,
        horizon=500,  # to speed up training
)

env = gym.vector.AsyncVectorEnv([
    lambda: MultiGoalIntersectionEnv(config),
    lambda: MultiGoalIntersectionEnv(config),
    lambda: MultiGoalIntersectionEnv(config),
    lambda: MultiGoalIntersectionEnv(config),
    lambda: MultiGoalIntersectionEnv(config),
    lambda: MultiGoalIntersectionEnv(config),
])

expert = make_SAC(env.single_observation_space.low.size,
                  env.single_action_space.low.size,
                  env.single_action_space.low,
                  env.single_action_space.high,
                  lr=1e-4)

episode_rewards = defaultdict(float)
obs, info = env.reset()
obs_recorder = defaultdict(list)

episode_ix = 0
episode_returns = np.zeros((500_000))

for i in tqdm(range(1, 500_000)):
    batch_actions = expert.batch_act(obs)
    obs, r, tm, tc, info = env.step(batch_actions)
    episode_returns[episode_ix] += r[0]

    if tm[0] or tc[0]:
        episode_ix += 1

    expert.batch_observe(obs, r, tm, tc)

    if (tm[0] or tc[0]):
        plt.plot(episode_returns[:episode_ix])
        plt.savefig("metadrive_2_returns.png")
        plt.close()
        if episode_ix %20 == 0:
            expert.save("metadrive_2_expert_" + str(episode_ix))

env.close()