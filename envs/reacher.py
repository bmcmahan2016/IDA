import numpy as np

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box


DEFAULT_CAMERA_CONFIG = {"trackbodyid": 0}


class ReacherEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 50,
    }

    def __init__(self, continuous=False, N=0, **kwargs):
        utils.EzPickle.__init__(self, **kwargs)
        observation_space = Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32)
        MujocoEnv.__init__(
            self,
            "reacher.xml",
            2,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )
        self._continuous = continuous
        self.goal_mask = np.array([True,
                                   True,
                                   True,
                                   True,
                                   False,
                                   False,
                                   True,
                                   True,
                                   False,
                                   False,
                                   True])
        self.solved_reward = -50  # considered a success
        self.max_t_steps = 50
        self._N = N
        if self._N > 0:
            self.initialize_goal_space()

    def initialize_goal_space(self):
        goal_space = []
        for ix in range(self._N):
            # sample a goal and append it to the goal space
            while True:
                goal = self.np_random.uniform(low=-0.2, high=0.2, size=2)
                if np.linalg.norm(goal) < 0.2:
                    break
            goal_space.append(goal)
            self.goal_space = np.array(goal_space)
    
    def step(self, a):
        vec = self.get_body_com("fingertip") - self.get_body_com("target")
        reward_dist = -np.linalg.norm(vec)
        reward_ctrl = -np.square(a).sum()
        reward = reward_dist + reward_ctrl

        self.do_simulation(a, self.frame_skip)
        if self.render_mode == "human":
            self.render()

        ob = self._get_obs()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return (
            ob,
            reward,
            False,
            False,
            dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl),
        )

    def _sample_discrete_goal(self):
        goal_choice = np.random.choice(self._N)
        self.goal = self.goal_space[goal_choice]

    def reset_model(self):
        qpos = (
            self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq)
            + self.init_qpos
        )
        if self._continuous:
             if self._N > 0:
                 self._sample_discrete_goal()
             else:
                while True:
                    self.goal = self.np_random.uniform(low=-0.2, high=0.2, size=2)
                    if np.linalg.norm(self.goal) < 0.2:
                        break
        else:
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

            goal_ix = np.random.choice(8)
            self.goal = candidate_goals[goal_ix]

        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv
        )
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        theta = self.data.qpos.flat[:2]
        return np.concatenate(
            [
                np.cos(theta),
                np.sin(theta),
                self.data.qpos.flat[2:],
                self.data.qvel.flat[:2],
                self.get_body_com("fingertip") - self.get_body_com("target"),
            ]
        ).astype(np.float32)
    
    def get_goal_agnostic_obs(self):
        """returns an observation of the environment without goal info

        Returns:
            numpy.array: goal agnostic observation with shape (7,)
        """        
        return self._get_obs()[self.goal_mask]
    
    def insert_goals(self, goal_agnostic_obs):
        """Inserts goals into goal agnostic observation

        Args:
            goal_agnostic_obs (numpy.array): a goal agnostic observation of the environment
                with shape (7,)
            possible_goals (numpy.array): an array of all possible goals to be considered with shape
                (N, goal_dim) where N is the total number of goals to be considered and goal_dim
                is the dimension of an individual goal (i.e. 2: an x and y position)

        Returns:
            observations (numpy.array): A matrix of observations containing all possible goals. Has 
                shape (N, obs_dim)
        """ 
        GOAL_AGNOSTIC_OBS_DIM = 7
        OBS_DIM = 11
        num_goals, goal_dim = self.goal_space.shape
        goal_agnostic_obs = goal_agnostic_obs.reshape(1,GOAL_AGNOSTIC_OBS_DIM)
        goal_agnostic_obs = goal_agnostic_obs.repeat(num_goals, axis=0)

        observations = np.hstack([goal_agnostic_obs[:,:4], 
                                  self.goal_space, 
                                  goal_agnostic_obs[:,4:6], 
                                  self.get_body_com("fingertip")[:2]-self.goal_space, 
                                  goal_agnostic_obs[:,-1:]])
        return observations