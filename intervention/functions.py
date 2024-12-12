'''
Generates a Goal Agnostic Intervention function
to determine if a copilot should intervent

Author: Brandon McMahan
Date: February 28, 2024
'''
import torch
import numpy as np

class InterventionFunction(object):
    def __init__(self, Q_intervention, env, num_goals=1, discount=0.0, margin=0.99, disable=False, goal_agnostic=False):
        MAXIMUM_SEQ_LENGTH = 1000
        self._NUM_GOALS = num_goals
        self._OBS_DIM = env.env.observation_space.high.shape[0]
        self._ACTION_DIM = env.env.action_space.low.shape[0]
        self._disable=disable  # turns off intervention
        self._goal_agnostic = goal_agnostic

        # create tensors and place them on GPU
        self._margin = margin
        self._observation_batch = torch.zeros((MAXIMUM_SEQ_LENGTH, self._NUM_GOALS, self._OBS_DIM)).cuda()
        self._action_batch = torch.zeros((MAXIMUM_SEQ_LENGTH, self._NUM_GOALS, self._ACTION_DIM)).cuda()
        discount_factor = np.array([discount**x for x in reversed(range(MAXIMUM_SEQ_LENGTH))])
        discount_factor = np.expand_dims(np.expand_dims(discount_factor, 1), 1)
        self._discount_factor = torch.from_numpy(discount_factor).cuda()
        self._ix = 0
        self._Q = Q_intervention

        self._env = env

    def reset(self):
        '''resets observations and actions for new episode'''
        self._observation_batch[:] = 0
        self._action_batch[:] = 0
        self._ix = 0

    def behavior_policy(self, goal_agnostic_obs, pilot_action, copilot_action):
        if self._disable:
            return copilot_action, 0
        if self._goal_agnostic:
            self._observation_batch[self._ix] = torch.tensor(self._env.env.insert_agnostic_goals(goal_agnostic_obs))
        else:
            self._observation_batch[self._ix] = torch.tensor(self._env.env.insert_goals(goal_agnostic_obs))
        pilot_intervention_score = self.compute_intervention(goal_agnostic_obs, pilot_action)
        copilot_intervention_score = self.compute_intervention(goal_agnostic_obs, copilot_action)
        copilot_advantage = torch.sum(torch.sign(copilot_intervention_score-pilot_intervention_score)) / len(copilot_intervention_score)
        copilot_advantage = copilot_advantage.item()
        if copilot_advantage > self._margin:
            behavioral_action = copilot_action
        else:
            behavioral_action = pilot_action
        self._action_batch[self._ix] = torch.tensor(behavioral_action)
        self._ix += 1
        return behavioral_action, copilot_advantage

    def compute_intervention(self, goal_agnostic_obs, action):
        """_summary_

        PSEUDOCODE
            insert all goals into the goal_agnostic_obs
            goal_agnostic_obs should now be (trajectory_len, num_goals, obs_dim)

            actions should be (trajectory_len, action_dim)
            actions need to be broadcast to (trajectory_len, num_goals, action_dim)

            advantage_values should initially be (trajectory_len, num_goals, 1)
            we want to first average over entire trajectory for each goal
                sum over axis=0
                advantage_values should now be (num_goals, 1)
            now we can average the advantage values

        Args:
            goal_agnostic_traj (numpy.array): should have shape (trajectory_len, obs_dim)
            action_traj (numpy.array): should have shape (trajectory_len, action_dim)

        Returns:
            _type_: _description_
        """       
        # casts lists to arrays

        action = np.expand_dims(action, 0)
        action = np.repeat(action, self._NUM_GOALS, axis=0)
        self._action_batch[self._ix] = torch.tensor(action)

        # # put back on torch
        advantage_values = self._Q([self._observation_batch[:self._ix+1], self._action_batch[:self._ix+1]])
        # applies discounting to advantage values
        advantage_values = self._discount_factor[-(1+self._ix):] * advantage_values
        traj_adv_values = torch.sum(advantage_values, dim=0)
        return traj_adv_values