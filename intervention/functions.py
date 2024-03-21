'''
Generates a Goal Agnostic Intervention function
to determine if a copilot should intervent

Author: Brandon McMahan
Date: February 28, 2024
'''
import torch
import numpy as np


def make_intervetion_function(Q_intervention, env):
    '''
    Returns a goal agnostic function by considering all goals over
    the goal space. 

    Q_intervention is an expert Q function for the environment

    goal_space is a tensor of candidate goals sampled accross the goal space
        goal_space has shape (N, goal_dim)
        where N is the number of possible goals in the environment
        and goal_dim is the dimensionality of the goal
    '''
    
    def intervention(goal_agnostic_obs, action):
        breakpoint()
        """Computes the goal agnostic advantage of an action

        Args:
            goal_agnostic_obs (torch.tensor): observation of the environment with goal
                information removed. Has shape [obs_dim]
            action (numpy.array): action in the environment has shape (action_dim,)

        Returns:
            _type_: _description_
        """        
        action_dim = len(action)
        # generates a batch of observations with all possible goals
        observation_batch = env.insert_goals(goal_agnostic_obs.numpy())
        num_goals = len(observation_batch)
        
        # broadcasts action to observation_batch
        action_batch = action.reshape(1,action_dim)
        action_batch = action_batch.repeat(num_goals, axis=0)

        # put back on torch
        observation_batch = torch.from_numpy(observation_batch).cuda().float()
        action_batch = torch.from_numpy(action_batch).cuda().float()
        advantage_value = torch.mean(Q_intervention([observation_batch, action_batch])).item()
        return advantage_value
    return intervention


def make_trajectory_intervetion_function(Q_intervention, env, discount=0.0):
    '''
    Returns a goal agnostic function by considering all goals over
    the goal space for the current trajectory. 

    Q_intervention is an expert Q function for the environment

    goal_space is a tensor of candidate goals sampled accross the goal space
        goal_space has shape (N, goal_dim)
        where N is the number of possible goals in the environment
        and goal_dim is the dimensionality of the goal
    '''
    
    def intervention(goal_agnostic_traj, action_traj):
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
        action_traj = np.array(action_traj)
        goal_agnostic_traj = np.array(goal_agnostic_traj)

        traj_len, action_dim = action_traj.shape
        # generates a batch of observations with all possible goals
        observation_batch = np.array(list(map(env.env.insert_goals, goal_agnostic_traj)))
        traj_len, num_goals, obs_dim = observation_batch.shape
        
        # # broadcasts action to observation_batch
        action_batch = np.expand_dims(action_traj, 1)
        action_batch = np.repeat(action_batch, num_goals, axis=1)
        # action_batch = action.reshape(1,action_dim)
        # action_batch = action_batch.repeat(num_goals, axis=0)

        # # put back on torch
        observation_batch = torch.from_numpy(observation_batch).cuda().float()
        action_batch = torch.from_numpy(action_batch).cuda().float()
        advantage_values = Q_intervention([observation_batch, action_batch])
        # applies discounting to advantage values
        discount_factor = np.array([discount**x for x in reversed(range(traj_len))])
        discount_factor = np.expand_dims(np.expand_dims(discount_factor, 1), 1)
        discount_factor = torch.from_numpy(discount_factor).float().cuda()
        advantage_values = discount_factor * advantage_values
        traj_adv_values = torch.sum(advantage_values, dim=0)
        avg_adv_value = torch.mean(traj_adv_values).item()
        # return advantage_value
        #return avg_adv_value
        return traj_adv_values
    return intervention
