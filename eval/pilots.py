import numpy as np

class SurrogatePilot(object):
    def __init__(self, 
                 expert, 
                 env,
                 corruption_type,
                 corruption_prob = 0.1,
                 restore_prob = 0.1):
        self._expert = expert
        self._corruption_type = corruption_type
        self._corruption_prob = corruption_prob
        self._restore_prob = restore_prob
        self._env = env

        self._last_action = None
        self._corruption_on = False

    def reset(self):
        self._prev_action = None
        self._corruption_on = False

    def _is_corrupted(self):
        # return True and possibly turn corruption off
        if self._corruption_on:
            if np.random.rand() < self._restore_prob:
                self._corruption_on = False
            return True
        
        # return False and possibly turn corruption on
        else:
            if np.random.rand() < self._corruption_prob:
                self._corruption_on = True
            return False

    def act(self, observation):
        with self._expert.eval_mode():

            # default expert action
            action = self._expert.act(observation)
            if self._corruption_type=='none':
                self._prev_action = action
                return action, False
            # no-op and random corruption are performed always if specified
            if self._corruption_type=="no-op":
                action = np.zeros_like(self._expert.act(observation))
                return action, False
            if self._corruption_type=="random":
                action = self._env.action_space.sample()
                return action, False
        
            # corrupt with noise or lag if corruption is on
            is_corrupted = self._is_corrupted()
            if is_corrupted:
                if self._corruption_type=='noise':
                    action = self._env.action_space.sample()
                if self._corruption_type=='lag':
                    if self._prev_action is None:
                        action = self._expert.act(observation)
                    else:
                        action = self._prev_action

        # return action and store previous
        self._prev_action = action
        return action, is_corrupted
