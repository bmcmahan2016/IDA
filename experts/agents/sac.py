'''
Creates an SAC agent using the PFRL Repo

Author: Brandon McMahan
Date: February 8, 2024
'''
import numpy as np
import torch
import torch.nn as nn
import torch.distributions as distributions
import pfrl
from pfrl.nn.lmbda import Lambda
from pfrl import replay_buffers

def make_SAC(obs_size, action_size, action_low, action_high, gpu=0, lr=3e-4):

    def make_q_func_with_optimizer():
        q_func = nn.Sequential(
            pfrl.nn.ConcatObsAndAction(),
            nn.Linear(obs_size + action_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        torch.nn.init.xavier_uniform_(q_func[1].weight)
        torch.nn.init.xavier_uniform_(q_func[3].weight)
        torch.nn.init.xavier_uniform_(q_func[5].weight)
        q_func_optimizer = torch.optim.Adam(q_func.parameters(), lr=lr)
        return q_func, q_func_optimizer

    def burnin_action_func():
        """Select random actions until model is updated one or more times."""
        return np.random.uniform(action_low, action_high).astype(np.float32)

    def squashed_diagonal_gaussian_head(x):
        assert x.shape[-1] == action_size * 2
        mean, log_scale = torch.chunk(x, 2, dim=1)
        log_scale = torch.clamp(log_scale, -20.0, 2.0)
        var = torch.exp(log_scale * 2)
        base_distribution = distributions.Independent(
            distributions.Normal(loc=mean, scale=torch.sqrt(var)), 1
        )
        # cache_size=1 is required for numerical stability
        return distributions.transformed_distribution.TransformedDistribution(
            base_distribution, [distributions.transforms.TanhTransform(cache_size=1)]
        )

    policy = nn.Sequential(
        nn.Linear(obs_size, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, action_size * 2),
        Lambda(squashed_diagonal_gaussian_head),
    )
    torch.nn.init.xavier_uniform_(policy[0].weight)
    torch.nn.init.xavier_uniform_(policy[2].weight)
    torch.nn.init.xavier_uniform_(policy[4].weight, gain=1)
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)

    q_func1, q_func1_optimizer = make_q_func_with_optimizer()
    q_func2, q_func2_optimizer = make_q_func_with_optimizer()   

    
    rbuf = replay_buffers.ReplayBuffer(10**6)


    agent = pfrl.agents.SoftActorCritic(
        policy,
        q_func1,
        q_func2,
        policy_optimizer,
        q_func1_optimizer,
        q_func2_optimizer,
        rbuf,
        gamma=0.99,
        replay_start_size=10000,
        gpu=gpu,
        minibatch_size=64,
        burnin_action_func=burnin_action_func,
        entropy_target=-action_size,
        temperature_optimizer_lr=3e-4,
    )
    return agent






