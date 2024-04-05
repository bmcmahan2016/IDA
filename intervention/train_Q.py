'''
Trains a Q function to be used to determine
intervention values.

Author: Brandon McMahan
Date: March 21, 2024
'''
import pfrl
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

class TupleDataset(Dataset):
    def __init__(self, tuples_path):
        self.tuples = np.load(tuples_path).astype(np.float32)

    def __len__(self):
        return self.tuples.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        tuples = self.tuples[idx]
        return tuples

def make_Q(obs_size, action_size):
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
    return q_func

data = DataLoader(TupleDataset("/home/necl/code/IDA/intervention/lunar_lander_SR_0.4015318706997274_policy_rollouts.npy"),
                  batch_size=512,
                  shuffle=True,
                  num_workers=8
)
Q = make_Q(9, 2).cuda()
Q.load_state_dict(torch.load("/home/necl/code/IDA/experts/runs/lunar_lander/901_reward_1169.172599966153/q_func1.pt"))
optimizer = torch.optim.AdamW(Q.parameters(), lr=1e-10)
gamma = 0.99
loss_fn = nn.MSELoss()
loss_hist = []
for _ in range(20):
    total_loss = 0
    for i, tuples in enumerate(tqdm(data)):
        tuples = tuples.cuda()
        # unpack flattened tuple
        state = tuples[:, :9]
        action = tuples[:, 9:11]
        reward = tuples[:, 11:12]
        done = tuples[:, 12:13]
        next_state = tuples[:, 13:22]
        next_action = tuples[:, 22:24]


        target = reward + (1-done)*gamma*Q([next_state, next_action])
        pred = Q([state, action])
        loss = loss_fn(target, pred)
        total_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    loss_hist.append(total_loss.item()/len(data))


torch.save(Q.state_dict(), "Q_epoch_" + str(_))
plt.figure()
plt.plot(loss_hist)
plt.savefig("Q_Training_Loss_control.png")
plt.close()
