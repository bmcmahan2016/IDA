from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import os

class ExpertDemonstrations(Dataset):
    """Expert Demonstrations in Lunar Lander w/o Goal Info."""

    def __init__(self, trajectories_file):
        """
        Arguments:
            trajectories_file (string): Path to the numpy file with expert 
            demonstrations
        """
        self.expert_demonstrations = np.load(trajectories_file)

    def __len__(self):
        return len(self.expert_demonstrations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        expert_demonstrations = self.expert_demonstrations[idx]
        return expert_demonstrations
    
if __name__ == '__main__':
    dataset = ExpertDemonstrations('/home/necl/code/lunar-lander/expert_trajectories_random_goal.npy')
    dataloader = DataLoader(dataset, batch_size=32,
                        shuffle=True, num_workers=0)
    breakpoint()