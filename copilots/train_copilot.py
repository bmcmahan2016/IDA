import os
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
from modules import SharedAutonomy
import logging
from copilots.diffusion import Diffusion
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from dataset import ExpertDemonstrations
from pathlib import Path

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


def train(args):
    '''
    - sample a batch of training tuples
    - add noise to the training tuples (using noise process)
    - run diffusion process on noised samples
    '''
    dataloader = DataLoader(ExpertDemonstrations(args.demonstration_data),
                            batch_size=args.batch_size, shuffle=True, num_workers=8)
    model = SharedAutonomy(obs_size=args.state_dim+args.action_dim).to(args.device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=64, device=args.device, action_dim=args.action_dim, state_dim=args.state_dim)
    l = len(dataloader)
    training_loss = []

    for epoch in range(args.epochs):
        epoch_loss = 0
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, (trajectories) in enumerate(pbar):
            trajectories = trajectories.to(args.device)
            #breakpoint()
            #images = torch.randn(4, 10)
            t = diffusion.sample_timesteps(trajectories.shape[0]).to(args.device)
            x_t, noise = diffusion.noise_images(trajectories, t)

            predicted_noise = model(x_t, t)
            loss = mse(noise, predicted_noise)
            epoch_loss += loss.detach().cpu().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())
        training_loss.append(epoch_loss/len(pbar))

        if epoch %10==0:
            # save the model and do sampling
            sampled_images = diffusion.sample(model, trajectories)
            
            #save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
            torch.save(model.state_dict(), os.path.join("models", args.output_dir, f"ckpt.pt"))
            plt.figure()
            plt.plot(training_loss)
            plt.xlabel("Epoch")
            plt.ylabel("Training Loss")
            plt.savefig("models" / Path(args.output_dir) / "training_loss.png")
            plt.close()


def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_Shared_Autonomy"
    args.epochs = 500
    args.batch_size = 8
    args.image_size = 64
    args.dataset_path = r"/home/necl/code/IDA/expert_trajectories_cartpole.npy"
    args.device = "cuda:0"
    args.lr = 1e-3
    train()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir')
    parser.add_argument('--state_dim', type=int)
    parser.add_argument('--action_dim', type=int)
    parser.add_argument('--demonstration_data')
    parser.add_argument('--epochs', default=500)
    parser.add_argument('--batch_size', default=40*512)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--lr', default=1e-3)
    args = parser.parse_args()
    train(args)

