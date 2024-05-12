import os
import torch
import torch.nn as nn
import numpy as np
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
import yaml

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


def train(config):
    '''
    - sample a batch of training tuples
    - add noise to the training tuples (using noise process)
    - run diffusion process on noised samples
    '''
    train_ds, val_ds = torch.utils.data.random_split(ExpertDemonstrations(config["demonstration_data"]), [0.95, 0.05])
    dataloader = DataLoader(train_ds,
                            batch_size=config["batch_size"],
                              shuffle=True, 
                              num_workers=config["num_workers"])
    
    val_dataloader = DataLoader(val_ds, 
                                batch_size=config["batch_size"],
                                shuffle=True,
                                num_workers=config["num_workers"])
    model = SharedAutonomy(obs_size=config["state_dim"]+config["action_dim"], config=config["copilot_architecture"]).to(config["device"])
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'])
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=64, device=config["device"], action_dim=config["action_dim"], state_dim=config["state_dim"])
    l = len(dataloader)
    training_loss = []
    validation_loss = []

    pbar = tqdm(range(config["epochs"]))
    steering_mse = []
    throttle_mse = []
    for epoch in pbar:
        steering_err = 0
        throttle_err = 0
        epoch_loss = 0
        for i, (trajectories) in enumerate(dataloader):
            trajectories = trajectories.to(config["device"])
            #images = torch.randn(4, 10)
            t = diffusion.sample_timesteps(trajectories.shape[0]).to(config["device"])
            x_t, noise = diffusion.noise_images(trajectories, t)

            predicted_noise = model(x_t, t)
            loss = mse(noise, predicted_noise)
            epoch_loss += loss.detach().cpu().item()

            # precict the correct actions
            #generated_actions = diffusion.sample(model, trajectories, gamma=0.2)
            #steering_err += mse(trajectories[:, -2], generated_actions[:, -2]).detach().cpu().item()
            #throttle_err += mse(trajectories[:, -1], generated_actions[:, -1]).detach().cpu().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())
        training_loss.append(epoch_loss/len(pbar))
        #steering_mse.append(steering_err / len(pbar))
        #throttle_mse.append(throttle_err / len(pbar))

        if epoch %10==0:
            # save the model and do sampling
            #sampled_images = diffusion.sample(model, trajectories)

            # evaluate on the validation set
            val_loss = 0
            model.eval()
            for trajectories in val_dataloader:
                trajectories = trajectories.to(config["device"])
                t = diffusion.sample_timesteps(trajectories.shape[0]).to(config["device"])
                x_t, noise = diffusion.noise_images(trajectories, t)
                
                generated_actions = diffusion.sample(model, trajectories, gamma=0.2)
                steering_err += mse(trajectories[:, -2], generated_actions[:, -2]).detach().cpu().item()
                throttle_err += mse(trajectories[:, -1], generated_actions[:, -1]).detach().cpu().item()

                predicted_noise = model(x_t, t)
                val_loss += mse(noise, predicted_noise).detach().cpu().item()
                #del trajectories
            validation_loss.append(val_loss)
            steering_mse.append(steering_err / len(val_dataloader))
            throttle_mse.append(throttle_err / len(val_dataloader))
            model.train()

            
            #save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
            torch.save(model.state_dict(), os.path.join("models", config["output_dir"], f"ckpt_" + str(epoch) +".pt"))
            plt.figure()
            plt.plot(np.log(training_loss))
            plt.plot([10*_ for _ in range(len(validation_loss))], np.log(validation_loss))
            plt.legend(['Training Loss', 'Validation Loss'])
            plt.xlabel("Epoch")
            plt.ylabel("Log Training Loss")
            plt.savefig("models" / Path(config["output_dir"]) / "training_loss.png")
            plt.close()
            plt.plot([10*_ for _ in range(len(validation_loss))], steering_mse)
            plt.plot([10*_ for _ in range(len(validation_loss))], throttle_mse)
            plt.legend(["Steering MSE", "Throttle MSE"])
            plt.savefig("models" / Path(config["output_dir"]) / "generative_loss.png")
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
    parser.add_argument('config_path')
    args = parser.parse_args()
    with open(args.config_path) as config_file:
        config = yaml.safe_load(config_file.read())
    if config["overwrite_data"]:
        os.makedirs("models/" + config["output_dir"], exist_ok=True)
    else:
        os.makedirs("models/" + config["output_dir"], exist_ok=False)

    with open("models/" + config["output_dir"] + '/config.yaml', 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    train(config)