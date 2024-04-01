'''
code taken from https://github.com/dome272/Diffusion-Models-pytorch
'''
import torch

class Diffusion:
    '''
    Diffusion model for shared autonomy
    
    Input is a state-conditioned action. The action is subject
    to noise but not the state.
    '''
    def __init__(self, 
                 action_dim=2,
                 state_dim=8,
                 noise_steps=50, 
                 beta_start=1e-4, 
                 beta_end=0.26, 
                 img_size=256, 
                 device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device
        self.action_dim=action_dim
        self.state_dim=state_dim

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        '''
        adds noise to a 1D feature vector
        '''
        states = x[:,:-self.action_dim]
        actions = x[:,-self.action_dim:]
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None]
        Ɛ = torch.randn_like(actions)
        
        actions = sqrt_alpha_hat * actions + sqrt_one_minus_alpha_hat * Ɛ
        x = torch.hstack([states, actions])
        Ɛ = torch.hstack([torch.zeros_like(states), Ɛ])
        return x, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, human_actions, gamma=1.0):
        '''
        '''
        n = human_actions.shape[0]
        #logging.info(f"Applying diffusion process to {n} human actions....")
        model.eval()
        with torch.no_grad():
            # human_actions should be batch_size x feature_size (e.g. 2048, 10)
            # run forward diffusion process on human actions
            if gamma==0:
                return human_actions

            num_t_steps = int(50 * gamma) - 1
            t = torch.ones((human_actions.shape[0]))*num_t_steps
            t = t.int()
            noised_human_actions, noise = self.noise_images(human_actions, t)
            noised_human_actions = noised_human_actions.to(self.device)
            x = noised_human_actions

            for i in reversed(range(1, num_t_steps)):
                t = (torch.ones(n) * i).long().to(self.device)
                # perform one reverse diffusion step
                predicted_noise = model(x, t)
                states = x[:,:-self.action_dim]
                actions = x[:,-self.action_dim:]
                alpha = self.alpha[t][:, None]
                alpha_hat = self.alpha_hat[t][:, None]
                beta = self.beta[t][:, None]
                if i > 1:
                    noise = torch.randn_like(actions)
                else:
                    noise = torch.zeros_like(actions)
                actions = 1 / torch.sqrt(alpha) * (actions - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise[:,-self.action_dim:]) + torch.sqrt(beta) * noise
                x = torch.hstack([states, actions])
        model.train()
        # x = (x.clamp(-1, 1) + 1) / 2
        # x = (x * 255).type(torch.uint8)
        return x

