'''
code taken from https://github.com/dome272/Diffusion-Models-pytorch
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedAutonomy(nn.Module):
    def __init__(self, obs_size=10, time_dim=128, device="cpu"):
        super().__init__()

        # 4-layer MLP
        self.layer1 = nn.Linear(obs_size, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 128)
        self.layer4 = nn.Linear(128, obs_size)
        self.softplus = nn.Softplus()
        self.device = device
        self.time_dim = time_dim     
        self.pos_encoding = nn.Embedding(50, 128) 


    # def pos_encoding(self, t, channels):
    #     breakpoint()
    #     t = nn.Embedding(50, 128)(t)
    #     inv_freq = 1.0 / (
    #         10000
    #         ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
    #     )
    #     pos_enc_a = torch.sin(t * inv_freq)
    #     pos_enc_b = torch.cos(t.repeat(channels // 2) * inv_freq)
    #     pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
    #     return pos_enc

    def forward(self, x, t):
        '''
        The forward pass for the shared autonomy model consists of 
        a 4 layer MLP, with softplus activation, and pointwise multiplication
        of the time embedding
        '''
        # embed time info
        t = t.type(torch.long)
        t = self.pos_encoding(t)

        # pass through first layer
        x = self.layer1(x)
        x = torch.mul(x, t)
        x = nn.Softplus()(x)

        # pass through second layer
        x = self.layer2(x)
        x = torch.mul(x, t)
        x = nn.Softplus()(x)

        # pass through third layer
        x = self.layer3(x)
        x = torch.mul(x, t)
        x = nn.Softplus()(x)

        # pass through fourth layer
        x = self.layer4(x)
        #x = torch.mul(x, t)
        #x = nn.Softplus()(x)
        return x


# class SharedAutonomy(nn.Module):
#     def __init__(self, 
#                  obs_size=10, 
#                  config=dict(latent_dim=128, num_layers=4, layer_type="linear"),
#                  device="cuda:0"):
#         super().__init__()

#         # build architecture from config
#         if config["layer_type"] == "linear":
#             make_layer = nn.Linear
#         self._layers = []
#         latent_dim = config["latent_dim"]
#         for layer_num in range(config["num_layers"]-1):
#             if layer_num == 0:
#                 self._layers.append(nn.Linear(obs_size, latent_dim).to(device=device))
#             else:
#                 self._layers.append(nn.Linear(latent_dim, latent_dim).to(device=device))
#             self.add_module("layer_{}".format(layer_num), self._layers[-1])
#         self._layers.append(nn.Linear(latent_dim, obs_size).to(device=device))
#         self.add_module("layer_{}".format(layer_num+1), self._layers[-1])
#         # 4-layer MLP
#         self.softplus = nn.Softplus()
#         self.device = device
#         self.time_dim = latent_dim
#         self.pos_encoding = nn.Embedding(50, latent_dim).to(device=device)


#     # def pos_encoding(self, t, channels):
#     #     breakpoint()
#     #     t = nn.Embedding(50, 128)(t)
#     #     inv_freq = 1.0 / (
#     #         10000
#     #         ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
#     #     )
#     #     pos_enc_a = torch.sin(t * inv_freq)
#     #     pos_enc_b = torch.cos(t.repeat(channels // 2) * inv_freq)
#     #     pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
#     #     return pos_enc

#     def forward(self, x, t):
#         '''
#         The forward pass for the shared autonomy model consists of 
#         a 4 layer MLP, with softplus activation, and pointwise multiplication
#         of the time embedding
#         '''
#         # embed time info
#         t = t.type(torch.long)
#         t = self.pos_encoding(t) 

#         # pass through all but final layer
#         for layer in self._layers[:-1]:
#             x = layer(x)
#             x = torch.mul(x, t)
#             x = self.softplus(x)

#         # pass through final layer
#         x = self._layers[-1](x)
#         return x