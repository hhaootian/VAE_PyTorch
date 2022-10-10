#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import mdtraj as md
from sklearn.preprocessing import MinMaxScaler
import torch
from vae import Encoder, Decoder, VAE
from utils import loss_function


# load data
traj_dir = "../adk/trajs/1ake_01_200ps_npt.dcd"
top_dir = "../adk/pdb/1ake.prmtop"

trajs = md.load(traj_dir, top=top_dir)
trajs = trajs.atom_slice(trajs.topology.select_atom_indices("heavy"))
raw_coors = trajs.xyz
raw_coors = raw_coors.reshape(trajs.n_frames, -1)
n_dims = raw_coors.shape[1]

scaler = MinMaxScaler()
coors = scaler.fit_transform(raw_coors)

# get device
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

train_dataset = torch.utils.data.DataLoader(
    coors, batch_size=16,
    shuffle=True, 
    pin_memory=torch.cuda.is_available()
)

# build model
encoder = Encoder(n_dims, [128, 64, 32], 2)
decoder = Decoder(n_dims, [32, 64, 128], 2)
vae = VAE(encoder, decoder).to(device)
vae.train()

optimizer = torch.optim.Adam(vae.parameters(), lr=0.001)

# train
for epoch in range(100):
    rl_loss = kl_loss = 0
    for x in train_dataset:
        x = x.view(-1, n_dims).to(device)
        optimizer.zero_grad()
        # calculate loss
        x_hat, mu, log_var = vae(x)
        loss, rl_val, kl_val = loss_function(x, x_hat, mu, log_var)
        rl_loss += rl_val
        kl_loss += kl_val
        loss.backward()
        optimizer.step()
    print(f"epoch: {epoch}  rl_loss: {rl_loss}  kl_loss: {kl_loss}")

# evaluation
vae.eval()

test_dataset = torch.utils.data.DataLoader(
    coors, batch_size=16,
    shuffle=True, 
    pin_memory=torch.cuda.is_available()
)

preds = []
for x in test_dataset:
    x_hat, mu, log_var = vae(x.view(-1, n_dims).to(device))
    x_hat = x_hat.cpu().detach().numpy().tolist()
    preds += x_hat

pred_coors = scaler.inverse_transform(preds)

# plot latent space
latents = []
for x in test_dataset:
    mu, log_var = encoder(x.view(-1, n_dims).to(device))
    mu = mu.cpu().detach().numpy().tolist()
    latents += mu
