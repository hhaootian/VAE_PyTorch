#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torchvision
from vae import Encoder, Decoder, VAE
from utils import loss_function


# get device
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# load data
transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(0.5, 1)
    ]
)
mnist = torchvision.datasets.MNIST('./', download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(
    mnist, batch_size=64,
    shuffle=True, 
    pin_memory=torch.cuda.is_available()
)

# build model
encoder = Encoder(28 * 28, [128, 64, 32], 2)
decoder = Decoder(28 * 28, [32, 64, 128], 2)
vae = VAE(encoder, decoder).to(device)
vae.train()

optimizer = torch.optim.Adam(vae.parameters(), lr=0.001)

# train
for epoch in range(100):
    overall_loss = 0
    for data in dataloader:
        x, _ = data
        x = x.view(-1, 28 * 28).to(device)

        optimizer.zero_grad()

        # calculate loss
        x_hat, mu, log_var = vae(x)
        loss = loss_function(x, x_hat, mu, log_var)
        overall_loss += loss.item()

        loss.backward()
        optimizer.step()

    print(epoch, overall_loss)
