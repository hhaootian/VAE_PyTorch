#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super(Encoder, self).__init__()
        modules = []
        feat_in = input_dim

        for dim in hidden_dims:
            modules.append(nn.Sequential(
                nn.Linear(feat_in, dim),
                nn.LeakyReLU()
            ))
            feat_in = dim

        self.model = nn.Sequential(*modules)
        self.mu = nn.Linear(feat_in, latent_dim)
        self.log_var = nn.Linear(feat_in, latent_dim)

    def forward(self, x):
        x = self.model(x)
        mu = self.mu(x)
        log_var = self.log_var(x)
        return mu, log_var


class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super(Decoder, self).__init__()
        modules = []
        feat_in = latent_dim

        for dim in hidden_dims:
            modules.append(nn.Sequential(
                nn.Linear(feat_in, dim),
                nn.LeakyReLU()
            ))
            feat_in = dim

        modules.append(
            nn.Sequential(
                nn.Linear(feat_in, input_dim),
                nn.Tanh()
            )
        )

        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)


class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        mu, log_var = self.encoder(x)

        # reparameterization
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = eps * std + mu

        x_hat = self.decoder(z)
        return x_hat, mu, log_var
