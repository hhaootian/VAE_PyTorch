#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch


def loss_function(x, x_hat, mu, log_var):
    reconstruction_loss = torch.nn.functional.mse_loss(x_hat, x)
    kl_divergence = -0.5 * torch.sum(
        1 + log_var - mu**2 - log_var.exp()
    )

    return (
        reconstruction_loss + kl_divergence,
        reconstruction_loss.item(),
        kl_divergence.item()
    )
