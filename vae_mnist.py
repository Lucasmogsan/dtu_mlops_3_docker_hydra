"""Adapted from https://github.com/Jackson-Kang/Pytorch-VAE-tutorial/blob/master/01_Variational_AutoEncoder.ipynb.

A simple implementation of Gaussian MLP Encoder and Decoder trained on MNIST
"""
import os
import logging

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from model import Decoder, Encoder, Model
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.utils import save_image

# For the hyperparameters config
import hydra
from omegaconf import OmegaConf

# Logger for this file. log.info() used instead of print() to save the output to a file.
log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="default_config.yaml")
def main(cfg):
    
    # Model Hyperparameters
    log.info(f"configuration: \n {OmegaConf.to_yaml(cfg)}")  # Just for printing the config
    hparams = cfg.hparam_experiment

    torch.manual_seed(hparams["seed"])  # Set seed for generating random numbers (for reproducibility)
    device = torch.device("cuda" if hparams["cuda"] else "cpu") if torch.cuda.is_available() else "cpu"

    # Data loading
    mnist_transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = MNIST(hparams["dataset_path"], transform=mnist_transform, train=True, download=True)
    test_dataset = MNIST(hparams["dataset_path"], transform=mnist_transform, train=False, download=True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=hparams["batch_size"], shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=hparams["batch_size"], shuffle=False)

    encoder = Encoder(
        input_dim=hparams["x_dim"],
        hidden_dim=hparams["hidden_dim"],
        latent_dim=hparams["latent_dim"]
    )
    decoder = Decoder(
        latent_dim=hparams["latent_dim"],
        hidden_dim=hparams["hidden_dim"],
        output_dim=hparams["x_dim"])

    model = Model(encoder=encoder, decoder=decoder).to(device)


    def loss_function(x, x_hat, mean, log_var):
        """Elbo loss function."""
        reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction="sum")
        kld = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        return reproduction_loss + kld


    optimizer = Adam(model.parameters(), lr=hparams["learning_rate"])


    log.info("Start training VAE...")
    model.train()
    for epoch in range(hparams["epochs"]):
        overall_loss = 0
        for batch_idx, (x, _) in enumerate(train_loader):
            if batch_idx % 100 == 0:
                log.info(batch_idx)
            x = x.view(hparams["batch_size"], hparams["x_dim"])
            x = x.to(device)

            optimizer.zero_grad()

            x_hat, mean, log_var = model(x)
            loss = loss_function(x, x_hat, mean, log_var)

            overall_loss += loss.item()

            loss.backward()
            optimizer.step()
        log.info(f"Epoch {epoch+1} complete!,  Average Loss: {overall_loss / (batch_idx*hparams['batch_size'])}")
    log.info("Finish!!")

    # save weights
    torch.save(model, f"{os.getcwd()}/trained_model.pt")

    # Generate reconstructions
    model.eval()
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(test_loader):
            if batch_idx % 100 == 0:
                log.info(batch_idx)
            x = x.view(hparams["batch_size"], hparams["x_dim"])
            x = x.to(device)
            x_hat, _, _ = model(x)
            break

    save_image(x.view(hparams["batch_size"], 1, 28, 28), "orig_data.png")
    save_image(x_hat.view(hparams["batch_size"], 1, 28, 28), "reconstructions.png")

    # Generate samples
    with torch.no_grad():
        noise = torch.randn(hparams["batch_size"], hparams["latent_dim"]).to(device)
        generated_images = decoder(noise)

    save_image(generated_images.view(hparams["batch_size"], 1, 28, 28), "generated_sample.png")



if __name__ == "__main__":
    main()