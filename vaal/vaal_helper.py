import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torchvision import transforms
import numpy as np
import random
import cv2


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class VAE(nn.Module):
    """Encoder-Decoder architecture for both WAE-MMD and WAE-GAN."""

    def __init__(self, z_dim=256, nc=3):
        super(VAE, self).__init__()
        self.z_dim = z_dim
        self.nc = nc
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 64, 4, 2, 1, bias=True),  # B,  64, 128
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=True),  # B,  128, 64
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=True),  # B,  256, 32
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=True),  # B,  512,  16
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 1024, 4, 2, 1, bias=True),  # B, 1024, 8, 8
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            View((-1, 1024 * 8 * 8)),  # B, 1024*4*4
        )

        self.fc_mu = nn.Linear(1024 * 8 * 8, z_dim)  # B, z_dim
        self.fc_logvar = nn.Linear(1024 * 8 * 8, z_dim)  # B, z_dim
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 1024 * 8 * 8),  # B, 1024*8*8
            View((-1, 1024, 8, 8)),  # B, 1024,  8,  8
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=True),  # B,  512, 16
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=True),  # B,  256, 32
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=True),  # B,  128,  64
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=True),  # B,  64, 128
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=True),  # B,  32,  256
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, nc, 1),  # B,   nc, 256, 256
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            try:
                for m in self._modules[block]:
                    kaiming_init(m)
            except:
                kaiming_init(block)

    def forward(self, imgs):
        x = []
        for img in imgs:
            img = F.interpolate(img.unsqueeze(0), (256, 256))
            x.append(img * 255)
        x = torch.cat(x)
        z = self._encode(x)
        mu, logvar = self.fc_mu(z), self.fc_logvar(z)
        z = self.reparameterize(mu, logvar)
        x_recon = self._decode(z)
        # if random.random() < 0.05:
        #     f = random.randint(1, 10)
        #     input_tensor = x_recon[0].squeeze().data
        #     # 从[0,1]转化为[0,255]，再从CHW转为HWC，最后转为cv2
        #     input_tensor = input_tensor.clamp_(0, 255).permute(1, 2, 0).type(
        #         torch.uint8).cpu().numpy()
        #     # RGB转BRG
        #     input_tensor = cv2.cvtColor(input_tensor, cv2.COLOR_RGB2BGR)
        #     cv2.imwrite('vis/{}_vae.jpg'.format(f), input_tensor)
        #     input_tensor = x[0].squeeze().data
        #     # 从[0,1]转化为[0,255]，再从CHW转为HWC，最后转为cv2
        #     input_tensor = input_tensor.clamp_(0, 255).permute(1, 2, 0).type(
        #         torch.uint8).cpu().numpy()
        #     # RGB转BRG
        #     input_tensor = cv2.cvtColor(input_tensor, cv2.COLOR_RGB2BGR)
        #     cv2.imwrite('vis/{}_ori.jpg'.format(f), input_tensor)
        return x_recon, z, mu, logvar

    def reparameterize(self, mu, logvar):
        stds = (0.5 * logvar).exp()
        epsilon = torch.randn(*mu.size())
        if mu.is_cuda:
            stds, epsilon = stds.cuda(), epsilon.cuda()
        latents = epsilon * stds + mu
        return latents

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)


class Discriminator(nn.Module):
    """Adversary architecture(Discriminator) for WAE-GAN."""

    def __init__(self, z_dim=256):
        super(Discriminator, self).__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, z):
        return self.net(z)


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()


mse_loss = nn.MSELoss(reduction='mean')


def vae_loss(imgs, recon, mu, logvar, beta):
    x = []
    for img in imgs:
        img = F.interpolate(img.unsqueeze(0), (256, 256))
        x.append(img * 255)
    x = torch.cat(x)
    MSE = mse_loss(recon, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD = KLD * beta
    return 0.1 * MSE + KLD


bce_loss = nn.BCELoss()


class AdversarySampler:
    def __init__(self, budget):
        self.budget = budget

    def sample(self, vae, discriminator, dataloader):
        all_preds = []
        all_indices = []

        for i, (images, _) in enumerate(dataloader):
            images = list(image.cuda() for image in images)
            x = []
            for img in images:
                img = F.interpolate(img.unsqueeze(0), (256, 256))
                x.append(img)
            x = torch.cat(x)
            with torch.no_grad():
                _, _, mu, _ = vae(x)
                preds = discriminator(mu)
            all_preds.extend(preds)
            all_indices.append(i)

        all_preds = torch.stack(all_preds)
        all_preds = all_preds.view(-1)
        # need to multiply by -1 to be able to use torch.topk
        all_preds *= -1

        # select the points which the discriminator things are the most likely to be unlabeled
        _, querry_indices = torch.topk(all_preds, int(self.budget))
        querry_pool_indices = np.asarray(all_indices)[querry_indices.cpu().numpy()]

        return querry_pool_indices


def sample_for_labeling(vae, discriminator, unlabeled_dataloader, budget):
    sampler = AdversarySampler(budget)
    querry_indices = sampler.sample(vae, discriminator, unlabeled_dataloader)
    return querry_indices
