import torchvision.transforms.functional as F
import detection.transforms as T
import random
import numpy as np
import torch


def HorizontalFlip(image, bbox):
    image = F.to_tensor(image)
    height, width = image.shape[-2:]
    image = image.flip(-1)
    bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
    return image, bbox


def ColorSwap(image):
    perms = ((0, 1, 2), (0, 2, 1), (1, 0, 2),
             (1, 2, 0), (2, 0, 1), (2, 1, 0))
    swap = perms[random.randint(0, len(perms) - 1)]
    image = F.to_tensor(image)
    image = image[swap, :, :]
    return image


def ColorAdjust(image):
    image = F.adjust_brightness(image, 3)
    image = F.adjust_contrast(image, 3)
    image = F.adjust_saturation(image, 3)
    return F.to_tensor(image)


def GaussianNoise(image, std=1):
    image = image + np.random.normal(0.0, std, image.shape)
    image = np.clip(image, 0, 255)
    return F.to_tensor(image)


def SaltPepperNoise(image, prob):
    image = F.to_tensor(image)
    noise = torch.rand(image.size())
    salt = torch.max(image)
    pepper = torch.min(image)
    image[noise < prob / 2] = salt
    image[noise > 1 - prob / 2] = pepper
    return image
