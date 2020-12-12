import torchvision.transforms.functional as F
import detection.transforms as T
import random


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
