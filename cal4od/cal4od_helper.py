import torchvision.transforms.functional as F
import detection.transforms as T
import random
import numpy as np
import torch
import PIL
from PIL import Image, ImageDraw


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


def ColorAdjust(image, factor):
    image = F.adjust_brightness(image, factor)
    image = F.adjust_contrast(image, factor)
    image = F.adjust_saturation(image, factor)
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


def cutout(image, boxes, labels, fill_val=0, bbox_remove_thres=0.4, bbox_min_thres=0.1):
    '''
        Cutout augmentation
        image: A PIL image
        boxes: bounding boxes, a tensor of dimensions (#objects, 4)
        labels: labels of object, a tensor of dimensions (#objects)
        fill_val: Value filled in cut out
        bbox_remove_thres: Theshold to remove bbox cut by cutout

        Out: new image, new_boxes, new_labels
    '''
    if type(image) == PIL.Image.Image:
        image = F.to_tensor(image)
    original_h = image.size(1)
    original_w = image.size(2)
    original_channel = image.size(0)

    count = 0
    for _ in range(50):
        # Random cutout size: [0.15, 0.5] of original dimension
        cutout_size_h = random.uniform(0.05 * original_h, 0.2 * original_h)
        cutout_size_w = random.uniform(0.05 * original_w, 0.2 * original_w)

        # Random position for cutout
        left = random.uniform(0, original_w - cutout_size_w)
        right = left + cutout_size_w
        top = random.uniform(0, original_h - cutout_size_h)
        bottom = top + cutout_size_h
        cutout = torch.FloatTensor([int(left), int(top), int(right), int(bottom)]).cuda()

        # Calculate intersect between cutout and bounding boxes
        overlap_size = intersect(cutout.unsqueeze(0), boxes)
        area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        ratio = overlap_size / area_boxes
        # If all boxes have Iou greater than bbox_remove_thres, try again
        if ratio.max().item() > bbox_remove_thres or ratio.max().item() < bbox_min_thres:
            continue

        cutout_arr = torch.full((original_channel, int(bottom) - int(top), int(right) - int(left)), fill_val)
        image[:, int(top):int(bottom), int(left):int(right)] = cutout_arr
        count += 1
        if count >= 2:
            break
    # draw_PIL_image(image, boxes, labels)
    return image


def intersect(boxes1, boxes2):
    '''
        Find intersection of every box combination between two sets of box
        boxes1: bounding boxes 1, a tensor of dimensions (n1, 4)
        boxes2: bounding boxes 2, a tensor of dimensions (n2, 4)

        Out: Intersection each of boxes1 with respect to each of boxes2,
             a tensor of dimensions (n1, n2)
    '''
    n1 = boxes1.size(0)
    n2 = boxes2.size(0)
    max_xy = torch.min(boxes1[:, 2:].unsqueeze(1).expand(n1, n2, 2),
                       boxes2[:, 2:].unsqueeze(0).expand(n1, n2, 2))

    min_xy = torch.max(boxes1[:, :2].unsqueeze(1).expand(n1, n2, 2),
                       boxes2[:, :2].unsqueeze(0).expand(n1, n2, 2))
    inter = torch.clamp(max_xy - min_xy, min=0)  # (n1, n2, 2)
    return inter[:, :, 0] * inter[:, :, 1]  # (n1, n2)


def draw_PIL_image(image, boxes, labels, name):
    '''
        Draw PIL image
        image: A PIL image
        labels: A tensor of dimensions (#objects,)
        boxes: A tensor of dimensions (#objects, 4)
    '''
    if type(image) != PIL.Image.Image:
        image = F.to_pil_image(image)
    new_image = image.copy()
    labels = labels.tolist()
    draw = ImageDraw.Draw(new_image)
    boxes = boxes.tolist()
    for i in range(len(boxes)):
        draw.rectangle(xy=boxes[i], outline=label_color_map[rev_label_map[labels[i]]])
    new_image.save('vis/{}.jpg'.format(name))


voc_labels = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
              'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
              'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
label_map = {k: v + 1 for v, k in enumerate(voc_labels)}
# Inverse mapping
rev_label_map = {v: k for k, v in label_map.items()}
# Colormap for bounding box
CLASSES = 20
distinct_colors = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                   for i in range(CLASSES)]
label_color_map = {k: distinct_colors[i] for i, k in enumerate(label_map.keys())}
