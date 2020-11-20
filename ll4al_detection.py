import argparse
import os
import random

# Torch
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler

# Torchvison
import torchvision.transforms as T
import torchvision.models as models
from torchvision.datasets import CIFAR100, CIFAR10

# Utils
import visdom
from tqdm import tqdm

# ll4al
import ll4al.models.resnet as resnet
import ll4al.models.lossnet as lossnet
from ll4al.config import *
from ll4al.data.sampler import SubsetSequentialSampler
from ll4al.main import *

# detection
from detection.train import *
from detection.coco_utils import get_coco, get_coco_kp
from detection.group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
from detection.engine import train_one_epoch, evaluate
from detection.frcnn_feature import fasterrcnn_resnet50_fpn_feature
import detection.utils
import detection.transforms as T
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn

torch.cuda.set_device(0)
parser = argparse.ArgumentParser(
    description=__doc__)

parser.add_argument('--data-path', default='/data/yuweiping/coco/', help='dataset')
parser.add_argument('--dataset', default='coco', help='dataset')
parser.add_argument('--model', default='fasterrcnn_resnet50_fpn', help='model')
parser.add_argument('--device', default='cuda', help='device')
parser.add_argument('-b', '--batch-size', default=2, type=int,
                    help='images per gpu, the total batch size is $NGPU x batch_size')
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--lr', default=0.001, type=float,
                    help='initial learning rate, 0.02 is the default value for training '
                         'on 8 gpus and 2 images_per_gpu')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--lr-step-size', default=8, type=int, help='decrease lr every step-size epochs')
parser.add_argument('--lr-steps', default=[16, 22], nargs='+', type=int, help='decrease lr every step-size epochs')
parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
parser.add_argument('--print-freq', default=20, type=int, help='print frequency')
parser.add_argument('--output-dir', default='.', help='path where to save')
parser.add_argument('--resume', default='', help='resume from checkpoint')
parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)
parser.add_argument(
    "--test-only",
    dest="test_only",
    help="Only test the model",
    action="store_true",
)
parser.add_argument(
    "--pretrained",
    dest="pretrained",
    help="Use pre-trained models from the modelzoo",
    action="store_true",
)

# distributed training parameters
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

args = parser.parse_args()

if args.output_dir:
    utils.mkdir(args.output_dir)

random.seed("ywp")
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)  #
# torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = False

if __name__ == '__main__':
    # Initialize dataset/data sampler/data loader/ and labeled/unlabeled data pool.
    dataset, num_classes = get_dataset(args.dataset, "train", get_transform(train=True), args.data_path)
    dataset_test, _ = get_dataset(args.dataset, "val", get_transform(train=False), args.data_path)
    num_images = len(dataset)
    indices = list(range(num_images))
    random.shuffle(indices)
    labeled_set = indices[:int(num_images * 0.1)]
    unlabeled_set = indices[int(num_images * 0.01):]
    train_sampler = SubsetRandomSampler(labeled_set)
    if args.aspect_ratio_group_factor >= 0:
        group_ids = create_aspect_ratio_groups(dataset, k=args.aspect_ratio_group_factor)
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)
    else:
        train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, args.batch_size)
    train_loader = DataLoader(dataset, batch_sampler=train_batch_sampler, pin_memory=True, collate_fn=utils.collate_fn)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    test_loader = DataLoader(dataset_test, batch_size=1, sampler=test_sampler, num_workers=args.workers,
                             collate_fn=utils.collate_fn)
    # Initialize task model and active learning model and optimizer and so on.
    task_model = fasterrcnn_resnet50_fpn(num_classes=num_classes)
    task_model.cuda()
    # model_without_ddp = task_model
    params = [p for p in task_model.parameters() if p.requires_grad]
    optimizer_task = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum,
                                     weight_decay=args.weight_decay)
    lr_scheduler_task = torch.optim.lr_scheduler.MultiStepLR(optimizer_task, milestones=args.lr_steps,
                                                             gamma=args.lr_gamma)

    # loss_module = lossnet.LossNet().cuda()
    # optimizer_loss = torch.optim.SGD(loss_module.parameters(), lr=args.lr, momentum=args.momentum,
    #                                  weight_decay=args.weight_decay)
    # lr_scheduler_loss = torch.optim.lr_scheduler.MultiStepLR(optimizer_task, milestones=args.lr_steps,
    #                                                          gamma=args.lr_gamma)
    # Start active learning cycles training
    for cycle in range(CYCLES):
        for epoch in range(args.epochs):
            task_model.train()
            it = 0
            for images, targets in train_loader:
                it += 1
                images = list(image.cuda() for image in images)
                targets = [{k: v.cuda() for k, v in t.items()} for t in targets]
                loss_dict = task_model(images, targets)
                # features = loss_dict[0]
                # loss_dict[1]['loss_classifier'] = sum(loss_dict[1]['loss_classifier'])
                # loss_dict[1]['loss_box_reg'] = sum(loss_dict[1]['loss_box_reg'])
                task_loss = sum(loss for loss in loss_dict.values())
                # task_loss = []
                # for i in range(args.batch_size):
                #     task_loss.append(sum(loss[i] for loss in task_losses))
                # print(task_losses)
                # features[0] = features['0']
                # features[1] = features['1']
                # features[2] = features['2']
                # features[3] = features['3']
                # if epoch > EPOCHL:
                #     # After EPOCHL epochs, stop the gradient from the loss prediction module propagated to the target model.
                #     features[0] = features[0].detach()
                #     features[1] = features[1].detach()
                #     features[2] = features[2].detach()
                #     features[3] = features[3].detach()
                # pred_loss = loss_module(features).cuda()
                # pred_loss = pred_loss.view(pred_loss.size(0))
                # m_backbone_loss = task_loss / args.batch_size
                # m_module_loss = LossPredLoss(pred_loss, target_loss, margin=MARGIN)
                # losses = m_backbone_loss + WEIGHT * m_module_loss

                optimizer_task.zero_grad()
                # optimizer_loss.zero_grad()
                task_loss.backward()
                optimizer_task.step()
                # optimizer_loss.step()
                if it % 20 == 0:
                    print('cycle {}, epoch {}, iter {}/{}, task_loss {:.4f} (lr {}), pred_loss {:.4f} (lr {})'.format(
                        cycle, epoch, it, len(train_loader), task_loss.item(),
                        optimizer_task.state_dict()['param_groups'][0]['lr'],
                        0, 0))
            lr_scheduler_task.step()
            # lr_scheduler_loss.step()
            evaluate(task_model, test_loader)
