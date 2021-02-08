r"""PyTorch Detection Training.

To run in a multi-gpu environment, use the distributed launcher::

    python -m torch.distributed.launch --nproc_per_node=$NGPU --use_env \
        train.py ... --world-size $NGPU

The default hyperparameters are tuned for training on 8 gpus and 2 images per gpu.
    --lr 0.02 --batch-size 2 --world-size 8
If you use different number of gpus, the learning rate should be changed to 0.02/8*$NGPU.

On top of that, for training Faster/Mask R-CNN, the default hyperparameters are
    --epochs 26 --lr-steps 16 22 --aspect-ratio-group-factor 3

Also, if you train Keypoint R-CNN, the default hyperparameters are
    --epochs 46 --lr-steps 36 43 --aspect-ratio-group-factor 3
Because the number of images is smaller in the person keypoint subset of COCO,
the number of epochs should be adapted so that we have the same number of iterations.
"""
import datetime
import os
import random
import sys
import time
import numpy as np
import math
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import torchvision
import torchvision.models.detection
import torchvision.models.detection.mask_rcnn
from detection import transforms as T
from detection import utils
from detection.coco_utils import get_coco, get_coco_kp
from detection.engine import coco_evaluate, voc_evaluate
from detection.group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
from detection.train import *
from ll4al.data.sampler import SubsetSequentialSampler
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn
from torchvision.models.detection.retinanet import retinanet_resnet50_fpn
from vaal.vaal_helper import *


def train_one_epoch(task_model, task_optimizer, vae, vae_optimizer, discriminator, discriminator_optimizer,
                    labeled_dataloader, unlabeled_dataloader, device, cycle, epoch, print_freq):
    def read_unlabeled_data(dataloader):
        while True:
            for images, _ in dataloader:
                yield list(image.to(device) for image in images)

    labeled_data = read_unlabeled_data(labeled_dataloader)
    unlabeled_data = read_unlabeled_data(unlabeled_dataloader)
    task_model.train()
    vae.train()
    discriminator.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('task_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Cycle:[{}] Epoch: [{}]'.format(cycle, epoch)

    task_lr_scheduler = None
    vae_lr_scheduler = None
    discriminator_lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(labeled_dataloader) - 1)

        task_lr_scheduler = utils.warmup_lr_scheduler(task_optimizer, warmup_iters, warmup_factor)
        vae_lr_scheduler = utils.warmup_lr_scheduler(vae_optimizer, warmup_iters, warmup_factor)
        discriminator_lr_scheduler = utils.warmup_lr_scheduler(discriminator_optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(labeled_dataloader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        task_loss_dict = task_model(images, targets)
        task_losses = sum(loss for loss in task_loss_dict.values())
        # reduce losses over all GPUs for logging purposes
        task_loss_dict_reduced = utils.reduce_dict(task_loss_dict)
        task_losses_reduced = sum(loss.cpu() for loss in task_loss_dict_reduced.values())
        task_loss_value = task_losses_reduced.item()
        losses = task_losses
        if not math.isfinite(task_loss_value):
            print("Loss is {}, stopping training".format(task_loss_value))
            print(task_loss_dict_reduced)
            sys.exit(1)
        task_optimizer.zero_grad()
        losses.backward()
        task_optimizer.step()
        if task_lr_scheduler is not None:
            task_lr_scheduler.step()
        metric_logger.update(task_loss=task_losses_reduced)
        metric_logger.update(task_lr=task_optimizer.param_groups[0]["lr"])

    for i in range(len(labeled_dataloader)):
        unlabeled_imgs = next(unlabeled_data)
        labeled_imgs = next(labeled_data)
        recon, z, mu, logvar = vae(labeled_imgs)
        unsup_loss = vae_loss(labeled_imgs, recon, mu, logvar, 1)
        unlab_recon, unlab_z, unlab_mu, unlab_logvar = vae(unlabeled_imgs)
        transductive_loss = vae_loss(unlabeled_imgs, unlab_recon, unlab_mu, unlab_logvar, 1)

        labeled_preds = discriminator(mu)
        unlabeled_preds = discriminator(unlab_mu)

        lab_real_preds = torch.ones(len(labeled_imgs)).cuda()
        unlab_real_preds = torch.ones(len(unlabeled_imgs)).cuda()

        if not len(labeled_preds.shape) == len(lab_real_preds.shape):
            dsc_loss = bce_loss(labeled_preds, lab_real_preds.unsqueeze(1)) + bce_loss(unlabeled_preds,
                                                                                       unlab_real_preds.unsqueeze(1))
        else:
            dsc_loss = bce_loss(labeled_preds, lab_real_preds) + bce_loss(unlabeled_preds, unlab_real_preds)
        total_vae_loss = unsup_loss + transductive_loss + dsc_loss
        vae_optimizer.zero_grad()
        total_vae_loss.backward()
        vae_optimizer.step()

        # Discriminator step
        with torch.no_grad():
            _, _, mu, _ = vae(labeled_imgs)
            _, _, unlab_mu, _ = vae(unlabeled_imgs)

        labeled_preds = discriminator(mu)
        unlabeled_preds = discriminator(unlab_mu)

        lab_real_preds = torch.ones(len(labeled_imgs)).cuda()
        unlab_fake_preds = torch.zeros(len(unlabeled_imgs)).cuda()

        if not len(labeled_preds.shape) == len(lab_real_preds.shape):
            dsc_loss = bce_loss(labeled_preds, lab_real_preds.unsqueeze(1)) + bce_loss(unlabeled_preds,
                                                                                       unlab_fake_preds.unsqueeze(1))
        else:
            dsc_loss = bce_loss(labeled_preds, lab_real_preds) + bce_loss(unlabeled_preds, unlab_fake_preds)
        discriminator_optimizer.zero_grad()
        dsc_loss.backward()
        discriminator_optimizer.step()

        if vae_lr_scheduler is not None:
            vae_lr_scheduler.step()
        if discriminator_lr_scheduler is not None:
            discriminator_lr_scheduler.step()
        if i == len(labeled_dataloader) - 1:
            print('vae_loss: {} dis_loss:{}'.format(total_vae_loss, dsc_loss))

    return metric_logger


def main(args):
    torch.cuda.set_device(0)
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    print(args)

    device = torch.device(args.device)

    # Data loading code
    print("Loading data")
    if 'voc2007' in args.dataset:
        dataset, num_classes = get_dataset(args.dataset, "trainval", get_transform(train=True), args.data_path)
        dataset_test, _ = get_dataset(args.dataset, "test", get_transform(train=False), args.data_path)
    else:
        dataset, num_classes = get_dataset(args.dataset, "train", get_transform(train=True), args.data_path)
        dataset_test, _ = get_dataset(args.dataset, "val", get_transform(train=False), args.data_path)
    if 'voc' in args.dataset:
        init_num = 500
        budget_num = 500
        if 'retina' in args.model:
            init_num = 1000
            budget_num = 500
    else:
        init_num = 5000
        budget_num = 1000
    print("Creating data loaders")
    num_images = len(dataset)
    indices = list(range(num_images))
    random.shuffle(indices)
    labeled_set = indices[:init_num]
    unlabeled_set = list(set(indices) - set(labeled_set))
    train_sampler = SubsetRandomSampler(labeled_set)
    unlabeled_sampler = SubsetRandomSampler(unlabeled_set)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    data_loader_test = DataLoader(dataset_test, batch_size=1, sampler=test_sampler, num_workers=args.workers,
                                  collate_fn=utils.collate_fn)
    for cycle in range(args.cycles):
        if args.aspect_ratio_group_factor >= 0:
            group_ids = create_aspect_ratio_groups(dataset, k=args.aspect_ratio_group_factor)
            train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)
            unlabeled_batch_sampler = GroupedBatchSampler(unlabeled_sampler, group_ids, args.batch_size)
        else:
            train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, args.batch_size, drop_last=True)
            unlabeled_batch_sampler = torch.utils.data.BatchSampler(unlabeled_sampler, args.batch_size, drop_last=True)
        data_loader = torch.utils.data.DataLoader(dataset, batch_sampler=train_batch_sampler, num_workers=args.workers,
                                                  collate_fn=utils.collate_fn)
        unlabeled_dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=unlabeled_batch_sampler,
                                                           num_workers=args.workers,
                                                           collate_fn=utils.collate_fn)
        print("Creating model")
        if 'voc' in args.dataset:
            if 'faster' in args.model:
                task_model = fasterrcnn_resnet50_fpn(num_classes=num_classes, min_size=600, max_size=1000)
            elif 'retina' in args.model:
                task_model = retinanet_resnet50_fpn(num_classes=num_classes, min_size=600, max_size=1000)
        else:
            if 'faster' in args.model:
                task_model = fasterrcnn_resnet50_fpn(num_classes=num_classes, min_size=800, max_size=1333)
            elif 'retina' in args.model:
                task_model = retinanet_resnet50_fpn(num_classes=num_classes, min_size=800, max_size=1333)
        task_model.to(device)

        params = [p for p in task_model.parameters() if p.requires_grad]
        task_optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        task_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(task_optimizer, milestones=args.lr_steps,
                                                                 gamma=args.lr_gamma)
        vae = VAE()
        params = [p for p in vae.parameters() if p.requires_grad]
        vae_optimizer = torch.optim.SGD(params, lr=args.lr / 10, momentum=args.momentum,
                                        weight_decay=args.weight_decay)
        vae_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(vae_optimizer, milestones=args.lr_steps,
                                                                gamma=args.lr_gamma)
        torch.nn.utils.clip_grad_value_(vae.parameters(), 1e5)

        vae.to(device)
        discriminator = Discriminator()
        params = [p for p in discriminator.parameters() if p.requires_grad]
        discriminator_optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum,
                                                  weight_decay=args.weight_decay)
        discriminator_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(discriminator_optimizer,
                                                                          milestones=args.lr_steps,
                                                                          gamma=args.lr_gamma)
        discriminator.to(device)
        # Start active learning cycles training
        if args.test_only:
            if 'coco' in args.dataset:
                coco_evaluate(task_model, data_loader_test)
            elif 'voc' in args.dataset:
                voc_evaluate(task_model, data_loader_test, args.dataset, False, path=args.results_path)
            return
        print("Start training")
        start_time = time.time()
        for epoch in range(args.start_epoch, args.total_epochs):
            train_one_epoch(task_model, task_optimizer, vae, vae_optimizer, discriminator, discriminator_optimizer,
                            data_loader, unlabeled_dataloader, device, cycle, epoch, args.print_freq)
            task_lr_scheduler.step()
            vae_lr_scheduler.step()
            discriminator_lr_scheduler.step()
            # evaluate after pre-set epoch
            if (epoch + 1) == args.total_epochs:
                if 'coco' in args.dataset:
                    coco_evaluate(task_model, data_loader_test)
                elif 'voc' in args.dataset:
                    voc_evaluate(task_model, data_loader_test, args.dataset, False, path=args.results_path)
        # Update the labeled dataset and the unlabeled dataset, respectively
        random.shuffle(unlabeled_set)
        if 'coco' in args.dataset:
            subset = unlabeled_set[:10000]
        else:
            subset = unlabeled_set
        unlabeled_loader = DataLoader(dataset, batch_size=1, sampler=SubsetSequentialSampler(subset),
                                      num_workers=args.workers, pin_memory=True, collate_fn=utils.collate_fn)
        tobe_labeled_inds = sample_for_labeling(vae, discriminator, unlabeled_loader, budget_num)
        tobe_labeled_set = [subset[i] for i in tobe_labeled_inds]
        labeled_set += tobe_labeled_set
        unlabeled_set = list(set(unlabeled_set) - set(tobe_labeled_set))
        # Create a new dataloader for the updated labeled dataset
        train_sampler = SubsetRandomSampler(labeled_set)
        unlabeled_sampler = SubsetRandomSampler(unlabeled_set)
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)

    parser.add_argument('-p', '--data-path', default='/data/yuweiping/coco/', help='dataset path')
    parser.add_argument('--dataset', default='voc2007', help='dataset')
    parser.add_argument('--model', default='fasterrcnn_resnet50_fpn', help='model')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=4, type=int,
                        help='images per gpu, the total batch size is $NGPU x batch_size')
    parser.add_argument('-cp', '--first-checkpoint-path', default='/data/yuweiping/coco/',
                        help='path to save checkpoint of first cycle')
    parser.add_argument('--task_epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-e', '--total_epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--cycles', default=7, type=int, metavar='N',
                        help='number of cycles epochs to run')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--lr', default=0.0025, type=float,
                        help='initial learning rate, 0.02 is the default value for training '
                             'on 8 gpus and 2 images_per_gpu')
    parser.add_argument('--ll-weight', default=0.5, type=float,
                        help='ll loss weight')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--lr-step-size', default=8, type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-steps', default=[16, 19], nargs='+', type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--print-freq', default=1000, type=int, help='print frequency')
    parser.add_argument('--output-dir', default=None, help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('-rp', '--results-path', default='results',
                        help='path to save detection results (only for voc)')
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)
    parser.add_argument('-i', "--init", dest="init", help="if use init sample", action="store_true")
    parser.add_argument("--test-only", dest="test_only", help="Only test the model", action="store_true")
    parser.add_argument('-s', "--skip", dest="skip", help="Skip first cycle and use pretrained model to save time",
                        action="store_true")
    parser.add_argument('-m', "--mutual", dest="mutual", help="use mutual information",
                        action="store_true")
    parser.add_argument('-mr', default=1.2, type=float, help='mutual range')
    parser.add_argument('-bp', default=1.15, type=float, help='base point')
    parser.add_argument("--pretrained", dest="pretrained", help="Use pre-trained models from the modelzoo",
                        action="store_true")
    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    args = parser.parse_args()

    if args.output_dir:
        utils.mkdir(args.output_dir)

    main(args)
