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
import time
import random
import math
import sys
import pickle

import torch
import torch.utils.data
from torch import nn
import torchvision
import torchvision.models.detection
import torchvision.models.detection.mask_rcnn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler

from detection.frcnn_ll import fasterrcnn_resnet50_fpn_feature
from detection.retina_ll import retinanet_mobilenet, retinanet_resnet50_fpn
from detection.coco_utils import get_coco, get_coco_kp
from detection.group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
from detection.engine import coco_evaluate, voc_evaluate
from detection import utils
from detection import transforms as T
from detection.train import *

import ll4al.models.resnet as resnet
import ll4al.models.lossnet as lossnet
from ll4al.config import *
from ll4al.data.sampler import SubsetSequentialSampler
from ll4al.main import *


def train_one_epoch(task_model, task_optimizer, ll_model, ll_optimizer, data_loader, device, cycle, epoch, print_freq):
    task_model.train()
    ll_model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('task_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('ll_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Cycle:[{}] Epoch: [{}]'.format(cycle, epoch)

    task_lr_scheduler = None
    ll_lr_scheduler = None

    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        task_lr_scheduler = utils.warmup_lr_scheduler(task_optimizer, warmup_iters, warmup_factor)
        ll_lr_scheduler = utils.warmup_lr_scheduler(ll_optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        features, task_loss_dict = task_model(images, targets)
        if 'faster' in args.model:
            _task_losses = sum(loss for loss in task_loss_dict.values())
            # print(_task_losses)
            task_loss_dict['loss_objectness'] = torch.mean(task_loss_dict['loss_objectness'])
            task_loss_dict['loss_rpn_box_reg'] = torch.mean(task_loss_dict['loss_rpn_box_reg'])
            task_loss_dict['loss_classifier'] = torch.mean(task_loss_dict['loss_classifier'])
            task_loss_dict['loss_box_reg'] = torch.mean(task_loss_dict['loss_box_reg'])
            task_losses = sum(loss for loss in task_loss_dict.values())
            # reduce losses over all GPUs for logging purposes
            task_loss_dict_reduced = utils.reduce_dict(task_loss_dict)
            task_losses_reduced = sum(loss.cpu() for loss in task_loss_dict_reduced.values())
            task_loss_value = task_losses_reduced.item()
            if epoch >= args.task_epochs:
                # After EPOCHL epochs, stop the gradient from the loss prediction module propagated to the target model.
                features['0'] = features['0'].detach()
                features['1'] = features['1'].detach()
                features['2'] = features['2'].detach()
                features['3'] = features['3'].detach()
            ll_pred = ll_model(features).cuda()
        elif 'retina' in args.model:
            _task_losses = sum(torch.stack(loss[1]) for loss in task_loss_dict.values())
            task_loss_dict['classification'] = task_loss_dict['classification'][0]
            task_loss_dict['bbox_regression'] = task_loss_dict['bbox_regression'][0]
            # for loss in task_loss_dict.values():
            #     print(loss)
            task_losses = sum(loss for loss in task_loss_dict.values())
            task_loss_dict_reduced = utils.reduce_dict(task_loss_dict)
            task_losses_reduced = sum(loss.cpu() for loss in task_loss_dict_reduced.values())
            task_loss_value = task_losses_reduced.item()
            if epoch >= args.task_epochs:
                # After EPOCHL epochs, stop the gradient from the loss prediction module propagated to the target model.
                _features = dict()
                _features['0'] = features[0].detach()
                _features['1'] = features[1].detach()
                _features['2'] = features[2].detach()
                _features['3'] = features[3].detach()
            else:
                _features = dict()
                _features['0'] = features[0]
                _features['1'] = features[1]
                _features['2'] = features[2]
                _features['3'] = features[3]
            ll_pred = ll_model(_features).cuda()
        ll_pred = ll_pred.view(ll_pred.size(0))
        ll_loss = args.ll_weight * LossPredLoss(ll_pred, _task_losses, margin=MARGIN)
        losses = task_losses + ll_loss
        if not math.isfinite(task_loss_value):
            print("Loss is {}, stopping training".format(task_loss_value))
            print(task_loss_dict_reduced)
            sys.exit(1)

        task_optimizer.zero_grad()
        ll_optimizer.zero_grad()
        losses.backward()
        task_optimizer.step()
        ll_optimizer.step()
        if task_lr_scheduler is not None:
            task_lr_scheduler.step()
        if ll_lr_scheduler is not None:
            ll_lr_scheduler.step()
        metric_logger.update(task_loss=task_losses_reduced)
        metric_logger.update(task_lr=task_optimizer.param_groups[0]["lr"])
        metric_logger.update(ll_loss=ll_loss.item())
        metric_logger.update(ll_lr=ll_optimizer.param_groups[0]["lr"])
    return metric_logger


def get_uncertainty(task_model, ll_model, unlabeled_loader):
    task_model.eval()
    ll_model.eval()
    uncertainty = torch.tensor([]).cuda()

    with torch.no_grad():
        for images, labels in unlabeled_loader:
            images = list(img.cuda() for img in images)
            torch.cuda.synchronize()
            features, _ = task_model(images)
            if 'retina' in args.model:
                _features = dict()
                _features['0'] = features[0].detach()
                _features['1'] = features[0].detach()
                _features['2'] = features[0].detach()
                _features['3'] = features[0].detach()
                ll_pred = ll_model(_features)  # pred_loss = criterion(scores, labels) # ground truth loss
            else:
                ll_pred = ll_model(features)  # pred_loss = criterion(scores, labels) # ground truth loss
            ll_pred = ll_pred.view(ll_pred.size(0))
            uncertainty = torch.cat((uncertainty, ll_pred), 0)
    return uncertainty.cpu()


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
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    data_loader_test = DataLoader(dataset_test, batch_size=1, sampler=test_sampler, num_workers=args.workers,
                                  collate_fn=utils.collate_fn)
    for cycle in range(args.cycles):
        if args.aspect_ratio_group_factor >= 0:
            group_ids = create_aspect_ratio_groups(dataset, k=args.aspect_ratio_group_factor)
            train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)
        else:
            train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, args.batch_size, drop_last=True)

        data_loader = torch.utils.data.DataLoader(dataset, batch_sampler=train_batch_sampler, num_workers=args.workers,
                                                  collate_fn=utils.collate_fn)

        print("Creating model")
        if 'voc' in args.dataset:
            if 'faster' in args.model:
                task_model = fasterrcnn_resnet50_fpn_feature(num_classes=num_classes, min_size=600, max_size=1000)
            elif 'retina' in args.model:
                task_model = retinanet_resnet50_fpn(num_classes=num_classes, min_size=600, max_size=1000)
        else:
            if 'faster' in args.model:
                task_model = fasterrcnn_resnet50_fpn_feature(num_classes=num_classes, min_size=800, max_size=1333)
            elif 'retina' in args.model:
                task_model = retinanet_resnet50_fpn(num_classes=num_classes, min_size=800, max_size=1333)
        task_model.to(device)

        params = [p for p in task_model.parameters() if p.requires_grad]
        task_optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        task_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(task_optimizer, milestones=args.lr_steps,
                                                                 gamma=args.lr_gamma)
        ll_model = lossnet.LossNet()
        ll_model.to(device)
        params_ll = [p for p in ll_model.parameters() if p.requires_grad]
        ll_optimizer = torch.optim.SGD(params_ll, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        ll_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(ll_optimizer, milestones=args.lr_steps,
                                                               gamma=args.lr_gamma)
        # Start active learning cycles training
        if args.test_only:
            if 'coco' in args.dataset:
                coco_evaluate(task_model, data_loader_test, feature=True)
            elif 'voc' in args.dataset:
                voc_evaluate(task_model, data_loader_test, args.dataset, True)
            return
        print("Start training")
        start_time = time.time()
        for epoch in range(args.start_epoch, args.total_epochs):
            train_one_epoch(task_model, task_optimizer, ll_model, ll_optimizer, data_loader, device, cycle, epoch,
                            args.print_freq)
            task_lr_scheduler.step()
            ll_lr_scheduler.step()
            # evaluate after pre-set epoch
            if (epoch + 1) == args.total_epochs:
                if 'coco' in args.dataset:
                    coco_evaluate(task_model, data_loader_test, feature=True)
                elif 'voc' in args.dataset:
                    voc_evaluate(task_model, data_loader_test, args.dataset, True, path=args.results_path)
        random.shuffle(unlabeled_set)
        if 'coco' in args.dataset:
            subset = unlabeled_set[:10000]
        else:
            subset = unlabeled_set
        unlabeled_loader = DataLoader(dataset, batch_size=args.batch_size,
                                      sampler=SubsetSequentialSampler(subset), num_workers=args.workers,
                                      # more convenient if we maintain the order of subset
                                      pin_memory=True, collate_fn=utils.collate_fn)
        uncertainty = get_uncertainty(task_model, ll_model, unlabeled_loader)
        labeled_loader = DataLoader(dataset, batch_size=args.batch_size,
                                    sampler=SubsetSequentialSampler(labeled_set), num_workers=args.workers,
                                    # more convenient if we maintain the order of subset
                                    pin_memory=True, collate_fn=utils.collate_fn)
        u = get_uncertainty(task_model, ll_model, labeled_loader)
        # with open("vis/ll_labeled_metric_{}_{}_{}.pkl".format(args.model, args.dataset, cycle),
        #           "wb") as fp:  # Pickling
        #     pickle.dump(u, fp)
        arg = np.argsort(uncertainty)
        # with open("vis/ll_unlabeled_metric_{}_{}_{}.pkl".format(args.model, args.dataset, cycle),
        #           "wb") as fp:  # Pickling
        #     pickle.dump(torch.tensor(uncertainty)[arg][-1 * budget_num:].numpy(), fp)
        # Update the labeled dataset and the unlabeled dataset, respectively
        labeled_set += list(torch.tensor(subset)[arg][-1 * budget_num:].numpy())
        labeled_set = list(set(labeled_set))
        # with open("vis/ll_{}_{}_{}.txt".format(args.model, args.dataset, cycle), "wb") as fp:  # Pickling
        #     pickle.dump(labeled_set, fp)
        unlabeled_set = list(set(indices) - set(labeled_set))

        # Create a new dataloader for the updated labeled dataset
        train_sampler = SubsetRandomSampler(labeled_set)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)

    parser.add_argument('-p', '--data-path', default='/data/yuweiping/coco/', help='dataset path')
    parser.add_argument('--dataset', default='voc2007', help='dataset')
    parser.add_argument('--model', default='faster_rcnn', help='model')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=4, type=int,
                        help='images per gpu, the total batch size is $NGPU x batch_size')
    parser.add_argument('-cp', '--first-checkpoint-path', default='/data/yuweiping/coco/',
                        help='path to save checkpoint of first cycle')
    parser.add_argument('-t', '--task_epochs', default=0, type=int, metavar='N',
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
    parser.add_argument('--ll-weight', default=1.0, type=float,
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
