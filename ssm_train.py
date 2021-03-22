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
import numpy as np

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

from detection.frcnn_ssm import fasterrcnn_resnet50_fpn_ssm
from detection.coco_utils import get_coco, get_coco_kp
from detection.group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
from detection.engine import coco_evaluate, voc_evaluate
from detection import utils
from detection import transforms as T
from detection.train import *
from detection.retina_ssm import retinanet_resnet50_fpn_ssm

from ll4al.data.sampler import SubsetSequentialSampler

from ssm.ssm_helper import *

import warnings
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn

warnings.filterwarnings("ignore")


def train_one_epoch(task_model, task_optimizer, data_loader, device, cycle, epoch, print_freq):
    task_model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('task_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Cycle:[{}] Epoch: [{}]'.format(cycle, epoch)

    task_lr_scheduler = None

    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        task_lr_scheduler = utils.warmup_lr_scheduler(task_optimizer, warmup_iters, warmup_factor)
    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
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
            sys.exit(1)

        task_optimizer.zero_grad()
        losses.backward()
        task_optimizer.step()
        if task_lr_scheduler is not None:
            task_lr_scheduler.step()
        metric_logger.update(task_loss=task_losses_reduced)
        metric_logger.update(task_lr=task_optimizer.param_groups[0]["lr"])
    return metric_logger


def softmax(ary):
    ary = ary.flatten()
    expa = np.exp(ary)
    dom = np.sum(expa)
    return expa / dom


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

    # SSM parameters
    gamma = 0.15
    clslambda = np.array([-np.log(0.9)] * (num_classes - 1))
    # Start active learning cycles training
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
                task_model = fasterrcnn_resnet50_fpn_ssm(num_classes=num_classes, min_size=600, max_size=1000)
            elif 'retina' in args.model:
                task_model = retinanet_resnet50_fpn_ssm(num_classes=num_classes, min_size=600, max_size=1000)
        else:
            if 'faster' in args.model:
                task_model = fasterrcnn_resnet50_fpn_ssm(num_classes=num_classes, min_size=800, max_size=1333)
            elif 'retina' in args.model:
                task_model = retinanet_resnet50_fpn_ssm(num_classes=num_classes, min_size=800, max_size=1333)
        task_model.to(device)

        if not args.init and cycle == 0 and args.skip:
            if 'faster' in args.model:
                checkpoint = torch.load(os.path.join(args.first_checkpoint_path,
                                                     '{}_frcnn_1st.pth'.format(args.dataset)), map_location='cpu')
            elif 'retina' in args.model:
                checkpoint = torch.load(os.path.join(args.first_checkpoint_path,
                                                     '{}_retinanet_1st.pth'.format(args.dataset)), map_location='cpu')
            task_model.load_state_dict(checkpoint['model'])
            if args.test_only:
                if 'coco' in args.dataset:
                    coco_evaluate(task_model, data_loader_test)
                elif 'voc' in args.dataset:
                    # task_model.ssm_mode(False)
                    voc_evaluate(task_model, data_loader_test, args.dataset, path=args.results_path)
                return
            # if 'coco' in args.dataset:
            #     coco_evaluate(task_model, data_loader_test)
            # elif 'voc' in args.dataset:
            #     voc_evaluate(task_model, data_loader_test, args.dataset)
            print("Getting stability")
            random.shuffle(unlabeled_set)
            if 'coco' in args.dataset:
                subset = unlabeled_set[:10000]
            else:
                subset = unlabeled_set
            unlabeled_loader = DataLoader(dataset, batch_size=1, sampler=SubsetSequentialSampler(subset),
                                          num_workers=args.workers,
                                          # more convenient if we maintain the order of subset
                                          pin_memory=True, collate_fn=utils.collate_fn)
            print("Getting detections from unlabeled set")
            allScore, allBox, allY, al_idx = get_uncertainty(task_model, unlabeled_loader)
            al_idx = [subset[i] for i in al_idx]
            # al_idx = subset[:budget_num]
            cls_sum = 0
            cls_loss_sum = np.zeros((num_classes - 1,))
            print(
                "First stage results: unlabeled set: {}, tobe labeled set: {}".format(len(subset), len(al_idx)))
            if len(al_idx) >= budget_num:
                al_idx = al_idx[:budget_num]
                labeled_set += al_idx
                subset = list(set(subset) - set(al_idx))
                print(len(set(labeled_set)))
                print(
                    "First stage results: unlabeled set: {}, tobe labeled set: {}".format(len(subset), len(al_idx)))
                # Create a new dataloader for the updated labeled dataset
                train_sampler = SubsetRandomSampler(labeled_set)
                clslambda = 0.9 * clslambda - 0.1 * np.log(softmax(cls_loss_sum / (cls_sum + 1e-30)))
                gamma = min(gamma + 0.05, 1)
                unlabeled_set = list(set(unlabeled_set) - set(al_idx))
                continue
            subset = list(set(subset) - set(al_idx))
            print("Image cross validation")
            for i in range(len(subset)):
                if len(al_idx) >= budget_num:
                    break
                cls_sum += len(allBox[i])
                for j, box in enumerate(allBox[i]):
                    if len(al_idx) >= budget_num:
                        break
                    score = allScore[i][j]
                    label = torch.tensor(allY[i][j]).cuda()
                    loss = -((1 + label.cpu().numpy()) / 2 * np.log(score.cpu().numpy()) + (
                            1 - label.cpu().numpy()) / 2 * np.log(1 - score.cpu().numpy() + 1e-30))
                    cls_loss_sum += loss
                    v, v_val = judge_uv(loss, gamma, clslambda)
                    if v:
                        # print(label)
                        if torch.sum(label == 1) == 1 and torch.where(label == 1)[0] != 0:
                            # add Imgae Cross Validation
                            pre_cls = torch.where(label == 1)[0]
                            pre_box = box
                            curr_ind = [subset[i]]
                            curr_sampler = SubsetSequentialSampler(curr_ind)
                            curr_loader = DataLoader(dataset, batch_size=1, sampler=curr_sampler,
                                                     num_workers=args.workers, pin_memory=True,
                                                     collate_fn=utils.collate_fn)
                            labeled_sampler = SubsetRandomSampler(labeled_set)
                            labeled_loader = DataLoader(dataset, batch_size=1, sampler=labeled_sampler,
                                                        num_workers=args.workers, pin_memory=True,
                                                        collate_fn=utils.collate_fn)
                            cross_validate, _ = image_cross_validation(
                                task_model, curr_loader, labeled_loader, pre_box, pre_cls)
                            if not cross_validate:
                                al_idx.append(subset[i])
                                break
                        else:
                            continue
                    else:
                        al_idx.append(subset[i])
                        break
            # Update the labeled dataset and the unlabeled dataset, respectively
            print(
                "Second stage results: unlabeled set: {}, tobe labeled set: {}".format(len(subset),
                                                                                       len(set(al_idx))))
            subset = list(set(subset) - set(al_idx))
            if len(al_idx) > budget_num:
                al_idx = al_idx[:budget_num]
            if len(al_idx) < budget_num:
                al_idx += list(set(subset) - set(al_idx))[:budget_num - len(al_idx)]
            labeled_set += al_idx
            unlabeled_set = list(set(unlabeled_set) - set(al_idx))
            print(
                "Second stage results: unlabeled set: {}, tobe labeled set: {}".format(len(subset), len(set(al_idx))))
            # Create a new dataloader for the updated labeled dataset
            train_sampler = SubsetRandomSampler(labeled_set)
            clslambda = 0.9 * clslambda - 0.1 * np.log(softmax(cls_loss_sum / (cls_sum + 1e-30)))
            gamma = min(gamma + 0.05, 1)
            continue

        params = [p for p in task_model.parameters() if p.requires_grad]
        task_optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        task_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(task_optimizer, milestones=args.lr_steps,
                                                                 gamma=args.lr_gamma)

        print("Start training")
        start_time = time.time()
        for epoch in range(args.start_epoch, args.total_epochs):
            train_one_epoch(task_model, task_optimizer, data_loader, device, cycle, epoch, args.print_freq)
            task_lr_scheduler.step()
            # evaluate after pre-set epoch
            if (epoch + 1) == args.total_epochs:
                if 'coco' in args.dataset:
                    coco_evaluate(task_model, data_loader_test)
                elif 'voc' in args.dataset:
                    task_model.ssm_mode(False)
                    voc_evaluate(task_model, data_loader_test, args.dataset, path=args.results_path)
        # if not args.skip and cycle == 0:
        #     utils.save_on_master({
        #         'model': task_model.state_dict(), 'args': args},
        #         os.path.join(args.first_checkpoint_path, '{}_frcnn_1st.pth'.format(args.dataset)))
        random.shuffle(unlabeled_set)
        if 'coco' in args.dataset:
            subset = unlabeled_set[:10000]
        else:
            subset = unlabeled_set
        unlabeled_loader = DataLoader(dataset, batch_size=1, sampler=SubsetSequentialSampler(subset),
                                      num_workers=args.workers,
                                      # more convenient if we maintain the order of subset
                                      pin_memory=True, collate_fn=utils.collate_fn)
        print("Getting detections from unlabeled set")
        allScore, allBox, allY, al_idx = get_uncertainty(task_model, unlabeled_loader)
        al_idx = [subset[i] for i in al_idx]
        cls_sum = 0
        cls_loss_sum = np.zeros((num_classes - 1,))
        print(
            "First stage results: unlabeled set: {}, tobe labeled set: {}".format(len(subset), len(set(al_idx))))
        if len(al_idx) >= budget_num:
            al_idx = al_idx[:budget_num]
            labeled_set += al_idx
            print(len(set(labeled_set)))
            subset = list(set(subset) - set(al_idx))
            print(
                "First stage results: unlabeled set: {}, tobe labeled set: {}".format(len(subset), len(set(al_idx))))
            # Create a new dataloader for the updated labeled dataset
            train_sampler = SubsetRandomSampler(labeled_set)
            clslambda = 0.9 * clslambda - 0.1 * np.log(softmax(cls_loss_sum / (cls_sum + 1e-30)))
            gamma = min(gamma + 0.05, 1)
            unlabeled_set = list(set(unlabeled_set) - set(al_idx))
            continue
        subset = list(set(subset) - set(al_idx))
        print("Image cross validation")
        for i in range(len(subset)):
            if len(al_idx) >= budget_num:
                break
            cls_sum += len(allBox[i])
            for j, box in enumerate(allBox[i]):
                if len(al_idx) >= budget_num:
                    break
                score = allScore[i][j]
                label = torch.tensor(allY[i][j]).cuda()
                loss = -((1 + label.cpu().numpy()) / 2 * np.log(score.cpu().numpy()) + (
                        1 - label.cpu().numpy()) / 2 * np.log(1 - score.cpu().numpy() + 1e-30))
                cls_loss_sum += loss
                v, v_val = judge_uv(loss, gamma, clslambda)
                if v:
                    if torch.sum(label == 1) == 1 and torch.where(label == 1)[0] != 0:
                        # add Imgae Cross Validation
                        pre_cls = torch.where(label == 1)[0]
                        pre_box = box
                        curr_ind = [subset[i]]
                        curr_sampler = SubsetSequentialSampler(curr_ind)
                        curr_loader = DataLoader(dataset, batch_size=1, sampler=curr_sampler,
                                                 num_workers=args.workers, pin_memory=True, collate_fn=utils.collate_fn)
                        labeled_sampler = SubsetRandomSampler(labeled_set)
                        labeled_loader = DataLoader(dataset, batch_size=1, sampler=labeled_sampler,
                                                    num_workers=args.workers, pin_memory=True,
                                                    collate_fn=utils.collate_fn)
                        cross_validate, _ = image_cross_validation(
                            task_model, curr_loader, labeled_loader, pre_box, pre_cls)
                        if not cross_validate:
                            al_idx.append(subset[i])
                            break
                else:
                    al_idx.append(subset[i])
                    break
        # Update the labeled dataset and the unlabeled dataset, respectively
        print("Second stage results: unlabeled set: {}, tobe labeled set: {}".format(len(subset), len(set(al_idx))))
        subset = list(set(subset) - set(al_idx))
        if len(al_idx) > budget_num:
            al_idx = al_idx[:budget_num]
        if len(al_idx) < budget_num:
            al_idx += list(set(subset) - set(al_idx))[:budget_num - len(al_idx)]
        labeled_set += al_idx
        unlabeled_set = list(set(unlabeled_set) - set(al_idx))
        print("Second stage results: unlabeled set: {}, tobe labeled set: {}".format(len(subset), len(set(al_idx))))
        # Create a new dataloader for the updated labeled dataset
        train_sampler = SubsetRandomSampler(labeled_set)
        clslambda = 0.9 * clslambda - 0.1 * np.log(softmax(cls_loss_sum / (cls_sum + 1e-30)))
        gamma = min(gamma + 0.05, 1)

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
