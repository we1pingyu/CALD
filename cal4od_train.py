import datetime
import os
import time
import random
import math
import sys
import numpy as np
import math
import scipy.stats
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

import torch
import torch.utils.data
from torch import nn
import torchvision
import torchvision.models.detection
import torchvision.models.detection.mask_rcnn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
import torchvision.transforms.functional as F

from detection.coco_utils import get_coco, get_coco_kp
from detection.group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
from detection.engine import coco_evaluate, voc_evaluate
from detection import utils
from detection import transforms as T
from detection.train import *
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn
from cal4od.cal4od_helper import *
from ll4al.data.sampler import SubsetSequentialSampler
from detection.frcnn_la import fasterrcnn_resnet50_fpn_feature


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
    # cls_counts = [0] * 20
    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        # for target in targets:
        #     for label in target['labels']:
        #         cls_counts[label.item() - 1] += 1
        task_loss_dict = task_model(images, targets)
        task_losses = sum(loss for loss in task_loss_dict.values())
        # reduce losses over all GPUs for logging purposes
        task_loss_dict_reduced = utils.reduce_dict(task_loss_dict)
        task_losses_reduced = sum(loss.cpu() for loss in task_loss_dict_reduced.values())
        task_loss_value = task_losses_reduced.item()
        if not math.isfinite(task_loss_value):
            print("Loss is {}, stopping training".format(task_loss_value))
            print(task_loss_dict_reduced)
            sys.exit(1)

        task_optimizer.zero_grad()
        task_losses.backward()
        task_optimizer.step()
        if task_lr_scheduler is not None:
            task_lr_scheduler.step()
        metric_logger.update(task_loss=task_losses_reduced)
        metric_logger.update(task_lr=task_optimizer.param_groups[0]["lr"])
    # print(cls_counts / np.sum(cls_counts))
    return metric_logger


def calcu_iou(A, B):
    '''
    calculate two box's iou
    '''
    width = min(A[2], B[2]) - max(A[0], B[0]) + 1
    height = min(A[3], B[3]) - max(A[1], B[1]) + 1
    if width <= 0 or height <= 0:
        return 0
    Aarea = (A[2] - A[0]) * (A[3] - A[1] + 1)
    Barea = (B[2] - B[0]) * (B[3] - B[1] + 1)
    iner_area = width * height
    return iner_area / (Aarea + Barea - iner_area)


def get_uncertainty(task_model, unlabeled_loader, aves=None):
    task_model.eval()
    with torch.no_grad():
        consistency_all = []
        mean_all = []
        for images, _ in unlabeled_loader:
            torch.cuda.synchronize()
            # only support 1 batch size
            aug_images = []
            aug_boxes = []
            for image in images:
                output = task_model([F.to_tensor(image).cuda()])
                ref_boxes, prob_max, ref_scores_cls, ref_labels = output[0]['boxes'], output[0][
                    'prob_max'], output[0]['scores_cls'], output[0]['labels']
                if output[0]['boxes'].shape[0] == 0:
                    consistency_all.append([0.0])
                    break
                # start augment
                # image = SaltPepperNoise(image, 0.05)
                flip_image, flip_boxes = HorizontalFlip(image, ref_boxes)
                aug_images.append(flip_image.cuda())
                aug_boxes.append(flip_boxes.cuda())
                # draw_PIL_image(flip_image, flip_boxes, ref_labels, '_1')
                # color_swap_image = ColorSwap(image)
                # aug_images.append(color_swap_image.cuda())
                # aug_boxes.append(reference_boxes)
                # draw_PIL_image(color_swap_image, reference_boxes, reference_labels, 'color_swap')
                # for i in range(2, 6):
                #     color_adjust_image = ColorAdjust(image, i)
                #     aug_images.append(color_adjust_image.cuda())
                #     aug_boxes.append(reference_boxes)
                #     draw_PIL_image(color_adjust_image, reference_boxes, reference_labels, i)
                # for i in range(1, 7):
                #     sp_image = SaltPepperNoise(image, i * 0.05)
                #     aug_images.append(sp_image.cuda())
                #     aug_boxes.append(ref_boxes)
                #     draw_PIL_image(sp_image, ref_boxes, ref_labels, i)
                cutout_image = cutout(image, ref_boxes, ref_labels)
                aug_images.append(cutout_image.cuda())
                aug_boxes.append(ref_boxes)
                # draw_PIL_image(cutout_image, ref_boxes, ref_labels, '_2')
                # flip_cutout_image = cutout(flip_image.cuda(), flip_boxes.cuda(), ref_labels)
                # aug_images.append(flip_cutout_image.cuda())
                # aug_boxes.append(flip_boxes.cuda())
                resize_image, resize_boxes = resize(image, ref_boxes, 0.7)
                aug_images.append(resize_image.cuda())
                aug_boxes.append(resize_boxes)
                # # draw_PIL_image(resize_image, resize_boxes, ref_labels, '_3')
                # resize_image, resize_boxes = resize(image, ref_boxes, 2.0)
                # aug_images.append(resize_image.cuda())
                # aug_boxes.append(resize_boxes)
                # draw_PIL_image(resize_image, resize_boxes, ref_labels, '_4')
                # rot_image, rot_boxes = rotate(flip_image, flip_boxes, 10)
                # aug_images.append(rot_image.cuda())
                # aug_boxes.append(rot_boxes)
                # draw_PIL_image(rot_image, ref_boxes, ref_labels, 1)
                # rot_image, rot_boxes = rotate(flip_image, flip_boxes, -10)
                # aug_images.append(rot_image.cuda())
                # aug_boxes.append(rot_boxes)
                # draw_PIL_image(rot_image, ref_boxes, ref_labels, 2)
                outputs = []
                for aug_image in aug_images:
                    outputs.append(task_model([aug_image])[0])
                # outputs = task_model(aug_images)
                consistency_aug = []
                mean_aug = []
                if aves is None:
                    aves = [0] * len(aug_images)
                for output, aug_box, aug_image, ave in zip(outputs, aug_boxes, aug_images, aves):
                    consistency_img = []
                    mean_img = []
                    boxes, scores_cls, pm, labels = output['boxes'], output['scores_cls'], output['prob_max'], output[
                        'labels']
                    if len(boxes) == 0:
                        consistency_aug.append(0)
                        mean_aug.append(0.0)
                        continue
                    for ab, ref_score_cls, ref_pm in zip(aug_box, ref_scores_cls, prob_max):
                        width = torch.min(ab[2], boxes[:, 2]) - torch.max(ab[0], boxes[:, 0])
                        height = torch.min(ab[3], boxes[:, 3]) - torch.max(ab[1], boxes[:, 1])
                        Aarea = (ab[2] - ab[0]) * (ab[3] - ab[1])
                        Barea = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                        iner_area = width * height
                        iou = iner_area / (Aarea + Barea - iner_area)
                        iou[width < 0] = 0.0
                        iou[height < 0] = 0.0
                        p = ref_score_cls.cpu().numpy()
                        q = scores_cls[torch.argmax(iou)].cpu().numpy()
                        m = (p + q) / 2
                        # kldiv = ((np.log(p) * np.log(p / q)).mean() + (np.log(q) * np.log(q / p)).mean())
                        js = 0.5 * scipy.stats.entropy(p, m) + 0.5 * scipy.stats.entropy(q, m)
                        # print(np.sum(p), np.sum(q))
                        if js < 0:
                            js = 0
                        consistency_img.append(torch.abs(
                            torch.max(iou) + 0.5 * (1 - js) * (ref_pm + pm[torch.argmax(iou)])).item())
                        mean_img.append(torch.abs(
                            torch.max(iou) + 0.5 * (1 - js) * (ref_pm + pm[torch.argmax(iou)])).item())
                        continue
                    consistency_aug.append(consistency_img)
                    mean_aug.append(np.mean(mean_img))
                    continue
                consistency_all.append(consistency_aug)
                mean_all.append(mean_aug)
                continue
    mean_aug = np.mean(mean_all, axis=0)
    print(mean_aug)
    call = []
    for consistency_aug in consistency_all:
        cau = []
        for consistency_img, mean_img in zip(consistency_aug, mean_aug):
            ci = 1.0
            if not isinstance(consistency_img, list):
                ci = 0
            else:
                for c in consistency_img:
                    ci = min(ci, np.abs(c - mean_img))
            cau.append(ci)
        call.append(np.mean(cau))
    return call  # , np.mean(mean_all, axis=0)


def init_uncertainty(task_model, unlabeled_loader):
    task_model.eval()
    with torch.no_grad():
        uncertainity = []
        for images, _ in unlabeled_loader:
            # only support 1 batch size
            aug_images = []
            aug_features = []
            for image in images:
                output = task_model([F.to_tensor(image).cuda()])
                ref_features = output[0]['features']
                # start augment
                flip_image, flip_features = HorizontalFlipFeatures(image, ref_features)
                aug_images.append(flip_image.cuda())
                aug_features.append(flip_features)
                # resize_image, resize_features = resizeFeatures(image, ref_features, 2)
                # aug_images.append(resize_image.cuda())
                # aug_features.append(resize_features)
                # resize_image, resize_features = resizeFeatures(image, ref_features, 0.5)
                # aug_images.append(resize_image.cuda())
                # aug_features.append(resize_features)
                outputs = task_model(aug_images)
                aug_uncertainty = []
                for output, aug_feature in zip(outputs, aug_features):
                    features = output['features']
                    fpn_uncertainty = []
                    for k in aug_feature:
                        single_l1 = torch.mean(torch.abs(aug_feature[k] - features[k]))
                        fpn_uncertainty.append(single_l1.item())
                    aug_uncertainty.append(np.mean(fpn_uncertainty))
                uncertainity.append(-1.0 * np.mean(aug_uncertainty))
    return uncertainity


def main(args):
    torch.cuda.set_device(0)
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    # torch.backends.cudnn.enabled = True
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    # Data loading code
    print("Loading data")

    if 'voc2007' in args.dataset:
        dataset, num_classes = get_dataset(args.dataset, "trainval", get_transform(train=True), args.data_path)
        dataset_aug, _ = get_dataset(args.dataset, "trainval", None, args.data_path)
        dataset_test, _ = get_dataset(args.dataset, "test", get_transform(train=False), args.data_path)
    else:
        dataset, num_classes = get_dataset(args.dataset, "train", get_transform(train=True), args.data_path)
        dataset_aug, _ = get_dataset(args.dataset, "train", None, args.data_path)
        dataset_test, _ = get_dataset(args.dataset, "val", get_transform(train=False), args.data_path)

    print("Creating data loaders")
    num_images = len(dataset)
    indices = list(range(num_images))
    random.shuffle(indices)
    if args.init:
        unlabeled_loader = DataLoader(dataset_aug, batch_size=1, sampler=SubsetSequentialSampler(indices),
                                      num_workers=args.workers, collate_fn=utils.collate_fn)
        pre_model = fasterrcnn_resnet50_fpn_feature(num_classes=num_classes, min_size=600, max_size=1000)
        pre_model.to(device)
        uncertainty = init_uncertainty(pre_model, unlabeled_loader)
        arg = np.argsort(uncertainty)
        labeled_set = list(torch.tensor(indices)[arg][10::10].numpy())
        unlabeled_set = list(set(indices) - set(labeled_set))
        # labeled_set = indices[:int(num_images * 0.1)]
        # unlabeled_set = indices[int(num_images * 0.1):]
        # print(len(set(labeled_set)) == len(set(labeled_set2)))
    else:
        labeled_set = indices[:int(num_images * 0.1)]
        unlabeled_set = indices[int(num_images * 0.1):]
    # print(len(set(labeled_set)))
    train_sampler = SubsetRandomSampler(labeled_set)
    data_loader_test = DataLoader(dataset_test, batch_size=1, sampler=SequentialSampler(dataset_test),
                                  num_workers=args.workers, collate_fn=utils.collate_fn)
    for cycle in range(args.cycles):
        if args.aspect_ratio_group_factor >= 0:
            group_ids = create_aspect_ratio_groups(dataset, k=args.aspect_ratio_group_factor)
            train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)
        else:
            train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, args.batch_size, drop_last=True)

        data_loader = torch.utils.data.DataLoader(dataset, batch_sampler=train_batch_sampler, num_workers=args.workers,
                                                  collate_fn=utils.collate_fn)

        print("Creating model")
        task_model = fasterrcnn_resnet50_fpn_feature(num_classes=num_classes, min_size=600, max_size=1000)
        task_model.to(device)
        if not args.init and cycle == 0:
            if '2007' in args.dataset:
                checkpoint = torch.load(os.path.join('basemodel', 'voc2007_frcnn_1st.pth'), map_location='cpu')
            elif '2012' in args.dataset:
                checkpoint = torch.load(os.path.join('basemodel', 'voc2012_frcnn_1st.pth'), map_location='cpu')
            task_model.load_state_dict(checkpoint['model'])
            # if 'coco' in args.dataset:
            #     coco_evaluate(task_model, data_loader_test)
            # elif 'voc' in args.dataset:
            #     voc_evaluate(task_model, data_loader_test, args.dataset)
            print("Getting stability")
            # labeled_loader = DataLoader(dataset_aug, batch_size=1, sampler=SubsetSequentialSampler(labeled_set),
            #                             num_workers=args.workers, pin_memory=True, collate_fn=utils.collate_fn)
            # _, aves = get_uncertainty(task_model, labeled_loader)
            random.shuffle(unlabeled_set)
            subset = unlabeled_set
            unlabeled_loader = DataLoader(dataset_aug, batch_size=1, sampler=SubsetSequentialSampler(subset),
                                          num_workers=args.workers, pin_memory=True, collate_fn=utils.collate_fn)
            uncertainty = get_uncertainty(task_model, unlabeled_loader)
            arg = np.argsort(uncertainty)

            # Update the labeled dataset and the unlabeled dataset, respectively
            labeled_set += list(torch.tensor(subset)[arg][:int(num_images * 0.05)].numpy())
            unlabeled_set = list(set(subset) - set(labeled_set))

            # Create a new dataloader for the updated labeled dataset
            train_sampler = SubsetRandomSampler(labeled_set)
            continue
        params = [p for p in task_model.parameters() if p.requires_grad]
        task_optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        task_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(task_optimizer, milestones=args.lr_steps,
                                                                 gamma=args.lr_gamma)

        # Start active learning cycles training
        if args.test_only:
            if 'coco' in args.dataset:
                coco_evaluate(task_model, data_loader_test)
            elif 'voc' in args.dataset:
                voc_evaluate(task_model, data_loader_test, args.dataset, path=args.results_path)
            return
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
                    voc_evaluate(task_model, data_loader_test, args.dataset, path=args.results_path)
        # if cycle == 0:
        #     utils.save_on_master({
        #         'model': task_model.state_dict(),
        #         'args': args},
        #         os.path.join('basemodel', 'voc2012_frcnn_1st.pth'))
        random.shuffle(unlabeled_set)
        subset = unlabeled_set
        unlabeled_loader = DataLoader(dataset_aug, batch_size=1, sampler=SubsetSequentialSampler(subset),
                                      num_workers=args.workers, pin_memory=True, collate_fn=utils.collate_fn)
        print("Getting stability")
        labeled_loader = DataLoader(dataset_aug, batch_size=1, sampler=SubsetSequentialSampler(labeled_set),
                                    num_workers=args.workers, pin_memory=True, collate_fn=utils.collate_fn)
        # _, aves = get_uncertainty(task_model, labeled_loader)
        uncertainty = get_uncertainty(task_model, unlabeled_loader)
        arg = np.argsort(uncertainty)
        # Update the labeled dataset and the unlabeled dataset, respectively
        labeled_set += list(torch.tensor(subset)[arg][:int(num_images * 0.05)].numpy())
        unlabeled_set = list(set(subset) - set(labeled_set))
        # Create a new dataloader for the updated labeled dataset
        train_sampler = SubsetRandomSampler(labeled_set)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)

    parser.add_argument('--data-path', default='/data/yuweiping/voc/', help='dataset')
    parser.add_argument('--dataset', default='voc2007', help='dataset')
    parser.add_argument('--model', default='fasterrcnn_resnet50_fpn', help='model')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=2, type=int,
                        help='images per gpu, the total batch size is $NGPU x batch_size')
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
    parser.add_argument('-p', '--results-path', default='results', help='path to save detection results (only for voc)')
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)
    parser.add_argument('-i', "--init", dest="init", help="if use init sample", action="store_true")
    parser.add_argument("--test-only", dest="test_only", help="Only test the model", action="store_true")
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
