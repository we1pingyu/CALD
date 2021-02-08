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
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
import torchvision.transforms.functional as F

from detection.coco_utils import get_coco, get_coco_kp
from detection.group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
from detection.engine import coco_evaluate, voc_evaluate
from detection import utils
from detection import transforms as T
from detection.train import *
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn
from torchvision.models.detection.retinanet import retinanet_resnet50_fpn
from cal4od.cal4od_helper import *
from ll4al.data.sampler import SubsetSequentialSampler
from detection.frcnn_la import fasterrcnn_resnet50_fpn_feature
from detection.retinanet_cal import retinanet_mobilenet, retinanet_resnet50_fpn_cal

A = 1


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
        # global A
        # for image, target in zip(images, targets):
        #     draw_PIL_image(image, target['boxes'], target['labels'], None, "unlabeled_{}".format(A))
        #     A += 1
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


def get_uncertainty(task_model, unlabeled_loader, augs, num_cls):
    for aug in augs:
        if aug not in ['flip', 'multi_ga', 'color_adjust', 'color_swap', 'multi_color_adjust', 'multi_sp', 'cut_out',
                       'multi_cut_out', 'multi_resize', 'larger_resize', 'smaller_resize', 'rotation']:
            print('{} is not in the pre-set augmentations!'.format(aug))
    task_model.eval()
    with torch.no_grad():
        consistency_all = []
        mean_all = []
        cls_all = []
        for images, _ in unlabeled_loader:
            torch.cuda.synchronize()
            # only support 1 batch size
            aug_images = []
            aug_boxes = []
            for image in images:
                output = task_model([F.to_tensor(image).cuda()])
                ref_boxes, prob_max, ref_scores_cls, ref_labels, ref_scores = output[0]['boxes'], output[0][
                    'prob_max'], output[0]['scores_cls'], output[0]['labels'], output[0]['scores']
                if len(ref_scores) > 50:
                    # inds = random.sample(range(0, len(ref_scores)), k=30)
                    inds = np.round(np.linspace(0, len(ref_scores) - 1, 50)).astype(int)
                    # print(len(set(inds)))
                    ref_boxes, prob_max, ref_scores_cls, ref_labels, ref_scores = ref_boxes[inds], prob_max[
                        inds], ref_scores_cls[inds], ref_labels[inds], ref_scores[inds]
                # print(torch.argmax(ref_scores))
                # draw_PIL_image(image, ref_boxes, ref_labels, 'ori', no=torch.topk(ref_scores, 2)[1].data)
                cls_corr = [0] * (num_cls - 1)
                for p, l in zip(prob_max, ref_labels):
                    # if p.item() > 0.4:
                    #     cls_corr[l - 1] += p.item()
                    cls_corr[l - 1] = max(cls_corr[l - 1], p.item())
                cls_corrs = [cls_corr]
                if output[0]['boxes'].shape[0] == 0:
                    consistency_all.append(0.0)
                    cls_all.append(np.mean(cls_corrs, axis=0))
                    break
                # start augment
                if 'flip' in augs:
                    flip_image, flip_boxes = HorizontalFlip(image, ref_boxes)
                    aug_images.append(flip_image.cuda())
                    aug_boxes.append(flip_boxes.cuda())
                    # draw_PIL_image(flip_image, flip_boxes, ref_labels, '1_', no=torch.topk(ref_scores, 2)[1].data)
                if 'multi_ga' in augs:
                    for i in range(1, 7):
                        ga_image = GaussianNoise(image, i * 8)
                        aug_images.append(ga_image.cuda())
                        aug_boxes.append(ref_boxes.cuda())
                if 'color_adjust' in augs:
                    color_adjust_image = ColorAdjust(image, 1.5)
                    aug_images.append(color_adjust_image.cuda())
                    aug_boxes.append(ref_boxes)
                    # draw_PIL_image(color_adjust_image, ref_boxes, ref_labels, 'color')
                if 'color_swap' in augs:
                    color_swap_image = ColorSwap(image)
                    aug_images.append(color_swap_image.cuda())
                    aug_boxes.append(ref_boxes)
                    # draw_PIL_image(color_swap_image, ref_boxes, ref_labels, 'color_swap')
                if 'multi_color_adjust' in augs:
                    for i in range(2, 6):
                        color_adjust_image = ColorAdjust(image, i)
                        aug_images.append(color_adjust_image.cuda())
                        aug_boxes.append(reference_boxes)
                        # draw_PIL_image(color_adjust_image, reference_boxes, reference_labels, i)
                if 'multi_sp' in augs:
                    for i in range(1, 7):
                        sp_image = SaltPepperNoise(image, i * 0.05)
                        aug_images.append(sp_image.cuda())
                        aug_boxes.append(ref_boxes)
                        # draw_PIL_image(sp_image, ref_boxes, ref_labels, i)
                if 'cut_out' in augs:
                    cutout_image = cutout(image, ref_boxes, ref_labels, 2)
                    aug_images.append(cutout_image.cuda())
                    aug_boxes.append(ref_boxes)
                    # draw_PIL_image(cutout_image, ref_boxes, ref_labels, '2_', no=torch.topk(ref_scores, 2)[1].data)
                if 'multi_cut_out' in augs:
                    for i in range(1, 5):
                        cutout_image = cutout(image, ref_boxes, ref_labels, i)
                        aug_images.append(cutout_image.cuda())
                        aug_boxes.append(ref_boxes)
                        # draw_PIL_image(cutout_image, ref_boxes, ref_labels, '_2')
                if 'multi_resize' in augs:
                    for i in range(7, 10):
                        resize_image, resize_boxes = resize(image, ref_boxes, i * 0.1)
                        aug_images.append(resize_image.cuda())
                        aug_boxes.append(resize_boxes)
                # draw_PIL_image(resize_image, resize_boxes, ref_labels, '_3')
                if 'larger_resize' in augs:
                    resize_image, resize_boxes = resize(image, ref_boxes, 1.2)
                    aug_images.append(resize_image.cuda())
                    aug_boxes.append(resize_boxes)
                if 'smaller_resize' in augs:
                    resize_image, resize_boxes = resize(image, ref_boxes, 0.8)
                    aug_images.append(resize_image.cuda())
                    aug_boxes.append(resize_boxes)
                    # draw_PIL_image(resize_image, resize_boxes, ref_labels, '3_', no=torch.topk(ref_scores, 2)[1].data)
                if 'rotation' in augs:
                    rot_image, rot_boxes = rotate(image, ref_boxes, 5)
                    aug_images.append(rot_image.cuda())
                    aug_boxes.append(rot_boxes)
                    # draw_PIL_image(rot_image, rot_boxes, ref_labels, '4_', no=torch.topk(ref_scores, 2)[1].data)
                    # rot_image, rot_boxes = rotate(flip_image, flip_boxes, -10)
                    # aug_images.append(rot_image.cuda())
                    # aug_boxes.append(rot_boxes)
                    # draw_PIL_image(rot_image, ref_boxes, ref_labels, 2)
                outputs = []
                for aug_image in aug_images:
                    outputs.append(task_model([aug_image])[0])
                consistency_aug = []
                mean_aug = []
                i = 1
                # print(len(ref_boxes))
                for output, aug_box, aug_image in zip(outputs, aug_boxes, aug_images):
                    consistency_img = 1.0
                    mean_img = []
                    boxes, scores_cls, pm, labels, scores = output['boxes'], output['scores_cls'], output['prob_max'], \
                                                            output['labels'], output['scores']
                    cls_corr = [0] * (num_cls - 1)
                    for p, l in zip(pm, labels):
                        # if p.item() > 0.4:
                        #     cls_corr[l - 1] += p.item()
                        cls_corr[l - 1] = max(cls_corr[l - 1], p.item())
                    cls_corrs.append(cls_corr)
                    if len(boxes) == 0:
                        consistency_aug.append(0.0)
                        mean_aug.append(0.0)
                        continue
                    j = 0
                    no = []
                    for ab, ref_score_cls, ref_pm, ref_score in zip(aug_box, ref_scores_cls, prob_max, ref_scores):
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
                        js = 0.5 * scipy.stats.entropy(p, m) + 0.5 * scipy.stats.entropy(q, m)
                        if js < 0:
                            js = 0
                        # if 0.7 < torch.max(iou) and 0.4 > 0.5 * (1 - js) * (
                        #         ref_pm + pm[torch.argmax(iou)]) and (ab[2] - ab[0]) > 100 and scores[
                        #     torch.argmax(iou)] > 0.4 and ref_score > 0.4:
                        #     # draw_PIL_image(image, boxes, ref_labels, i, no=[ab], color='greenyellow')
                        #     no = [j, torch.argmax(iou).item()]
                        #     draw_PIL_image_2(aug_image.cpu(), aug_box, boxes, ref_labels, labels, ref_scores, pm,
                        #                      'bad_2', no=no, color='red')
                        #     print(1 / 0)
                        j += 1
                        consistency_img = min(consistency_img, torch.abs(
                            torch.max(iou) + 0.5 * (1 - js) * (ref_pm + pm[torch.argmax(iou)]) - args.bp).item())
                        mean_img.append(torch.abs(
                            torch.max(iou) + 0.5 * (1 - js) * (ref_pm + pm[torch.argmax(iou)])).item())
                        # if j <= 1:
                        #     no.append(torch.argmax(iou).item())
                        #     j += 1
                        # if j == 2:
                        #     draw_PIL_image(aug_image.cpu(), boxes, ref_labels, i, no=no, color='red')
                        #     i += 1
                        #     break
                    consistency_aug.append(consistency_img)
                    mean_aug.append(np.mean(mean_img))
                # if torch.min(torch.topk(ref_scores, 2)[0]) > 0.85 and image.size[0] > image.size[1]:
                #     print(1 / 0)
                consistency_all.append(np.mean(consistency_aug))
                mean_all.append(mean_aug)
                # print(np.mean(consistency_aug))
                cls_corrs = np.mean(np.array(cls_corrs), axis=0)
                cls_all.append(cls_corrs)
                # print(len(consistency_all))
    mean_aug = np.mean(mean_all, axis=0)
    print(mean_aug)
    return consistency_all, cls_all


def cls_kldiv(labeled_loader, cls_corrs, budget):
    cls_inds = []
    result = []
    for _, targets in labeled_loader:
        for target in targets:
            cls_corr = [0] * cls_corrs[0].shape[0]
            for l in target['labels']:
                cls_corr[l - 1] += 1
            result.append(cls_corr)
    for a in list(np.where(np.sum(cls_corrs, axis=1) == 0)[0]):
        cls_inds.append(a)
        result.append(cls_corrs[a])
    while len(cls_inds) < budget:
        # max = 0
        # max_ind = 0
        # for i in range(0, len(cls_corrs)):
        #     if type(cls_corrs[i]) == np.ndarray:
        #         # print(result)
        #         # print(cls_corrs[i])
        #         if np.sum(cls_corrs[i]) == 0:
        #             max_ind = i
        #             break
        #         # if np.sum(result) == 0:
        #         #     p = np.ones(cls_corrs[i].shape) / cls_corrs[i].shape[0]
        #         # else:
        #         #     p = np.mean(np.array(result), axis=0) / np.sum(np.mean(np.array(result), axis=0))
        #         # q = cls_corrs[i] / np.sum(cls_corrs[i])
        #         # # print(p, q)
        #         # m = (p + q) / 2
        #         # js = 0.5 * scipy.stats.entropy(p, m) + 0.5 * scipy.stats.entropy(q, m)
        #
        #         if js > max:
        #             max = js
        #             max_ind = i
        # batch cls_corrs together to accelerate calculating
        KLDivLoss = nn.KLDivLoss(reduction='none')
        _cls_corrs = torch.tensor(cls_corrs)
        _result = torch.tensor(np.mean(np.array(result), axis=0)).unsqueeze(0)
        p = torch.nn.functional.softmax(_result, -1)
        q = torch.nn.functional.softmax(_cls_corrs, -1)
        # p = _result / torch.sum(_result, dim=1).unsqueeze(1)
        # q = _cls_corrs / torch.sum(_cls_corrs, dim=1).unsqueeze(1)
        log_mean = ((p + q) / 2).log()
        jsdiv = torch.sum(KLDivLoss(log_mean, p), dim=1) / 2 + torch.sum(KLDivLoss(log_mean, q), dim=1) / 2
        jsdiv[cls_inds] = -1
        max_ind = torch.argmax(jsdiv).item()
        cls_inds.append(max_ind)
    return cls_inds


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
    if 'voc' in args.dataset:
        init_num = 500
        budget_num = 500
        if 'retina' in args.model:
            init_num = 1000
            budget_num = 500
    else:
        init_num = 5000
        budget_num = 1000
    indices = list(range(num_images))
    random.shuffle(indices)
    if args.init:
        unlabeled_loader = DataLoader(dataset_aug, batch_size=1, sampler=SubsetSequentialSampler(indices),
                                      num_workers=args.workers, collate_fn=utils.collate_fn)
        pre_model = fasterrcnn_resnet50_fpn_feature(num_classes=num_classes, min_size=600, max_size=1000)
        pre_model.to(device)
        uncertainty = init_uncertainty(pre_model, unlabeled_loader)
        arg = np.argsort(uncertainty)
        labeled_set = list(torch.tensor(indices)[arg][2000:2500].numpy())
        unlabeled_set = list(set(indices) - set(labeled_set))
    else:
        labeled_set = indices[:init_num]
        unlabeled_set = list(set(indices) - set(labeled_set))
    train_sampler = SubsetRandomSampler(labeled_set)
    data_loader_test = DataLoader(dataset_test, batch_size=1, sampler=SequentialSampler(dataset_test),
                                  num_workers=args.workers, collate_fn=utils.collate_fn)
    augs = ['flip', 'cut_out', 'smaller_resize', 'rotation']  # 'larger_resize'
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
                task_model = retinanet_resnet50_fpn_cal(num_classes=num_classes, min_size=600, max_size=1000)
        else:
            if 'faster' in args.model:
                task_model = fasterrcnn_resnet50_fpn_feature(num_classes=num_classes, min_size=800, max_size=1333)
            elif 'retina' in args.model:
                task_model = retinanet_resnet50_fpn_cal(num_classes=num_classes, min_size=800, max_size=1333)
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
                    voc_evaluate(task_model, data_loader_test, args.dataset, False, path=args.results_path)
                return
            print("Getting stability")
            random.shuffle(unlabeled_set)
            if 'coco' in args.dataset:
                subset = unlabeled_set[:5000]
            else:
                subset = unlabeled_set
            if args.mutual:
                unlabeled_loader = DataLoader(dataset_aug, batch_size=1, sampler=SubsetSequentialSampler(subset),
                                              num_workers=args.workers, pin_memory=True, collate_fn=utils.collate_fn)
                uncertainty, _cls_corrs = get_uncertainty(task_model, unlabeled_loader, augs, num_classes)
                # labeled_loader = DataLoader(dataset_aug, batch_size=1, sampler=SubsetSequentialSampler(labeled_set),
                #                             num_workers=args.workers, pin_memory=True, collate_fn=utils.collate_fn)
                # u, _ = get_uncertainty(task_model, labeled_loader, augs, num_classes)
                # with open("vis/cal_labeled_metric_{}_{}_{}.pkl".format(args.model, args.dataset, cycle),
                #           "wb") as fp:  # Pickling
                #     pickle.dump(u, fp)
                arg = np.argsort(np.array(uncertainty))
                # with open("vis/cal_unlabeled_metric_{}_{}_{}.pkl".format(args.model, args.dataset, cycle),
                #           "wb") as fp:  # Pickling
                #     pickle.dump(torch.tensor(uncertainty)[arg][:budget_num].numpy(), fp)
                cls_corrs_set = arg[:int(args.mr * budget_num)]
                cls_corrs = [_cls_corrs[i] for i in cls_corrs_set]
                labeled_loader = DataLoader(dataset_aug, batch_size=1, sampler=SubsetSequentialSampler(labeled_set),
                                            num_workers=args.workers, pin_memory=True, collate_fn=utils.collate_fn)
                tobe_labeled_set = cls_kldiv(labeled_loader, cls_corrs, budget_num)
                # Update the labeled dataset and the unlabeled dataset, respectively
                tobe_labeled_set = list(torch.tensor(subset)[arg][tobe_labeled_set].numpy())
                labeled_set += tobe_labeled_set
                unlabeled_set = list(set(indices) - set(labeled_set))
            else:
                unlabeled_loader = DataLoader(dataset_aug, batch_size=1, sampler=SubsetSequentialSampler(subset),
                                              num_workers=args.workers, pin_memory=True, collate_fn=utils.collate_fn)
                uncertainty, _ = get_uncertainty(task_model, unlabeled_loader, augs, num_classes)
                arg = np.argsort(np.array(uncertainty))
                # Update the labeled dataset and the unlabeled dataset, respectively
                labeled_set += list(torch.tensor(subset)[arg][:budget_num].numpy())
                labeled_set = list(set(labeled_set))
                unlabeled_set = list(set(indices) - set(labeled_set))

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
                voc_evaluate(task_model, data_loader_test, args.dataset, False, path=args.results_path)
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
                    voc_evaluate(task_model, data_loader_test, args.dataset, False, path=args.results_path)
        # if not args.skip and cycle == 0:
        #     if 'faster' in args.model:
        #         utils.save_on_master({
        #             'model': task_model.state_dict(), 'args': args},
        #             os.path.join(args.first_checkpoint_path, '{}_frcnn_1st.pth'.format(args.dataset)))
        #     elif 'retina' in args.model:
        #         utils.save_on_master({
        #             'model': task_model.state_dict(), 'args': args},
        #             os.path.join(args.first_checkpoint_path, '{}_retinanet_1st.pth'.format(args.dataset)))
        random.shuffle(unlabeled_set)
        if 'coco' in args.dataset:
            subset = unlabeled_set[:5000]
        else:
            subset = unlabeled_set
        print("Getting stability")
        if args.mutual:
            unlabeled_loader = DataLoader(dataset_aug, batch_size=1, sampler=SubsetSequentialSampler(subset),
                                          num_workers=args.workers, pin_memory=True, collate_fn=utils.collate_fn)
            uncertainty, _cls_corrs = get_uncertainty(task_model, unlabeled_loader, augs, num_classes)
            labeled_loader = DataLoader(dataset_aug, batch_size=1, sampler=SubsetSequentialSampler(labeled_set),
                                        num_workers=args.workers, pin_memory=True, collate_fn=utils.collate_fn)
            # u, _ = get_uncertainty(task_model, labeled_loader, augs, num_classes)
            # with open("vis/cal_labeled_metric_{}_{}_{}.pkl".format(args.model, args.dataset, cycle),
            #           "wb") as fp:  # Pickling
            #     pickle.dump(u, fp)
            arg = np.argsort(np.array(uncertainty))
            # with open("vis/cal_unlabeled_metric_{}_{}_{}.pkl".format(args.model, args.dataset, cycle),
            #           "wb") as fp:  # Pickling
            #     pickle.dump(torch.tensor(uncertainty)[arg][:budget_num].numpy(), fp)
            cls_corrs_set = arg[:int(args.mr * budget_num)]
            cls_corrs = [_cls_corrs[i] for i in cls_corrs_set]
            labeled_loader = DataLoader(dataset_aug, batch_size=1, sampler=SubsetSequentialSampler(labeled_set),
                                        num_workers=args.workers, pin_memory=True, collate_fn=utils.collate_fn)
            tobe_labeled_set = cls_kldiv(labeled_loader, cls_corrs, budget_num)
            # Update the labeled dataset and the unlabeled dataset, respectively
            tobe_labeled_set = list(torch.tensor(subset)[arg][tobe_labeled_set].numpy())
            labeled_set += tobe_labeled_set
            # with open("vis/cal_{}_{}_{}.txt".format(args.model, args.dataset, cycle), "wb") as fp:  # Pickling
            #     pickle.dump(labeled_set, fp)
            unlabeled_set = list(set(indices) - set(labeled_set))
        else:
            unlabeled_loader = DataLoader(dataset_aug, batch_size=1, sampler=SubsetSequentialSampler(subset),
                                          num_workers=args.workers, pin_memory=True, collate_fn=utils.collate_fn)
            uncertainty, _ = get_uncertainty(task_model, unlabeled_loader, augs, num_classes)
            arg = np.argsort(np.array(uncertainty))
            # Update the labeled dataset and the unlabeled dataset, respectively
            labeled_set += list(torch.tensor(subset)[arg][:budget_num].numpy())
            labeled_set = list(set(labeled_set))
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
    parser.add_argument('--model', default='fasterrcnn_resnet50_fpn', help='model')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=4, type=int,
                        help='images per gpu, the total batch size is $NGPU x batch_size')
    parser.add_argument('-cp', '--first-checkpoint-path', default='/data/yuweiping/coco/',
                        help='path to save checkpoint of first cycle')
    parser.add_argument('--task_epochs', default=26, type=int, metavar='N',
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
    parser.add_argument('--lr-steps', default=[16, 22], nargs='+', type=int, help='decrease lr every step-size epochs')
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
    parser.add_argument('-bp', default=1.2, type=float, help='base point')
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
