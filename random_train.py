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

from detection.coco_utils import get_coco, get_coco_kp
from detection.group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
from detection.engine import coco_evaluate, voc_evaluate
from detection import utils
from detection import transforms as T
from detection.train import *
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn


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
            print(task_loss_dict_reduced)
            sys.exit(1)

        task_optimizer.zero_grad()
        losses.backward()
        task_optimizer.step()
        if task_lr_scheduler is not None:
            task_lr_scheduler.step()
        metric_logger.update(task_loss=task_losses_reduced)
        metric_logger.update(task_lr=task_optimizer.param_groups[0]["lr"])
    return metric_logger


def main(args):
    torch.cuda.set_device(0)
    random.seed('ywp')
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
        dataset_test, _ = get_dataset(args.dataset, "test", get_transform(train=False), args.data_path)
    else:
        dataset, num_classes = get_dataset(args.dataset, "train", get_transform(train=True), args.data_path)
        dataset_test, _ = get_dataset(args.dataset, "val", get_transform(train=False), args.data_path)
    print("Creating data loaders")
    num_images = len(dataset)
    indices = list(range(num_images))
    random.shuffle(indices)
    labeled_set = indices[:int(num_images * 0.1)]
    labeled_set = [60, 2084, 1878, 834, 3865, 2576, 1883, 304, 619, 4865, 9, 646, 1090, 3840, 2242, 3729, 3593, 4317,
                   1498, 3475, 4176, 806, 3832, 2385, 4033, 4889, 4839, 3877, 4468, 4251, 3465, 4457, 1884, 3120, 2424,
                   4052, 229, 330, 1343, 4731, 3505, 2610, 177, 849, 1714, 3077, 4024, 868, 1467, 1673, 4125, 2267,
                   1536, 3720, 1872, 216, 4107, 142, 106, 1219, 1406, 3883, 3209, 2244, 4882, 3483, 1354, 236, 3576,
                   544, 1281, 1561, 2433, 4784, 1747, 919, 400, 2741, 2747, 4389, 2899, 971, 1450, 2115, 457, 1647,
                   4646, 2286, 1506, 2043, 2467, 1499, 3698, 439, 1083, 2798, 1221, 855, 3050, 4764, 2451, 2485, 2624,
                   1586, 2793, 3411, 64, 4677, 4048, 2398, 4118, 3464, 4692, 4167, 4712, 2730, 219, 2579, 1754, 4246,
                   891, 875, 2162, 2153, 4131, 3538, 3347, 3172, 3146, 3087, 1925, 1839, 2476, 551, 2717, 4382, 1087,
                   239, 4065, 1476, 4982, 79, 4394, 2764, 3524, 2426, 3144, 1101, 4419, 1430, 1102, 702, 3229, 847,
                   4513, 3151, 1776, 4136, 357, 1497, 4519, 2057, 2936, 3011, 2036, 4910, 4521, 3105, 4749, 755, 817,
                   2112, 3808, 1842, 3797, 2696, 2463, 4605, 3934, 2155, 808, 854, 2083, 1212, 369, 1840, 2458, 4358,
                   2613, 2163, 4197, 24, 861, 4815, 2348, 335, 1468, 3037, 1263, 4860, 1798, 2317, 4658, 2223, 3567,
                   844, 4462, 34, 3236, 1882, 648, 1733, 783, 705, 3900, 556, 4894, 4220, 4110, 2349, 3030, 4960, 4430,
                   4769, 1577, 570, 249, 1590, 4352, 2266, 186, 4215, 4005, 793, 320, 4418, 4026, 827, 172, 4122, 4347,
                   3014, 456, 31, 4175, 1765, 4283, 1225, 714, 4422, 4561, 2686, 824, 3010, 833, 3994, 4330, 1082, 2997,
                   2246, 2675, 4380, 3277, 2350, 968, 3430, 3251, 1095, 565, 3499, 989, 2739, 1606, 3932, 852, 3764,
                   918, 4905, 1378, 2226, 248, 3170, 3055, 298, 423, 4661, 251, 1017, 4667, 4640, 2132, 2748, 3938,
                   1206, 2142, 1148, 829, 2847, 684, 936, 1053, 3613, 1591, 2276, 997, 1452, 4477, 2697, 3834, 411,
                   1072, 2119, 2506, 1262, 742, 200, 3131, 4289, 1309, 2833, 2092, 4626, 4798, 4628, 1487, 2551, 1014,
                   4654, 3299, 772, 1437, 3363, 3552, 1566, 2377, 4343, 1887, 3872, 4170, 2895, 4121, 2760, 3985, 3515,
                   4001, 1408, 408, 2314, 2750, 3217, 1144, 2442, 1650, 3141, 3416, 3235, 4595, 2945, 1996, 4566, 1809,
                   228, 3484, 525, 2280, 4164, 1257, 3374, 4143, 4767, 464, 2015, 4664, 1608, 2405, 4630, 2372, 2178,
                   4431, 84, 2950, 4320, 4492, 2894, 3743, 1810, 4707, 2396, 3233, 1812, 1571, 1451, 145, 3395, 4019,
                   2356, 603, 3385, 3002, 4433, 962, 1463, 639, 4785, 2011, 1347, 1122, 1945, 621, 2766, 4738, 3951,
                   4941, 787, 1539, 2660, 474, 2526, 262, 3220, 277, 1393, 2429, 1699, 4309, 2919, 1298, 2870, 3015,
                   421, 1689, 3730, 1986, 4461, 473, 3591, 4010, 656, 2595, 583, 497, 2323, 2584, 1858, 3930, 3844,
                   4644, 4181, 2723, 118, 196, 4219, 209, 2848, 3621, 1003, 97, 1190, 3972, 1848, 373, 5005, 1198, 985,
                   4705, 4311, 2872, 4736, 3246, 4689, 4135, 4318, 2626, 4981, 1723, 2378, 1527, 1731, 2234, 2634, 3556,
                   311, 4039, 4387, 3774, 3310, 2859, 1865, 1292, 3069, 99, 735, 4279, 4203, 935, 3455, 4895, 3215,
                   2131, 1112, 2740]

    unlabeled_set = indices[int(num_images * 0.1):]
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
        task_model = fasterrcnn_resnet50_fpn(num_classes=num_classes, min_size=600, max_size=1000)
        task_model.to(device)
        if cycle == 8:
            checkpoint = torch.load(os.path.join('basemodel', 'voc2007_frcnn_1st.pth'), map_location='cpu')
            task_model.load_state_dict(checkpoint['model'])
            # if 'coco' in args.dataset:
            #     coco_evaluate(task_model, data_loader_test)
            # elif 'voc' in args.dataset:
            #     voc_evaluate(task_model, data_loader_test, args.dataset)
            random.shuffle(unlabeled_set)
            # Update the labeled dataset and the unlabeled dataset, respectively
            labeled_set += unlabeled_set[:int(0.05 * num_images)]
            unlabeled_set = unlabeled_set[int(0.05 * num_images):]

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
                voc_evaluate(task_model, data_loader_test, args.dataset)
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
                    voc_evaluate(task_model, data_loader_test, args.dataset)
        random.shuffle(unlabeled_set)
        # Update the labeled dataset and the unlabeled dataset, respectively
        labeled_set += unlabeled_set[:int(0.05 * num_images)]
        unlabeled_set = unlabeled_set[int(0.05 * num_images):]

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

    main(args)
