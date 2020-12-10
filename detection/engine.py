import math
import sys
import time
import torch
import itertools
from terminaltables import AsciiTable
import numpy as np
import cv2
from mmcv.utils import print_log

import torchvision.models.detection.mask_rcnn

from .coco_utils import get_coco_api_from_dataset
from .coco_eval import CocoEvaluator
from . import utils
from .voc_eval import _write_voc_results_file, _do_python_eval


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


VOC_CLASSES = (
    "aeroplane", "bicycle",
    "bird", "boat", "bottle", "bus", "car",
    "cat", "chair", "cow", "diningtable", "dog",
    "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor",
)


@torch.no_grad()
def voc_evaluate(model, data_loader, year):
    device = 'cuda'
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    all_boxes = [[] for i in range(21)]
    image_index = []
    c = 0
    for image, targets in metric_logger.log_every(data_loader, 5000, header):
        image = list(img.to(device) for img in image)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        torch.cuda.synchronize()
        outputs = model(image)

        name = ''.join([chr(i) for i in targets[0]['name'].tolist()])
        image_index.append(name)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

        image_boxes = [[] for i in range(21)]
        for o in outputs:
            for i in range(o['boxes'].shape[0]):
                image_boxes[o['labels'][i]].extend([torch.cat([o['boxes'][i], o['scores'][i].unsqueeze(0)], dim=0)])
        # if cycle == 0:
        #     for img, label, out in zip(image, targets, outputs):
        #         img = (img * 255).permute(1, 2, 0).type(torch.uint8).cpu().numpy()
        #         img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        #         for b, l in zip(label['boxes'], label['labels']):
        #             cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 255, 0))
        #             cv2.putText(img, VOC_CLASSES[l - 1], (b[0], b[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
        #                         color=(0, 255, 0), thickness=1)
        #         for b, l, s in zip(out['boxes'], out['labels'], out['scores']):
        #             if s > 0.3:
        #                 cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255))
        #                 cv2.putText(img, VOC_CLASSES[l - 1] + ':' + str(np.round(s.item(), 2)),
        #                             (int(b[0]), int(b[3] - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color=(0, 0, 255),
        #                             thickness=1)
        #     cv2.imwrite('/data/yuweiping/vis_voc_cycle_1/{}.jpg'.format(i), img)
        #     c += 1
        # makes sure that the all_boxes is filled with empty array when
        # there are no boxes in image_boxes
        for i in range(21):
            if image_boxes[i] != []:
                all_boxes[i].append([torch.stack(image_boxes[i])])
            else:
                all_boxes[i].append([])
    metric_logger.synchronize_between_processes()

    all_boxes_gathered = utils.all_gather(all_boxes)
    image_index_gathered = utils.all_gather(image_index)

    # results from all processes are gathered here
    if utils.is_main_process():
        all_boxes = [[] for i in range(21)]
        for abgs in all_boxes_gathered:
            for ab, abg in zip(all_boxes, abgs):
                ab += abg
        image_index = []
        for iig in image_index_gathered:
            image_index += iig
        _write_voc_results_file(all_boxes, image_index, data_loader.dataset.root,
                                data_loader.dataset._transforms.transforms[0].CLASSES)
        _do_python_eval(data_loader, year)
    torch.set_num_threads(n_threads)


COCO_CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
                'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
                'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')


@torch.no_grad()
def coco_evaluate(model, data_loader, classwise=True):
    device = 'cuda'
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)
    for images, targets in metric_logger.log_every(data_loader, 1000, header):
        images = list(img.to(device) for img in images)
        torch.cuda.synchronize()
        model_time = time.time()
        _, outputs = model(images)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    # coco_evaluator.evaluate()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    cat_ids = coco.get_cat_ids(cat_names=COCO_CLASSES)
    if classwise:  # Compute per-category AP
        # Compute per-category AP
        # from https://github.com/facebookresearch/detectron2/
        precisions = coco_evaluator.coco_eval['bbox'].eval['precision']
        # precision: (iou, recall, cls, area range, max dets)
        assert len(cat_ids) == precisions.shape[2]

        results_per_category = []
        for idx, catId in enumerate(cat_ids):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            nm = coco.loadCats(catId)[0]
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            if precision.size:
                ap = np.mean(precision)
            else:
                ap = float('nan')
            results_per_category.append(
                (f'{nm["name"]}', f'{float(ap):0.3f}'))

        num_columns = min(6, len(results_per_category) * 2)
        results_flatten = list(
            itertools.chain(*results_per_category))
        headers = ['category', 'AP'] * (num_columns // 2)
        results_2d = itertools.zip_longest(*[
            results_flatten[i::num_columns]
            for i in range(num_columns)
        ])
        table_data = [headers]
        table_data += [result for result in results_2d]
        table = AsciiTable(table_data)
        print_log('\n' + table.table)
    torch.set_num_threads(n_threads)
    return coco_evaluator
