import torch
import torchvision
from torchvision import utils as vutils
import numpy as np
import random
import cv2


def get_uncertainty(task_model, unlabeled_loader):
    task_model.eval()
    task_model.ssm_mode(True)
    allBox = []
    allScore = []
    allY = []
    al_idx = []
    with torch.no_grad():
        for i, (images, _) in enumerate(unlabeled_loader):
            images = list(img.cuda() for img in images)
            dets = task_model(images)
            # only support batch_size=1 when testing
            boxes = dets[0]['boxes']
            scores = dets[0]['scores']
            labels = dets[0]['labels']
            al = dets[0]['al']
            if al == 1:
                al_idx.append(i)
                # print(scores)
                continue
            allBox.append(boxes)
            allScore.append(scores)
            allY.append(labels)

    return allScore, allBox, allY, al_idx


def judge_uv(loss, gamma, clslambda):
    '''
    return
    u: scalar
    v: R^kind vector
    '''
    lsum = np.sum(loss)
    dim = loss.shape[0]
    v_val = np.zeros((dim,))

    if (lsum > gamma):
        return False, v_val
    elif lsum < gamma:
        for i, l in enumerate(loss):
            if l > clslambda[i]:
                v_val[i] = 0
            else:
                v_val[i] = 1 - l / clslambda[i]
    return True, v_val


@torch.no_grad()
def image_cross_validation(model, curr_loader, labeled_sampler, pre_box, pre_cls):
    '''
    implement image cross validation function
    to choose the highest consistant proposal
    '''
    model.eval()
    model.ssm_mode(False)
    total_select = 5  # total_select images to paste
    curr_select = 0
    cross_validation = 0
    avg_score = 0
    for images, _ in curr_loader:
        curr_img = list(img.cuda() for img in images)[0]
    # crop proposal from image
    unlabeled_patch = curr_img[:, int(pre_box[0]):int(pre_box[2]), int(pre_box[1]):int(pre_box[3])]
    if unlabeled_patch.shape[1] <= 0 or unlabeled_patch.shape[2] <= 0:
        return False, 0
    for images, targets in labeled_sampler:
        image = list(image.cuda() for image in images)[0]
        target = [{k: v.cuda() for k, v in t.items()} for t in targets][0]
        labeled_img = image
        labeled_cls = target['labels']
        # select image
        if pre_cls not in labeled_cls.cpu().numpy():
            if unlabeled_patch.shape[1] > labeled_img.shape[1] or unlabeled_patch.shape[2] > labeled_img.shape[2]:
                continue
            start_y = random.randint(0, labeled_img.shape[1] - unlabeled_patch.shape[1])
            start_x = random.randint(0, labeled_img.shape[2] - unlabeled_patch.shape[2])
            original_box = [start_x, start_y, start_x + unlabeled_patch.shape[2], start_y + unlabeled_patch.shape[1]]
            labeled_img[:, start_y:start_y + unlabeled_patch.shape[1], start_x:start_x + unlabeled_patch.shape[2]] \
                = unlabeled_patch
            # redetect pasted_image
            dets = model([labeled_img])
            labels = dets[0]['labels']
            boxes = dets[0]['boxes'][labels == pre_cls]
            scores = dets[0]['scores'][labels == pre_cls]
            if len(boxes) == 0:
                continue
            index = torch.argmax(scores)
            score = scores[index]
            box = boxes[index]
            overlap_iou = calcu_iou(original_box, box)
            curr_select += 1
            if score > 0.5 and overlap_iou > 0.5:
                cross_validation += 1
                avg_score += score
            if curr_select >= total_select:
                break
        else:
            continue
    if cross_validation > total_select / 2:
        return True, avg_score / cross_validation
    else:
        return False, 0


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
