import os
import random
import time
import copy

import torch

import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np


#TODO: given bounding boxes and corresponding scores, perform non max suppression
def nms(bounding_boxes, confidence_score, threshold=0.05):
    """
    bounding boxes of shape     Nx4
    confidence scores of shape  N
    threshold: confidence threshold for boxes to be considered

    return: list of bounding boxes and scores
    """

    # extract rois above the predetermined threshold
    boolScoreThresh = confidence_score >= threshold
    indicesScoreThresh = torch.where(boolScoreThresh)

    boxes = []
    scores = []

    if type(indicesScoreThresh) is tuple:
        indicesScoreThresh = indicesScoreThresh[0]
    
    indicesScoreThresh = indicesScoreThresh.squeeze()

    # continue to next loop if the size is zero
    if indicesScoreThresh.nelement() == 1:
        boxes = [list(bounding_boxes[0][indicesScoreThresh].detach().cpu().numpy().astype("float64"))]
        scores = [float(confidence_score[indicesScoreThresh].detach().cpu().numpy().astype("float64"))]
    elif indicesScoreThresh.nelement() == 2:
        scoreThresh = confidence_score[indicesScoreThresh]
        roisThresh = bounding_boxes[0][indicesScoreThresh]

        indicesLargeSmall = torch.argsort(scoreThresh, descending=True)
        indicesLargeSmall = indicesLargeSmall.squeeze()

        allowedIndices = [int(indicesLargeSmall[0])]

        box1 = roisThresh[int(indicesLargeSmall[0])]
        box2 = roisThresh[int(indicesLargeSmall[1])]

        iouScore = iou(box1, box2)

        if iouScore < 0.3:
            allowedIndices.append(int(indicesLargeSmall[1]))
        
        boxes = list(roisThresh[allowedIndices].detach().cpu().numpy().astype("float64"))
        scores = list(scoreThresh[allowedIndices].detach().cpu().numpy().astype("float64"))

    elif indicesScoreThresh.nelement() > 2:

        scoreThresh = confidence_score[indicesScoreThresh]
        roisThresh = bounding_boxes[0][indicesScoreThresh]

        indicesLargeSmall = torch.argsort(scoreThresh, descending=True)
        indicesLargeSmall = indicesLargeSmall.squeeze()

        allowedIndices = [int(indicesLargeSmall[0])]

        for i in range(scoreThresh.nelement()-1):
            box1 = roisThresh[indicesLargeSmall[i]]

            # compare box1 to all other boxes that are below it in terms of score
            for j in range(i+1, scoreThresh.nelement()):
                if int(indicesLargeSmall[j]) in allowedIndices:
                    continue

                box2 =  roisThresh[indicesLargeSmall[j]]
                iouScore = iou(box1, box2)
                if iouScore<0.3:
                    allowedIndices.append(int(indicesLargeSmall[j]))

        if len(allowedIndices) == 1:
            boxes = roisThresh[allowedIndices[0]].detach().cpu().numpy()
            scores = scoreThresh[allowedIndices[0]].detach().cpu().numpy()
            return list(boxes), [float(scores)]
        else:
            boxes = roisThresh[allowedIndices]
            scores = scoreThresh[allowedIndices]

        # sort through boxes and scores by highest to lowest scores
        indicesLargeSmall = torch.argsort(scores, descending=True).squeeze()
        boxes = list(boxes[indicesLargeSmall].detach().cpu().numpy().astype("float64"))
        scores = list(scores[indicesLargeSmall].detach().cpu().numpy().astype("float64"))


    return boxes, scores


#TODO: calculate the intersection over union of two boxes
def iou(box1, box2):
    """
    Calculates Intersection over Union for two bounding boxes (xmin, ymin, xmax, ymax)
    returns IoU vallue
    """
    box1 = torch.tensor(box1).detach().cpu()
    box2 = torch.tensor(box2).detach().cpu()

    if box1.numpy().ndim > 1:
        box1 = box1[0]
    
    if box2.numpy().ndim > 1:
        box2 = box2[0]

    # box1 is the bounding box with the highest confidence score
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = max(box1[2], box2[2])
    yB = max(box1[3], box2[3])

    inter = max((xB-xA), 0) * max((yB-yA), 0)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2

    iou = inter / (union - inter)

    return iou


def tensor_to_PIL(image):
    """
    converts a tensor normalized image (imagenet mean & std) into a PIL RGB image
    will not work with batches (if batch size is 1, squeeze before using this)
    """
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
        std=[1/0.229, 1/0.224, 1/0.255],
    )

    inv_tensor = inv_normalize(image)
    inv_tensor = torch.clamp(inv_tensor, 0, 1)
    original_image = transforms.ToPILImage()(inv_tensor).convert("RGB")

    return original_image


def get_box_data(classes, bbox_coordinates):
    """
    classes : tensor containing class predictions/gt
    bbox_coordinates: tensor containing [[xmin0, ymin0, xmax0, ymax0], [xmin1, ymin1, ...]] (Nx4)

    return list of boxes as expected by the wandb bbox plotter
    """
    box_list = [{
            "position": {
                "minX": bbox_coordinates[i][0],
                "minY": bbox_coordinates[i][1],
                "maxX": bbox_coordinates[i][2],
                "maxY": bbox_coordinates[i][3],
            },
            "class_id" : classes[i],
        } for i in range(len(classes))
        ]

    return box_list




