import torch
import torchvision
import torch.utils.data as data
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

import numpy as np

# from torchvision.ops import roi_pool, RoIPool, roi_align
from torchvision.ops import roi_pool, RoIPool, roi_align

class WSDDN(nn.Module):
    n_classes = 20
    classes = np.asarray([
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
        'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ])

    def __init__(self, classes=None):
        super(WSDDN, self).__init__()

        if classes is not None:
            self.classes = classes
            self.n_classes = len(classes)
            print(classes)

        roiPoolSize = 6

        #TODO: Define the WSDDN model
        self.features   = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True), # pool5 exists after ReLU in AlexNet
            )
        self.roi_pool   = RoIPool((roiPoolSize,roiPoolSize), 1.0/16)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(roiPoolSize*roiPoolSize*256, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            )

        self.score_fc   = nn.Linear(4096, self.n_classes)
        self.bbox_fc    = nn.Linear(4096, self.n_classes)

        
        # loss
        self.cross_entropy = None


    @property
    def loss(self):
        return self.cross_entropy

    def forward(self,
                image,
                rois=None,
                gt_vec=None,
                ):
        

        #TODO: Use image and rois as input
        # compute cls_prob which are N_roi X 20 scores

        imgSize = image.size()[2]

        out = self.features(image)
        # out = self.roi_pool(out, list(rois[0]*imgSize))
        out = self.roi_pool(out, list(rois*imgSize))

        # flatten output
        out = out.view(len(rois[0]), -1)
        out = self.classifier(out)

        out_fc = self.score_fc(out)
        out_bbox = self.bbox_fc(out)

        fc_scores = F.softmax(out_fc, dim=1)
        bbox_scores = F.softmax(out_bbox, dim=0)
        
        cls_prob = fc_scores*bbox_scores


        if self.training:
            label_vec = gt_vec.view(self.n_classes, -1)
            self.cross_entropy = self.build_loss(cls_prob, label_vec)
        
        return cls_prob

    
    def build_loss(self, cls_prob, label_vec):
        """Computes the loss

        :cls_prob: N_roix20 output scores
        :label_vec: 1x20 one hot label vector
        :returns: loss

        """
        #TODO: Compute the appropriate loss using the cls_prob that is the
        #output of forward()
        #Checkout forward() to see how it is called

        bceFunc = nn.BCELoss(reduction="sum")

        image_level_scores = torch.sum(cls_prob, dim=0)
        image_level_scores = torch.clamp(image_level_scores, min=0.0, max=1.0)
        loss = bceFunc(image_level_scores, torch.squeeze(label_vec))


        return loss
