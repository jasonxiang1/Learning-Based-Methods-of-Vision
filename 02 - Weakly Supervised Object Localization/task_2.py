from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import torch.utils.model_zoo as model_zoo
from torch.nn.parameter import Parameter
import numpy as np
from datetime import datetime
import pickle as pkl

# imports
from wsddn import WSDDN
from voc_dataset import *
import wandb
from utils import nms, iou, tensor_to_PIL, get_box_data
from PIL import Image, ImageDraw


# hyper-parameters
# ------------
start_step = 0
end_step = 20000
lr_decay_steps = {150000}
lr_decay = 1. / 10
rand_seed = 1024

lr = 0.0001
momentum = 0.9
weight_decay = 0.0005
# ------------

USE_WANDB = True

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def test_net(model, epoch, val_loader=None, thresh=0.05):
    """
    tests the networks and visualize the detections
    thresh is the confidence threshold
    """

    ap7 = []
    ap11 = []
    ap17 = []
    mAP = []

    apArr = [7, 11, 17]
    randomImageIndex = [12, 55, 75, 104, 150, 211, 444, 555, 645, 791, 994, 1064, 1086, 1423, 1500, 1701, 2153, 2643, 3259]
    epochArr = [0, 5, 6]

    class_id_to_label = dict(enumerate(val_loader.dataset.CLASS_NAMES))
    imgTransform = transforms.Compose([transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224 , 1/0.225]), transforms.ToPILImage(), transforms.Resize((512,512))])

    for iter, data in enumerate(val_loader):

        # one batch = data for one image
        image           = data['image'].to('cuda')
        target          = data['label'].to('cuda')
        wgt             = data['wgt'].to('cuda')
        rois            = data['rois'].float().to('cuda')
        gt_boxes        = data['gt_boxes']
        gt_class_list   = data['gt_classes']

        #TODO: perform forward pass, compute cls_probs
        output = model.forward(image, rois, target)

        ap = []

        epsilon = 1e-5

        # TODO: Iterate over each class (follow comments)
        for class_num in range(20):            
            # get valid rois and cls_scores based on thresh
            # extract rois from the specific class
            scoreClassIter = output[:,class_num]

            # use NMS to get boxes and scores
            boxes, scores = nms(rois, scoreClassIter, threshold=thresh)

            # if class does not appear here, then continue
            # else create a one hot vector for the ground truth class
            if class_num not in gt_class_list:
                continue
            else:
                if scores==[] or boxes==[] or boxes==[[]]:
                    if class_num == apArr[0]:
                        ap7.append(0)
                    elif class_num == apArr[1]:
                        ap11.append(0)
                    elif class_num == apArr[2]:
                        ap17.append(0)
                    ap.append(0)
                    continue

                scoresTensor = torch.tensor(scores)
                boxesTensor = torch.tensor(boxes)

                truePos = torch.zeros(scoresTensor.nelement())
                falsePos = torch.zeros(scoresTensor.nelement())

                # determine how many ground truth boxes are there for this class
                gtClassOneHot = torch.where(torch.tensor(gt_class_list) == torch.tensor(class_num), torch.tensor(1) ,torch.tensor(0))
                totalgtClass = torch.sum(gtClassOneHot)
                detectionFound = torch.zeros(len(gtClassOneHot))

                for proposal in range(scoresTensor.nelement()):

                    best_iou = 0

                    for gtBox in range(len(gtClassOneHot)):
                        if 0 in gtClassOneHot[gtBox]:
                            continue

                        if boxesTensor.nelement() == 4:
                            tempIou = iou(gt_boxes[gtBox], boxes)
                        else:
                            tempIou = iou(gt_boxes[gtBox], boxes[proposal])
                        if tempIou > best_iou:
                            best_iou = tempIou
                            best_gt_idx = gtBox
                    if best_iou > 0.3:
                        if detectionFound[best_gt_idx] == 0:
                            truePos[proposal] = 1
                            detectionFound[best_gt_idx] = 1
                        else:
                            falsePos[proposal] = 1
                    else:
                        falsePos[proposal] = 1
                        
                truePosCumSum = torch.cumsum(truePos, dim=0)
                falsePosCumSum = torch.cumsum(falsePos, dim=0)

                recalls = truePosCumSum / (totalgtClass + epsilon)
                precisions = truePosCumSum / (truePosCumSum + falsePosCumSum + epsilon)
                precisions = torch.cat((torch.tensor([1]).float(), precisions))
                recalls = torch.cat((torch.tensor([0]).float(), recalls))

                if class_num == apArr[0]:
                    ap7.append(torch.trapz(precisions, recalls))
                elif class_num == apArr[1]:
                    ap11.append(torch.trapz(precisions, recalls))
                elif class_num == apArr[2]:
                    ap17.append(torch.trapz(precisions, recalls))

                ap.append(torch.trapz(precisions, recalls))
            
                # log bounding boxes over image of under image set and if under epoch set
                if USE_WANDB and iter in randomImageIndex and epoch in epochArr:
                    if np.array(boxes).ndim == 1:
                        boxes = [boxes]

                    classes = [class_num]*len(boxes)

                    img = wandb.Image(torch.squeeze(image).cpu(), boxes={
                                    "predictions": {
                                        "box_data": get_box_data(classes, list(np.array(boxes).astype("float64"))),
                                        "class_labels": class_id_to_label,
                                    },
                                })

                    wandb.log({"Epoch-"+str(epoch)+"-RandomImage-"+str(iter)+"-ClassNumber-"+str(class_num): img})



        
        mAP.append(torch.mean(torch.tensor(ap).float()))
            

        #TODO: visualize bounding box predictions when required

        #TODO: Calculate mAP on test set
        if USE_WANDB: 
            wandb.log({'epoch': epoch, 'test/mAP': torch.mean(torch.tensor(mAP).float()) })
            wandb.log({'epoch': epoch, 'test/class-index-7-AP': torch.mean(torch.tensor(ap7).float()) })
            wandb.log({'epoch': epoch, 'test/class-index-11-AP': torch.mean(torch.tensor(ap11).float()) })
            wandb.log({'epoch': epoch, 'test/class-index-17-AP': torch.mean(torch.tensor(ap17).float()) })

    return torch.mean(torch.tensor(mAP))

if __name__ == '__main__':

    if USE_WANDB:
        wandb.init(project="vlr2_task2", reinit=True)
    

    if rand_seed is not None:
        np.random.seed(rand_seed)

    # load datasets and create dataloaders

    train_dataset = VOCDataset(split='trainval',image_size=512, top_n = 200)
    val_dataset = VOCDataset(split='test',image_size=512, top_n = 200)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,   # batchsize is one for this implementation
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        sampler=None,
        drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=True)


    # Create network and initialize
    net = WSDDN(classes=train_dataset.CLASS_NAMES)
    print(net)

    if os.path.exists('pretrained_alexnet.pkl'):
        pret_net = pkl.load(open('pretrained_alexnet.pkl', 'rb'))
    else:
        pret_net = model_zoo.load_url(
            'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth')
        pkl.dump(pret_net, open('pretrained_alexnet.pkl', 'wb'),
                pkl.HIGHEST_PROTOCOL)
    own_state = net.state_dict()

    for name, param in pret_net.items():
        print(name)
        if name not in own_state:
            continue
        if isinstance(param, Parameter):
            param = param.data
        try:
            own_state[name].copy_(param)
            print('Copied {}'.format(name))
        except:
            print('Did not find {}'.format(name))
            continue


    # Move model to GPU and set train mode
    net.load_state_dict(own_state)
    net.cuda()
    net.train()

    # # TODO: Create optimizer for network parameters from conv2 onwards
    # # (do not optimize conv1)
    optimizer = torch.optim.SGD(list(net.parameters())[2:], lr=lr, momentum=momentum, weight_decay=weight_decay)




    output_dir = "./"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # training
    train_loss = 0
    tp, tf, fg, bg = 0., 0., 0, 0
    step_cnt = 0
    re_cnt = False
    disp_interval = 10
    val_interval = 1000
    start_epoch = 0
    end_epoch = 6

    for epoch in range(start_epoch, end_epoch):

        lossVar = AverageMeter()

        for iter, data in enumerate(train_loader):

            #TODO: get one batch and perform forward pass
            # one batch = data for one image
            image           = data['image'].to('cuda')
            target          = data['label'].to('cuda')
            wgt             = data['wgt'].to('cuda')
            rois            = data['rois'].float().to('cuda')
            gt_boxes        = data['gt_boxes']
            gt_class_list   = data['gt_classes']
            

            #TODO: perform forward pass - take care that proposal values should be in pixels for the fwd pass
            # also convert inputs to cuda if training on GPU
            net.training = True
            output = net.forward(image, rois, target)


            # backward pass and update
            loss = net.loss    
            train_loss += loss.item()
            step_cnt += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lossVar.update(loss.item(),1)
            print('train loss',lossVar.avg)

            #TODO: evaluate the model every N iterations (N defined in handout)
            
            if iter%val_interval == 0 and iter != 0:
                net.eval()
                ap = test_net(net, epoch, val_loader)
                print("AP ", ap)
                # print("Loss ", train_loss)
                print("Loss ", lossVar.avg)
                net.train()

            if iter%500 and USE_WANDB:
                wandb.log({'epoch':epoch, 'train/loss': lossVar.avg})


        #TODO: Perform all visualizations here
        #The intervals for different things are defined in the handout



