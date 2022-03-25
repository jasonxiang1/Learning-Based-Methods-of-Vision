import argparse
import os
import shutil
import time
import sys
import sklearn
import sklearn.metrics

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models


from AlexNet import *
from voc_dataset import *
from utils import *

import wandb
from PIL import Image
USE_WANDB = True # use flags, wandb is not convenient for debugging


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--arch', default='localizer_alexnet')
parser.add_argument(
    '-j',
    '--workers',
    default=4,
    type=int,
    metavar='N',
    help='number of data loading workers (default: 4)')
parser.add_argument(
    '--epochs',
    default=2, # TODO: default value is 30
    type=int,
    metavar='N',
    help='number of total epochs to run')
parser.add_argument(
    '--start-epoch',
    default=0,
    type=int,
    metavar='N',
    help='manual epoch number (useful on restarts)')
parser.add_argument(
    '-b',
    '--batch-size',
    default=32, # TODO: default value is 256
    type=int,
    metavar='N',
    help='mini-batch size (default: 256)')
parser.add_argument(
    '--lr',
    '--learning-rate',
    default=0.01, # TODO: default value is 0.1
    type=float,
    metavar='LR',
    help='initial learning rate')
parser.add_argument(
    '--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument(
    '--weight-decay',
    '--wd',
    default=1e-4,
    type=float,
    metavar='W',
    help='weight decay (default: 1e-4)')
parser.add_argument(
    '--print-freq',
    '-p',
    default=10,
    type=int,
    metavar='N',
    help='print frequency (default: 10)')
parser.add_argument(
    '--eval-freq',
    default=2,
    type=int,
    metavar='N',
    help='print frequency (default: 10)')
parser.add_argument(
    '--resume',
    default='',
    type=str,
    metavar='PATH',
    help='path to latest checkpoint (default: none)')
parser.add_argument(
    '-e',
    '--evaluate',
    dest='evaluate',
    action='store_true',
    help='evaluate model on validation set')
parser.add_argument(
    '--pretrained',
    dest='pretrained',
    action='store_false',
    help='use pre-trained model')
parser.add_argument(
    '--world-size',
    default=1,
    type=int,
    help='number of distributed processes')
parser.add_argument(
    '--dist-url',
    default='tcp://224.66.41.62:23456',
    type=str,
    help='url used to set up distributed training')
parser.add_argument(
    '--dist-backend', default='gloo', type=str, help='distributed backend')
parser.add_argument('--vis', action='store_true')

best_prec1 = 0

batchChannels = [30, 120]
epochChannels = [0, 14, 29, 44]
# epochChannels = [0, 1, 14, 29]


def main():
    global args, best_prec1
    args = parser.parse_args()
    args.distributed = args.world_size > 1

    # create model
    # args.arch  = 'localizer_alexnet_robust'
    print("=> creating model '{}'".format(args.arch))
    if args.arch == 'localizer_alexnet':
        model = localizer_alexnet(pretrained=args.pretrained)
    elif args.arch == 'localizer_alexnet_robust':
        model = localizer_alexnet_robust(pretrained=args.pretrained)
    print(model)

    model.features = torch.nn.DataParallel(model.features)
    model.cuda()

    # TODO:
    # define loss function (criterion) and optimizer
    # also use an LR scheduler to decay LR by 10 every 30 epochs
    # you can also use PlateauLR scheduler, which usually works well

    criterion = nn.BCEWithLogitsLoss().cuda()
    optimizer = torch.optim.SGD(model.classifier.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # TODO: fimplement PlateauLR scheduler
    # FORNOW: choose stepLR scheduler
    step_size = 10
    gamma = 0.1
    LRScheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True


    # Data loading code
    
    #TODO: Create Datasets and Dataloaders using VOCDataset - Ensure that the sizes are as required
    # Also ensure that data directories are correct - the ones use for testing by TAs might be different
    # Resize the images to 512x512

    train_dataset = VOCDataset(split='trainval',image_size=512, top_n = 30)
    val_dataset = VOCDataset(split='test',image_size=512, top_n = 30)



    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        # shuffle=(train_sampler is None),
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return
    
    # TODO: Create loggers for wandb - ideally, use flags since wandb makes it harder to debug code.
    if USE_WANDB:
        wandb.init(project="vlr2_task01", reinit=True)
        wandb.config.update(args)

    # get class names from dataset
    class_id_to_label = train_dataset.CLASS_NAMES


    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        if epoch % args.eval_freq == 0 or epoch == args.epochs - 1:
            m1, m2 = validate(val_loader, model, criterion, epoch)

            score = m1 * m2
            # remember best prec@1 and save checkpoint
            is_best = score > best_prec1
            best_prec1 = max(score, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
            }, is_best)

        if LRScheduler is not None:
            LRScheduler.step()
        # adjust_learning_rate(optimizer, epoch)        


#TODO: You can add input arguments if you wish
def train(train_loader, model, criterion, optimizer, epoch):
    #TODO add the inputs from the given reference code
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    avg_m1 = AverageMeter()
    avg_m2 = AverageMeter()


    # switch to train mode
    model.train()

    end = time.time()
    count = 0
    for i, (data) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = data['label'].to('cuda')
        wgt = data['wgt'].to('cuda')


        # TODO: Get inputs from the data dict
        img_input = data['image'].to('cuda')

        # TODO: Get output from model
        outputModel = model(img_input)


        # TODO: Perform any necessary functions on the output such as clamping
        if args.arch == 'localizer_alexnet':
            # output = maxPoolOutput(outputModel)
            maxPoolFunc = nn.MaxPool2d(kernel_size=outputModel.size()[-1])
            output = maxPoolFunc(outputModel)
            output = torch.squeeze(output)
        elif args.arch == 'localizer_alexnet_robust':
            avgPoolFunc = nn.AvgPool2d(kernel_size=outputModel.size()[-1])
            output = avgPoolFunc(outputModel)
            output = torch.squeeze(output)
        

        # TODO: Compute loss using ``criterion``
        loss = criterion(output, target*wgt)

        input = img_input


        # measure metrics and record loss
        m1 = metric1(output, target, wgt)
        m2 = metric2(output, target, wgt)
        losses.update(loss.item(), input.size(0))
        avg_m1.update(m1)
        avg_m2.update(m2)


        # TODO:
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Metric1 {avg_m1.val:.3f} ({avg_m1.avg:.3f})\t'
                  'Metric2 {avg_m2.val:.3f} ({avg_m2.avg:.3f})'.format(
                      epoch,
                      i,
                      len(train_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      avg_m1=avg_m1,
                      avg_m2=avg_m2))
        
        if i in batchChannels and USE_WANDB and epoch in epochChannels:
            # create heatmap based on all ground truth channels

            groundTruthIndices = torch.nonzero(target[0], as_tuple=True)

            transformInputImage = transforms.Compose([transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224 , 1/0.225]), transforms.ToPILImage(), transforms.Resize((512,512))])

            posImgInput = img_input[0].cpu()
            inputImgPIL = transformInputImage(posImgInput)


            wandb.log({"0th-Original-Image-Epoch-" + str(epoch) +"-batch-"+str(i) : [wandb.Image(inputImgPIL,caption="Image" )]})


            if isinstance(groundTruthIndices,tuple) == 0:
                groundTruthIndices = (groundTruthIndices)

            

            for idx in groundTruthIndices[0]:
                imgHeatMap = createHeatMap(input, outputModel, target, idx)
                wandb.log({"Epoch-"+str(epoch)+"-Batch-"+str(i)+"-Image-0-"+"Ground Truth Index-"+str(idx.item()): [wandb.Image(imgHeatMap.permute(1,2,0).numpy(),caption="Image")]})





    #TODO: Visualize/log things as mentioned in handout
    #TODO: Visualize at appropriate intervals
    if USE_WANDB and epoch%2==0:
        wandb.log({'epoch': epoch,'train/loss': losses.avg})
        wandb.log({'epoch': epoch,'train/metric1': avg_m1.avg})
        wandb.log({'epoch': epoch,'train/metric2': avg_m2.avg})


        # End of train()


def validate(val_loader, model, criterion, epoch = 0):
    batch_time = AverageMeter()
    losses = AverageMeter()
    avg_m1 = AverageMeter()
    avg_m2 = AverageMeter()


    # switch to evaluate mode
    model.eval()

    end = time.time()

    for i, (data) in enumerate(val_loader):

        # TODO: Get inputs from the data dict
        img_input = data['image'].to('cuda')
        
        target = data['label'].to('cuda')
        wgt = data['wgt'].to('cuda')

        # TODO: Get output from model
        outputModel = model(img_input)

        # TODO: Perform any necessary functions on the output
        if args.arch == 'localizer_alexnet':
            # output = maxPoolOutput(outputModel)
            maxPoolFunc = nn.MaxPool2d(kernel_size=outputModel.size()[-1])
            output = maxPoolFunc(outputModel)
            output = torch.squeeze(output)
        elif args.arch == 'localizer_alexnet_robust':
            avgPoolFunc = nn.AvgPool2d(kernel_size=outputModel.size()[-1])
            output = avgPoolFunc(outputModel)
            output = torch.squeeze(output)

        # TODO: Compute loss using ``criterion``
        loss = criterion(output, target*wgt)
        
        input = img_input

        # measure metrics and record loss
        m1 = metric1(output, target, wgt)
        m2 = metric2(output, target, wgt)
        losses.update(loss.item(), input.size(0))
        avg_m1.update(m1)
        avg_m2.update(m2)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Metric1 {avg_m1.val:.3f} ({avg_m1.avg:.3f})\t'
                  'Metric2 {avg_m2.val:.3f} ({avg_m2.avg:.3f})'.format(
                      i,
                      len(val_loader),
                      batch_time=batch_time,
                      loss=losses,
                      avg_m1=avg_m1,
                      avg_m2=avg_m2))

        #TODO: Visualize things as mentioned in handout
        #TODO: Visualize at appropriate intervals
        
        if USE_WANDB and epoch%2 == 0:
            wandb.log({'epoch': epoch,'test/loss': losses.avg})
            wandb.log({'epoch': epoch,'test/metric1': avg_m1.avg})
            wandb.log({'epoch': epoch,'test/metric2': avg_m2.avg})



    print(' * Metric1 {avg_m1.avg:.3f} Metric2 {avg_m2.avg:.3f}'.format(
        avg_m1=avg_m1, avg_m2=avg_m2))

    if epoch == 44:
        #choose 3 random images and plot heatmaps
        idxMat =[50, 500, 2500]

        for i in range(3):
            # idx = torch.randint(0,4000,(1,1))
            idx = idxMat[i]
            data = val_loader.dataset[idx]
            img_input = data['image'].to('cuda')
            target = data['label'].to('cuda')
            wgt = data['wgt'].to('cuda')

            # TODO: Get output from model
            outputModel = model(img_input[None,:,:,:])

            transformInputImage = transforms.Compose([transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224 , 1/0.225]), transforms.ToPILImage(), transforms.Resize((512,512))])

            posImgInput = img_input.cpu()
            inputImgPIL = transformInputImage(posImgInput)


            groundTruthIndices = torch.nonzero(target, as_tuple=True)

            wandb.log({"RandomOriginalImage-Index-"+str(i) : [wandb.Image(inputImgPIL,caption="RandomImageNumber-"+str(i)+"-Epoch-"+str(epoch)+"-Batch-"+str(i)+"-Image-0-"+"Original Image")]})

            if isinstance(groundTruthIndices,tuple) == 0:
                groundTruthIndices = (groundTruthIndices)

            for idx in groundTruthIndices[0]:
                imgHeatMap = createHeatMap(img_input, outputModel, target, idx)
                wandb.log({ "RandomImageNumber-"+str(i)+"-Epoch-"+str(epoch)+"-Batch-"+str(i)+"-Image-0-"+"Ground Truth Index-"+str(idx.item()) : [wandb.Image(imgHeatMap.permute(1,2,0).numpy(),caption="Image")]})


    return avg_m1.avg, avg_m2.avg

# TODO: You can make changes to this function if you wish (not necessary)
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


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


def metric1(output, target, wgt):
    # TODO: Ignore for now - proceed till instructed

    # precision (AP)
    # mAP score code from the previous homework

    nclasses = target.shape[1]
    AP = []
    SigFunc = nn.Sigmoid()
    # outputSig = SigFunc(output)
    wtarget = target
    # outputSig = output*wgt
    for cid in range(nclasses):
        gt_cls = wtarget[:, cid].cpu().numpy().astype('float32')
        pred_cls = output[:, cid].detach().cpu().numpy().astype('float32')
        pred_cls -= 1e-5 * gt_cls
        pred_cls = SigFunc(torch.from_numpy(pred_cls)).cpu().numpy().astype('float32')
        ap = sklearn.metrics.average_precision_score(gt_cls, pred_cls)
        if np.isnan(ap):
            continue
        else:
            AP.append(ap)
    
    mAP = np.mean(AP)

    # print(mAP)
    
    return mAP


def metric2(output, target, wgt):
    #TODO: Ignore for now - proceed till instructed

    # recall = truePositive / (truePositive + falseNegative)

    nclasses = target.shape[1]
    recallMat = []
    SigFunc = nn.Sigmoid()
    # outputSig = SigFunc(output)
    wtarget = target
    # outputSig = output*wgt
    for cid in range(nclasses):
        gt_cls = wtarget[:, cid].cpu().numpy().astype('float32')
        if np.amax(gt_cls)==0:
            continue
        pred_cls = output[:, cid].detach().cpu().numpy().astype('float32')
        pred_cls -= 1e-5 * gt_cls
        pred_cls = SigFunc(torch.from_numpy(pred_cls)).cpu().numpy().astype('float32')
        predOneHot = np.where(pred_cls >= 0.5, 1, 0)
        recall = sklearn.metrics.recall_score(gt_cls, predOneHot)
        recallMat.append(recall)
    
    mRecall = np.mean(recallMat)
    
    return mRecall


# helper function
def maxPoolOutput(x):
    output = torch.max(x,2)
    output = torch.max(output[0],2)
    output = torch.Tensor(output[0].detach().cpu()).cuda()
    
    return output

def createHeatMap(img, preMaxOutput, groundTruthChannel, groundTruthIndex):
    # extract ground truth channel
    # transform and upscale model
    # apply cmap to the image

    # define transforms for the image output
    transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((512,512))])
    transTensor = transforms.Compose([transforms.ToTensor()])
    transPIL = transforms.Compose([transforms.ToPILImage()])

    # choose the 0th image of the first ground truth channel from the first image in the batch
    chosenGTChannel = groundTruthIndex

    chosenOutput = preMaxOutput[0,chosenGTChannel,:,:]
    nonRGB_hImg = transform(chosenOutput.detach().cpu().numpy().astype("float32"))

    # choose not to blend with noise
    returnMap = transTensor(nonRGB_hImg)
    return returnMap


if __name__ == '__main__':
    main()
