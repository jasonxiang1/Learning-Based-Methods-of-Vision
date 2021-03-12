# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# Modified by Sudeep Dasari
# --------------------------------------------------------
from __future__ import print_function

import torch
import numpy as np

import utils
from voc_dataset import VOCDataset

import torchvision
import torch.nn as nn
import os
from torch.utils.tensorboard import SummaryWriter


def save_this_epoch(args, epoch):
        # TODO: Q2 check if model should be saved this epoch
        if (epoch % args.save_freq == 0):
            return True
 
        return False

        # raise NotImplementedError


def save_model(epoch, model_name, model):
    # TODO: Q2 Implement code for model saving

    torch.save({
        'model name: ': model_name,
        'epoch': epoch,
        'model_state_dict':model.state_dict()},
        os.path.join('saved_models/',model_name + '-' + str(epoch)+'.pth'))
    # raise NotImplementedError

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train(args, model, optimizer, scheduler=None, model_name='model'):
    # TODO: Q1.5 Initialize your visualizer here!
    writer = SummaryWriter("runs/CaffeNet_Q2_Conv1Filters")
    # TODO: Q1.2 complete your dataloader in voc_dataset.py
    train_loader = utils.get_data_loader('voc', train=True, batch_size=args.batch_size, split='trainval')
    test_loader = utils.get_data_loader('voc', train=False, batch_size=args.test_batch_size, split='test')

    # Ensure model is in correct mode and on right device
    model.train()
    model = model.to(args.device)

    # TODO: Q1.4 Implement model training code!
    cnt = 0
    prev_epoch = 0

    # for filter output
    prev_epochArr = []
    epoch_filter_list = [5, 10, 15, 30, 45]
    for epoch in range(args.epochs):
        for batch_idx, (findex, data, target, wgt) in enumerate(train_loader):
            # Get a batch of data
            data, target, wgt = data.to(args.device), target.to(args.device), wgt.to(args.device)
            optimizer.zero_grad()
            # Forward pass
            output = model(data)
            # Calculate the loss
            # TODO: your loss for multi-label clf?
            # loss = nn.functional.binary_cross_entropy_with_logits(output,target*wgt)# .to(args.device)
            m = nn.Sigmoid()
            n = nn.BCELoss()
            loss = n(m(output),target*wgt).to(args.device)
            # Calculate gradient w.r.t the loss
            loss.backward()
            # Optimizer takes one step
            optimizer.step()
            # add image ouputted from conv1 filter
            if (epoch in epoch_filter_list and epoch not in prev_epochArr):
                conv1_grid = torchvision.utils.make_grid(model.conv1.weight)
                writer.add_image('Conv1 Filters at Epoch: ' + str(epoch),conv1_grid)
                prev_epochArr.append(epoch)
            # Log info
            if cnt % args.log_every == 0:
                # todo: add your visualization code
                # values to record: loss, map, learning rate, histogram of gradients of conv1
                model.eval()
                trainap, trainmap = utils.eval_dataset_map(model, args.device, test_loader)
                model.train()
                writer.add_scalar('Loss', loss, epoch)
                writer.add_scalar('mAP',trainmap,epoch)
                writer.add_scalar('Learning Rate',get_lr(optimizer),epoch)

                # writer.add_histogram('conv1.weight.grad',model.conv1.weight.grad,epoch)
                print('Train Epoch: {} [{} ({:.0f}%)]\tLoss: {:.6f} | mAP: {:.6f}'.format(
                    epoch, cnt, 100. * batch_idx / len(train_loader), loss.item(), trainmap))
            # Validation iteration
            if cnt % args.val_every == 0:
                model.eval()
                ap, map = utils.eval_dataset_map(model, args.device, test_loader)
                model.train()
            if (prev_epoch != epoch):
                if(save_this_epoch(args,epoch)):
                    save_model(epoch,model.__class__.__name__,model)
                    epoch += 1
            cnt += 1
        if scheduler is not None:
            scheduler.step()
    # check to save model at end
    if(args.save_at_end):
        save_model(epoch,model.__class__.__name__,model)
    writer.close()

    # Validation iteration
    test_loader = utils.get_data_loader('voc', train=False, batch_size=args.test_batch_size, split='test')
    ap, map = utils.eval_dataset_map(model, args.device, test_loader)
    return ap, map

# implemented to avoid freeze_support() error
if __name__ == '__main__':
    train(args, model, optimizer, scheduler=None, model_name='model')