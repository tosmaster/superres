from __future__ import print_function
import argparse
from math import log10

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dbpn import Net as DBPN
from dbpn_v1 import Net as DBPNLL
from dbpns import Net as DBPNS
from dbpn_iterative import Net as DBPNITER
from data import get_training_set,get_eval_set
import pdb
import socket
import time
import wandb
import glob
import numpy as np
from PIL import Image

from functools import reduce
#from scipy.misc import imsave
import scipy.io as sio
import time
import cv2

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=8, help="super resolution upscale factor")
parser.add_argument('--batchSize', type=int, default=1, help='training batch size')
parser.add_argument('--nEpochs', type=int, default=7, help='number of epochs to train for')
parser.add_argument('--snapshots', type=int, default=2, help='Snapshots')
parser.add_argument('--self_ensemble', type=bool, default=True)
parser.add_argument('--chop_forward', type=bool, default=False)
parser.add_argument('--start_iter', type=int, default=1, help='Starting Epoch')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=0.01')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=1, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
parser.add_argument('--data_dir', type=str, default='./data')
parser.add_argument('--data_augmentation', type=bool, default=True)
parser.add_argument('--hr_train_dataset', type=str, default='train')
parser.add_argument('--hr_test_dataset', type=str, default='test')
parser.add_argument('--model_type', type=str, default='DBPN-RES-MR64-3')
parser.add_argument('--residual', type=bool, default=True)
parser.add_argument('--patch_size', type=int, default=40, help='Size of cropped HR image')
parser.add_argument('--pretrained_sr', default='supperres.pth', help='sr pretrained base model')
parser.add_argument('--pretrained', type=bool, default=False)
parser.add_argument('--save_folder', default='weights/', help='Location to save checkpoint models')
parser.add_argument('--prefix', default='tpami_residual_filter8', help='Location to save checkpoint models')

opt = parser.parse_args()
gpus_list = range(opt.gpus)
hostname = str(socket.gethostname())
cudnn.benchmark = True
print(opt)

# After transforming of pytorch, the shape is [3,255,255]
def perceptual_distance(y_true, y_pred):
    """Calculate perceptual distance, DO NOT ALTER"""
    y_true *= 255
    y_pred *= 255
    rmean = (y_true[:, 0, :, :] + y_pred[:, 0, :, :]) / 2
    r = y_true[:, 0, :, :] - y_pred[:, 0, :, :]
    g = y_true[:, 1, :, :] - y_pred[:, 1, :, :]
    b = y_true[:, 2, :, :] - y_pred[:, 2, :, :]

    return torch.mean(torch.sqrt((((512+rmean)*r*r)/256) + 4*g*g + (((767-rmean)*b*b)/256)))

def train(epoch):
    epoch_loss = 0
    epoch_metric = 0
    model.train()
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target, bicubic = Variable(batch[0]), Variable(batch[1]), Variable(batch[2])
        if cuda:
            input = input.cuda(gpus_list[0])
            target = target.cuda(gpus_list[0])
            bicubic = bicubic.cuda(gpus_list[0])

        optimizer.zero_grad()
        t0 = time.time()
        prediction = model(input)

        if opt.residual:
            prediction = prediction + bicubic

        loss = criterion(prediction, target)
        epoch_loss += loss.data
        t1 = time.time()
        #epoch_metric += metric.data
        loss.backward()
        optimizer.step()
        #print(prediction.shape,target.shape)
        metric = perceptual_distance(prediction, target)
        epoch_metric += metric.data
        print("===> Epoch[{}]({}/{}): Loss: {:.4f} Perceptual_dist: {:.4f}|| Timer: {:.4f} sec.".format(epoch, iteration, len(training_data_loader), loss.data, metric.data, (t1 - t0)))
        #if iteration == 4: break

    print("===> Epoch {} Complete: Avg. Loss: {:.4f} perceptual_distance: {:.4f}".format(epoch, epoch_loss*opt.batchSize/len(training_data_loader), epoch_metric*opt.batchSize / len(training_data_loader)))
    wandb.log({"loss":epoch_loss*opt.batchSize / len(training_data_loader),"perceptual_distance":epoch_metric*opt.batchSize / len(training_data_loader)})

def eval():
    model.eval()
    for batch in testing_data_loader:
        with torch.no_grad():
            input, bicubic, name = Variable(batch[0]), Variable(batch[1]), batch[2]
        if cuda:
            input = input.cuda(gpus_list[0])
            bicubic = bicubic.cuda(gpus_list[0])

        t0 = time.time()
        if opt.chop_forward:
            with torch.no_grad():
                prediction = chop_forward(input, model, opt.upscale_factor)
        else:
            if opt.self_ensemble:
                with torch.no_grad():
                    prediction = x8_forward(input, model)
            else:
                with torch.no_grad():
                    prediction = model(input)

        if opt.residual:
            prediction = prediction + bicubic

        t1 = time.time()
        print("===> Processing: %s || Timer: %.4f sec." % (name[0], (t1 - t0)))
        save_img(prediction.cpu().data, name[0])

def save_img(img, img_name):
    save_img = img.squeeze().clamp(0, 1).numpy().transpose(1,2,0)
    # save img
    save_dir=os.path.join(opt.output,opt.test_dataset)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_fn = save_dir +'/'+ img_name
    cv2.imwrite(save_fn, cv2.cvtColor(save_img*255, cv2.COLOR_BGR2RGB),  [cv2.IMWRITE_PNG_COMPRESSION, 0])

def x8_forward(img, model, precision='single'):
    def _transform(v, op):
        if precision != 'single': v = v.float()

        v2np = v.data.cpu().numpy()
        if op == 'vflip':
            tfnp = v2np[:, :, :, ::-1].copy()
        elif op == 'hflip':
            tfnp = v2np[:, :, ::-1, :].copy()
        elif op == 'transpose':
            tfnp = v2np.transpose((0, 1, 3, 2)).copy()
        if cuda:
                ret = torch.Tensor(tfnp).cuda()
        else:
                ret = torch.Tensor(tfnp)

        if precision == 'half':
            ret = ret.half()
        elif precision == 'double':
            ret = ret.double()

        with torch.no_grad():
            ret = Variable(ret)

        return ret

    inputlist = [img]
    for tf in 'vflip', 'hflip', 'transpose':
        inputlist.extend([_transform(t, tf) for t in inputlist])

    outputlist = [model(aug) for aug in inputlist]
    for i in range(len(outputlist)):
        if i > 3:
            outputlist[i] = _transform(outputlist[i], 'transpose')
        if i % 4 > 1:
            outputlist[i] = _transform(outputlist[i], 'hflip')
        if (i % 4) % 2 == 1:
            outputlist[i] = _transform(outputlist[i], 'vflip')

    output = reduce((lambda x, y: x + y), outputlist) / len(outputlist)

    return output

def chop_forward(x, model, scale, shave=8, min_size=80000, nGPUs=opt.gpus):
    b, c, h, w = x.size()
    h_half, w_half = h // 2, w // 2
    h_size, w_size = h_half + shave, w_half + shave
    inputlist = [
        x[:, :, 0:h_size, 0:w_size],
        x[:, :, 0:h_size, (w - w_size):w],
        x[:, :, (h - h_size):h, 0:w_size],
        x[:, :, (h - h_size):h, (w - w_size):w]]

    if w_size * h_size < min_size:
        outputlist = []
        for i in range(0, 4, nGPUs):
            with torch.no_grad():
                input_batch = torch.cat(inputlist[i:(i + nGPUs)], dim=0)
            if opt.self_ensemble:
                with torch.no_grad():
                    output_batch = x8_forward(input_batch, model)
            else:
                with torch.no_grad():
                    output_batch = model(input_batch)
            outputlist.extend(output_batch.chunk(nGPUs, dim=0))
    else:
        outputlist = [
            chop_forward(patch, model, scale, shave, min_size, nGPUs) \
            for patch in inputlist]

    h, w = scale * h, scale * w
    h_half, w_half = scale * h_half, scale * w_half
    h_size, w_size = scale * h_size, scale * w_size
    shave *= scale

    with torch.no_grad():
        output = Variable(x.data.new(b, c, h, w))

    output[:, :, 0:h_half, 0:w_half] \
        = outputlist[0][:, :, 0:h_half, 0:w_half]
    output[:, :, 0:h_half, w_half:w] \
        = outputlist[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
    output[:, :, h_half:h, 0:w_half] \
        = outputlist[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
    output[:, :, h_half:h, w_half:w] \
        = outputlist[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

    return output

def test():
    model.eval()
    val_loss = 0
    epoch_metric = 0
    sample = {}
    for i,batch in enumerate(testing_data_loader,1):
        input, target,bicubicle = Variable(batch[0]), Variable(batch[1]),Variable(batch[2])
        if cuda:
            input = input.cuda(gpus_list[0])
            target = target.cuda(gpus_list[0])
            bicubic = bicubic.cuda(gpus_list[0])

        t0 = time.time()
        if opt.chop_forward:
            with torch.no_grad():
                prediction = chop_forward(input, model, opt.upscale_factor)
        else:
            if opt.self_ensemble:
                with torch.no_grad():
                    prediction = x8_forward(input, model)
            else:
                with torch.no_grad():
                    prediction = model(input)

        if opt.residual:
            prediction = prediction + bicubicle

        t1 = time.time()

        #prediction = model(input)
        #prediction += bicubicle
        loss = criterion(prediction, target)
        perceptual = perceptual_distance(prediction, target)
        #psnr = 10 * log10(1 / mse.data[0])
        epoch_metric += perceptual.data
        val_loss += loss.data
        print(i,"perceptual",perceptual.data)
        sample["input"] = input
        sample["prediction"] = prediction
        sample["target"] = target
        #if i == 4: break
    print("===> val_loss: {:.4f} ".format(val_loss * opt.batchSize / len(testing_data_loader)))
    input = sample["input"].permute(0,2,3,1).detach().clamp(0, 1).numpy()*255
    pred = sample["prediction"].permute(0,2,3,1).detach().numpy()
    target = sample["target"].permute(0,2,3,1).detach().numpy()
    #print(input.shape,pred.shape,target.shape)
    #orig = Image.fromarray(input[0],'RGB')
    #orig.save()
    wandb.log({"examples":[ wandb.Image(np.concatenate([input[i].repeat(8, axis=0).repeat(8, axis=1), pred[i] * 255, target[i] * 255], axis=1)) for i in range(opt.batchSize)]},commit=False)
    wandb.log({"val_loss":val_loss*opt.batchSize / len(testing_data_loader),"val_perceptual_distance":epoch_metric*opt.batchSize / len(testing_data_loader)})

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

def checkpoint(epoch):
    model_out_path = opt.save_folder+opt.hr_train_dataset+hostname+opt.model_type+opt.prefix+"_epoch_{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

def loadmodel(filename):
    model_out_path = opt.save_folder + filename
    model.load_state_dict(torch.load(model_out_path, map_location=lambda storage, loc: storage))
    print("load model from {}".format(model_out_path))

cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

# Init wandb
run = wandb.init(project='superres')

config = run.config

config.num_epochs = opt.nEpochs
config.batch_size = opt.batchSize
config.input_height = 32
config.input_width = 32
config.output_height = 256
config.output_width = 256

config.steps_per_epoch = len(
    glob.glob(opt.data_dir + '/' + opt.hr_train_dataset + "/*-in.jpg")) // config.batch_size
config.val_steps_per_epoch = len(
    glob.glob(opt.data_dir + '/' + opt.hr_test_dataset + "/*-in.jpg")) // config.batch_size

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
train_set = get_training_set(opt.data_dir, opt.hr_train_dataset, opt.upscale_factor, opt.patch_size, opt.data_augmentation)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)

test_set = get_eval_set(opt.data_dir +'/'+opt.hr_test_dataset, opt.upscale_factor)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=False)

print('===> Building model ', opt.model_type)
if opt.model_type == 'DBPNLL':
    model = DBPNLL(num_channels=3, base_filter=64,  feat = 256, num_stages=10, scale_factor=opt.upscale_factor)
elif opt.model_type == 'DBPN-RES-MR64-3':
    print("Modle DBPN-RES-MR64-3 is used.")
    model = DBPNITER(num_channels=3, base_filter=64,  feat = 256, num_stages=3, scale_factor=opt.upscale_factor)
else:
    model = DBPN(num_channels=3, base_filter=64,  feat = 256, num_stages=7, scale_factor=opt.upscale_factor)

model = torch.nn.DataParallel(model, device_ids=gpus_list)

wandb.watch(model)

criterion = nn.L1Loss()

print('---------- Networks architecture -------------')
print_network(model)
print('----------------------------------------------')

if opt.pretrained:
    model_name = os.path.join(opt.save_folder + opt.pretrained_sr)
    if os.path.exists(model_name):
        #model= torch.load(model_name, map_location=lambda storage, loc: storage)
        model.load_state_dict(torch.load(model_name, map_location=lambda storage, loc: storage))
        print('Pre-trained SR model is loaded.')
if cuda:
    model = model.cuda(gpus_list[0])
    criterion = criterion.cuda(gpus_list[0])

optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)

#ploadmodel("RES_1.pth")
#loadmodel("RES_2.pth")
#loadmodel("epoch_3.pth")
for epoch in range(opt.start_iter, opt.nEpochs + 1):
    #break
    train(epoch)
    # learning rate is decayed by a factor of 10 every half of total epochs
    if (epoch+1) % (opt.nEpochs/2) == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 10.0
        print('Learning rate decay: lr={}'.format(optimizer.param_groups[0]['lr']))

    if (epoch+1) % (opt.snapshots) == 0:
        checkpoint(epoch)
        test()
print("test starting...")

#loadmodel("RES_2.pth")
test()

print('===> Building model ', opt.model_type)
if opt.model_type == 'DBPNLL':
    model = DBPNLL(num_channels=3, base_filter=64,  feat = 256, num_stages=10, scale_factor=opt.upscale_factor) 
elif opt.model_type == 'DBPN-RES-MR64-3':
    model = DBPNITER(num_channels=3, base_filter=64,  feat = 256, num_stages=3, scale_factor=opt.upscale_factor)
else:
    model = DBPN(num_channels=3, base_filter=64,  feat = 256, num_stages=7, scale_factor=opt.upscale_factor) 
    
model = torch.nn.DataParallel(model, device_ids=gpus_list)
criterion = nn.L1Loss()

print('---------- Networks architecture -------------')
print_network(model)
print('----------------------------------------------')

if opt.pretrained:
    model_name = os.path.join(opt.save_folder + opt.pretrained_sr)
    if os.path.exists(model_name):
        #model= torch.load(model_name, map_location=lambda storage, loc: storage)
        model.load_state_dict(torch.load(model_name, map_location=lambda storage, loc: storage))
        print('Pre-trained SR model is loaded.')

if cuda:
    model = model.cuda(gpus_list[0])
    criterion = criterion.cuda(gpus_list[0])

optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)

for epoch in range(opt.start_iter, opt.nEpochs + 1):
    train(epoch)
    #break
    # learning rate is decayed by a factor of 10 every half of total epochs
    if (epoch+1) % (opt.nEpochs/2) == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 10.0
        print('Learning rate decay: lr={}'.format(optimizer.param_groups[0]['lr']))
            
    if (epoch+1) % (opt.snapshots) == 0:
        checkpoint(epoch)

#test()
