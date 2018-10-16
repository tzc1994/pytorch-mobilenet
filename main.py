import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

#from torch.nn.parameter import Parameter

import torch.nn.functional as F

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

model_names.append('mobilenet')

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
best_prec1 = 0
"""
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
    
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(  3,  32, 2), 
            conv_dw( 32,  64, 1),
            conv_dw( 64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(1024, 1000)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x
"""
 

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3,32,kernel_size=3, stride=2, padding=1, groups=1, bias=False)
        self.conv2 = nn.Conv2d(32,32,kernel_size=3, stride=1, padding=1, groups=32, bias=False)
        self.conv3 = nn.Conv2d(32,32,kernel_size=1, stride=1, groups=1, bias=False)
        self.conv4 = nn.Conv2d(64,64,kernel_size=3, stride=2, padding=1, groups=64, bias=False)
        self.conv5 = nn.Conv2d(64,64,kernel_size=1, stride=1, groups=1, bias=False)
        self.conv6 = nn.Conv2d(128,128,kernel_size=3, stride=1, padding=1, groups=128, bias=False)
        self.conv7 = nn.Conv2d(128,64,kernel_size=1, stride=1, groups=1, bias=False)
        self.conv8 = nn.Conv2d(128,128,kernel_size=3, stride=2, padding=1, groups=128, bias=False)
        self.conv9 = nn.Conv2d(128,128,kernel_size=1, stride=1, groups=1, bias=False)
        self.conv10 = nn.Conv2d(256,256,kernel_size=3, stride=1, padding=1, groups=256, bias=False)
        self.conv11 = nn.Conv2d(256,128,kernel_size=1, stride=1, groups=1, bias=False)
        self.conv12 = nn.Conv2d(256,256,kernel_size=3, stride=2, padding=1, groups=256, bias=False)
        self.conv13 = nn.Conv2d(256,256,kernel_size=1, stride=1, groups=1, bias=False)
        self.conv14 = nn.Conv2d(512,512,kernel_size=3, stride=1, padding=1, groups=512, bias=False)
        self.conv15 = nn.Conv2d(512,256,kernel_size=1, stride=1, groups=1, bias=False)
        self.conv16 = nn.Conv2d(512,512,kernel_size=3, stride=1, padding=1, groups=512, bias=False)
        self.conv17 = nn.Conv2d(512,256,kernel_size=1, stride=1, groups=1, bias=False)
        self.conv18 = nn.Conv2d(512,512,kernel_size=3, stride=1, padding=1, groups=512, bias=False)
        self.conv19 = nn.Conv2d(512,256,kernel_size=1, stride=1, groups=1, bias=False)
        self.conv20 = nn.Conv2d(512,512,kernel_size=3, stride=1, padding=1, groups=512, bias=False)
        self.conv21 = nn.Conv2d(512,256,kernel_size=1, stride=1, groups=1, bias=False)
        self.conv22 = nn.Conv2d(512,512,kernel_size=3, stride=1, padding=1, groups=512, bias=False)
        self.conv23 = nn.Conv2d(512,256,kernel_size=1, stride=1, groups=1, bias=False)
        self.conv24 = nn.Conv2d(512,512,kernel_size=3, stride=2, padding=1, groups=512, bias=False)
        self.conv25 = nn.Conv2d(512,512,kernel_size=1, stride=1, groups=1, bias=False)
        self.conv26 = nn.Conv2d(1024,1024,kernel_size=3, stride=1, padding=1, groups=1024, bias=False)
        self.conv27 = nn.Conv2d(1024,512,kernel_size=1, stride=1, groups=1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm2d(128)
        self.batch_norm4 = nn.BatchNorm2d(256)
        self.batch_norm5 = nn.BatchNorm2d(512)
        self.batch_norm6 = nn.BatchNorm2d(1024)
        self.fc = nn.Linear(1024, 1000)
    def forward(self, x):
        x = F.relu(self.batch_norm1(self.conv1(x)),inplace=True) #112 * 112 * 32
        #############################################################################
        x = F.relu(self.batch_norm1(self.conv2(x)),inplace=True) #112 * 112 * 32 depthwise
        x1_1 = self.batch_norm1(self.conv3(x))
        weight1 = torch.randn((32,16,1,1),requires_grad=True).cuda()
        weight1_flip = torch.flip(weight1,[1])
        weight1 = torch.cat((weight1,weight1_flip),1)
        x1_2 = self.batch_norm1(F.conv2d(x,weight1,bias=None,padding=0,groups=1))
        x = torch.cat((x1_1,x1_2),1)
        x = F.relu(x,inplace=True) # 112 * 112 * 64
        ##############################################################################
        x = F.relu(self.batch_norm2(self.conv4(x)),inplace=True) #56 * 56 * 64 depthwise
        x2_1 = self.batch_norm2(self.conv5(x))
        weight2 = torch.randn((64,32,1,1),requires_grad=True).cuda()
        weight2_flip = torch.flip(weight2,[1])
        weight2 = torch.cat((weight2,weight2_flip),1)
        x2_2 = self.batch_norm2(F.conv2d(x,weight2,bias=None,padding=0,groups=1))
        x = torch.cat((x2_1,x2_2),1)
        x = F.relu(x,inplace=True) # 56 * 56 * 128
        ###############################################################################
        x = F.relu(self.batch_norm3(self.conv6(x)),inplace=True) #56 * 56 * 128 depthwise
        x3_1 = self.batch_norm2(self.conv7(x))
        weight3 = torch.randn((64,64,1,1),requires_grad=True).cuda()
        weight3_flip = torch.flip(weight3,[1])
        weight3 = torch.cat((weight3,weight3_flip),1)
        x3_2 = self.batch_norm2(F.conv2d(x,weight3,bias=None,padding=0,groups=1))
        x = torch.cat((x3_1,x3_2),1)
        x = F.relu(x,inplace=True) # 56 * 56 * 128
        ################################################################################
        x = F.relu(self.batch_norm3(self.conv8(x)),inplace=True) #28 * 28 * 128 depthwise
        x4_1 = self.batch_norm3(self.conv9(x))
        weight4 = torch.randn((128,64,1,1),requires_grad=True).cuda()
        weight4_flip = torch.flip(weight4,[1])
        weight4 = torch.cat((weight4,weight4_flip),1)
        x4_2 = self.batch_norm3(F.conv2d(x,weight4,bias=None,padding=0,groups=1))
        x = torch.cat((x4_1,x4_2),1)
        x = F.relu(x,inplace=True) # 28 * 28 * 256
        ################################################################################
        x = F.relu(self.batch_norm4(self.conv10(x)),inplace=True) #28 * 28* 256 depthwise
        x5_1 = self.batch_norm3(self.conv11(x))
        weight5 = torch.randn((128,128,1,1),requires_grad=True).cuda()
        weight5_flip = torch.flip(weight5,[1])
        weight5 = torch.cat((weight5,weight5_flip),1)
        x5_2 = self.batch_norm3(F.conv2d(x,weight5,bias=None,padding=0,groups=1))
        x = torch.cat((x5_1,x5_2),1)
        x = F.relu(x,inplace=True) # 28 * 28 * 256
        ################################################################################
        x = F.relu(self.batch_norm4(self.conv12(x)),inplace=True) #14 * 14 * 256 depthwise
        x6_1 = self.batch_norm4(self.conv13(x))
        weight6 = torch.randn((256,128,1,1),requires_grad=True).cuda()
        weight6_flip = torch.flip(weight6,[1])
        weight6 = torch.cat((weight6,weight6_flip),1)
        x6_2 = self.batch_norm4(F.conv2d(x,weight6,bias=None,padding=0,groups=1))
        x = torch.cat((x6_1,x6_2),1)
        x = F.relu(x,inplace=True) # 14 * 14 * 512
        ##################################################################################
        ##################################################################################
        x = F.relu(self.batch_norm5(self.conv14(x)),inplace=True) #14 * 14 * 512 depthwise
        x7_1 = self.batch_norm4(self.conv15(x))
        weight7 = torch.randn((256,256,1,1),requires_grad=True).cuda()
        weight7_flip = torch.flip(weight7,[1])
        weight7 = torch.cat((weight7,weight7_flip),1)
        x7_2 = self.batch_norm4(F.conv2d(x,weight7,bias=None,padding=0,groups=1))
        x = torch.cat((x7_1,x7_2),1)
        x = F.relu(x,inplace=True) # 14 * 14 * 512
        ###################################################################################
        x = F.relu(self.batch_norm5(self.conv16(x)),inplace=True) #14 * 14 * 512 depthwise
        x8_1 = self.batch_norm4(self.conv17(x))
        weight8 = torch.randn((256,256,1,1),requires_grad=True).cuda()
        weight8_flip = torch.flip(weight8,[1])
        weight8 = torch.cat((weight8,weight8_flip),1)
        x8_2 = self.batch_norm4(F.conv2d(x,weight8,bias=None,padding=0,groups=1))
        x = torch.cat((x8_1,x8_2),1)
        x = F.relu(x,inplace=True) # 14 * 14 * 512
        ####################################################################################
        x = F.relu(self.batch_norm5(self.conv18(x)),inplace=True) #14 * 14 * 512 depthwise
        x9_1 = self.batch_norm4(self.conv19(x))
        weight9 = torch.randn((256,256,1,1),requires_grad=True).cuda()
        weight9_flip = torch.flip(weight9,[1])
        weight9 = torch.cat((weight9,weight9_flip),1)
        x9_2 = self.batch_norm4(F.conv2d(x,weight9,bias=None,padding=0,groups=1))
        x = torch.cat((x9_1,x9_2),1)
        x = F.relu(x,inplace=True) # 14 * 14 * 512
        #####################################################################################
        x = F.relu(self.batch_norm5(self.conv20(x)),inplace=True) #14 * 14 * 512 depthwise
        x10_1 = self.batch_norm4(self.conv21(x))
        weight10 = torch.randn((256,256,1,1),requires_grad=True).cuda()
        weight10_flip = torch.flip(weight10,[1])
        weight10 = torch.cat((weight10,weight10_flip),1)
        x10_2 = self.batch_norm4(F.conv2d(x,weight10,bias=None,padding=0,groups=1))
        x = torch.cat((x10_1,x10_2),1)
        x = F.relu(x,inplace=True) # 14 * 14 * 512
        ######################################################################################
        x = F.relu(self.batch_norm5(self.conv22(x)),inplace=True) #514 * 14 * 512 depthwise
        x11_1 = self.batch_norm4(self.conv23(x))
        weight11 = torch.randn((256,256,1,1),requires_grad=True).cuda()
        weight11_flip = torch.flip(weight11,[1])
        weight11 = torch.cat((weight11,weight11_flip),1)
        x11_2 = self.batch_norm4(F.conv2d(x,weight11,bias=None,padding=0,groups=1))
        x = torch.cat((x11_1,x11_2),1)
        x = F.relu(x,inplace=True) # 14 * 14 * 512
        #######################################################################################
        #######################################################################################
        x = F.relu(self.batch_norm5(self.conv24(x)),inplace=True) #7 * 7 * 512 depthwise
        x12_1 = self.batch_norm5(self.conv25(x))
        weight12 = torch.randn((512,256,1,1),requires_grad=True).cuda()
        weight12_flip = torch.flip(weight12,[1])
        weight12 = torch.cat((weight12,weight12_flip),1)
        x12_2 = self.batch_norm5(F.conv2d(x,weight12,bias=None,padding=0,groups=1))
        x = torch.cat((x12_1,x12_2),1)
        x = F.relu(x,inplace=True) # 7 * 7 * 1024
        #######################################################################################
        x = F.relu(self.batch_norm6(self.conv26(x)),inplace=True) #7 * 7 * 1024 depthwise
        x13_1 = self.batch_norm5(self.conv27(x))
        weight13 = torch.randn((512,512,1,1),requires_grad=True).cuda()
        weight13_flip = torch.flip(weight13,[1])
        weight13 = torch.cat((weight13,weight13_flip),1)
        x13_2 = self.batch_norm5(F.conv2d(x,weight13,bias=None,padding=0,groups=1))
        x = torch.cat((x13_1,x13_2),1)
        x = F.relu(x,inplace=True) # 7 * 7 * 1024
        #######################################################################################
        x = F.avg_pool2d(x,7)
        #######################################################################################
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x
"""  
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3,32,kernel_size=3, stride=2, padding=1, groups=1, bias=False)
        self.conv2 = nn.Conv2d(32,32,kernel_size=3, stride=1, padding=1, groups=32, bias=False)
        self.conv3 = nn.Conv2d(32,32,kernel_size=1, stride=1, groups=1, bias=False)
        self.conv4 = nn.Conv2d(64,64,kernel_size=3, stride=2, padding=1, groups=64, bias=False)
        self.conv5 = nn.Conv2d(64,64,kernel_size=1, stride=1, groups=1, bias=False)
        self.conv6 = nn.Conv2d(128,128,kernel_size=3, stride=1, padding=1, groups=128, bias=False)
        self.conv7 = nn.Conv2d(128,64,kernel_size=1, stride=1, groups=1, bias=False)
        self.conv8 = nn.Conv2d(128,128,kernel_size=3, stride=2, padding=1, groups=128, bias=False)
        self.conv9 = nn.Conv2d(128,128,kernel_size=1, stride=1, groups=1, bias=False)
        self.conv10 = nn.Conv2d(256,256,kernel_size=3, stride=1, padding=1, groups=256, bias=False)
        self.conv11 = nn.Conv2d(256,128,kernel_size=1, stride=1, groups=1, bias=False)
        self.conv12 = nn.Conv2d(256,256,kernel_size=3, stride=2, padding=1, groups=256, bias=False)
        self.conv13 = nn.Conv2d(256,256,kernel_size=1, stride=1, groups=1, bias=False)
        self.conv14 = nn.Conv2d(512,512,kernel_size=3, stride=1, padding=1, groups=512, bias=False)
        self.conv15 = nn.Conv2d(512,256,kernel_size=1, stride=1, groups=1, bias=False)
        self.conv16 = nn.Conv2d(512,512,kernel_size=3, stride=1, padding=1, groups=512, bias=False)
        self.conv17 = nn.Conv2d(512,256,kernel_size=1, stride=1, groups=1, bias=False)
        self.conv18 = nn.Conv2d(512,512,kernel_size=3, stride=1, padding=1, groups=512, bias=False)
        self.conv19 = nn.Conv2d(512,256,kernel_size=1, stride=1, groups=1, bias=False)
        self.conv20 = nn.Conv2d(512,512,kernel_size=3, stride=1, padding=1, groups=512, bias=False)
        self.conv21 = nn.Conv2d(512,256,kernel_size=1, stride=1, groups=1, bias=False)
        self.conv22 = nn.Conv2d(512,512,kernel_size=3, stride=1, padding=1, groups=512, bias=False)
        self.conv23 = nn.Conv2d(512,256,kernel_size=1, stride=1, groups=1, bias=False)
        self.conv24 = nn.Conv2d(512,512,kernel_size=3, stride=2, padding=1, groups=512, bias=False)
        self.conv25 = nn.Conv2d(512,512,kernel_size=1, stride=1, groups=1, bias=False)
        self.conv26 = nn.Conv2d(1024,1024,kernel_size=3, stride=1, padding=1, groups=1024, bias=False)
        self.conv27 = nn.Conv2d(1024,512,kernel_size=1, stride=1, groups=1, bias=False)
        self.fc = nn.Linear(1024, 1000)
    def forward(self, x):
        x = F.relu(nn.BatchNorm2d(self.conv1(x),running_mean=torch.randn(32).cuda(),running_var=torch.ones(32).cuda(),
            weight = Parameter(torch.Tensor(32)).cuda(),bias = Parameter(torch.Tensor(32)).cuda(),training=True, momentum=0.1, eps=1e-05),inplace=True) #112 * 112 * 32
        #############################################################################
        x = F.relu(F.batch_norm(self.conv2(x),running_mean=torch.randn(32).cuda(),running_var=torch.ones(32).cuda(),
            weight = Parameter(torch.Tensor(32)).cuda(),bias = Parameter(torch.Tensor(32)).cuda(),training=True, momentum=0.1, eps=1e-05),inplace=True) #112 * 112 * 32 depthwise
        x1_1 = F.batch_norm(self.conv3(x),running_mean=torch.randn(32).cuda(),running_var=torch.ones(32).cuda(),
            weight = Parameter(torch.Tensor(32)).cuda(),bias = Parameter(torch.Tensor(32)).cuda(),training=True, momentum=0.1, eps=1e-05)
        weight1 = torch.nn.init(torch.empty(32,16,1,1),a = -1, b = 1 ).cuda()
        weight1.requires_grad = True
        weight1_flip = torch.flip(weight1,[1])
        weight1 = torch.cat((weight1,weight1_flip),1)
        x1_2 = F.batch_norm(F.conv2d(x,weight1,bias=None,padding=0,groups=1),running_mean=torch.randn(32).cuda(),running_var=torch.ones(32).cuda(),
            weight = Parameter(torch.Tensor(32)).cuda(),bias = Parameter(torch.Tensor(32)).cuda(),training=True, momentum=0.1, eps=1e-05)
        x = torch.cat((x1_1,x1_2),1)
        x = F.relu(x,inplace=True) # 112 * 112 * 64
        ##############################################################################
        x = F.relu(F.batch_norm(self.conv4(x),running_mean=torch.randn(64).cuda(),running_var=torch.ones(64).cuda(),
            weight = Parameter(torch.Tensor(64)).cuda(),bias = Parameter(torch.Tensor(64)).cuda(),training=True, momentum=0.1, eps=1e-05),inplace=True) #56 * 56 * 64 depthwise
        x2_1 = F.batch_norm(self.conv5(x),running_mean=torch.randn(64).cuda(),running_var=torch.ones(64).cuda(),
            weight = Parameter(torch.Tensor(64)).cuda(),bias = Parameter(torch.Tensor(64)).cuda(),training=True, momentum=0.1, eps=1e-05)
        weight2 = torch.randn((64,32,1,1),requires_grad=True).cuda()
        weight2_flip = torch.flip(weight2,[1])
        weight2 = torch.cat((weight2,weight2_flip),1)
        x2_2 = F.batch_norm(F.conv2d(x,weight2,bias=None,padding=0,groups=1),running_mean=torch.randn(64).cuda(),running_var=torch.ones(64).cuda(),
            weight = Parameter(torch.Tensor(64)).cuda(),bias = Parameter(torch.Tensor(64)).cuda(),training=True, momentum=0.1, eps=1e-05)
        x = torch.cat((x2_1,x2_2),1)
        x = F.relu(x,inplace=True) # 56 * 56 * 128
        ###############################################################################
        x = F.relu(F.batch_norm(self.conv6(x),running_mean=torch.randn(128).cuda(),running_var=torch.ones(128).cuda(),
            weight = Parameter(torch.Tensor(128)).cuda(),bias = Parameter(torch.Tensor(128)).cuda(),training=True, momentum=0.1, eps=1e-05),inplace=True) #56 * 56 * 64 depthwise
        x3_1 = F.batch_norm(self.conv7(x),running_mean=torch.randn(64).cuda(),running_var=torch.ones(64).cuda(),
            weight = Parameter(torch.Tensor(64)).cuda(),bias = Parameter(torch.Tensor(64)).cuda(),training=True, momentum=0.1, eps=1e-05)
        weight3 = torch.randn((64,64,1,1),requires_grad=True).cuda()
        weight3_flip = torch.flip(weight3,[1])
        weight3 = torch.cat((weight3,weight3_flip),1)
        x3_2 = F.batch_norm(F.conv2d(x,weight3,bias=None,padding=0,groups=1),running_mean=torch.randn(64).cuda(),running_var=torch.ones(64).cuda(),
            weight = Parameter(torch.Tensor(64)).cuda(),bias = Parameter(torch.Tensor(64)).cuda(),training=True, momentum=0.1, eps=1e-05)
        x = torch.cat((x3_1,x3_2),1)
        x = F.relu(x,inplace=True) # 56 * 56 * 128
        ################################################################################
        x = F.relu(F.batch_norm(self.conv8(x),running_mean=torch.randn(128).cuda(),running_var=torch.ones(128).cuda(),
            weight = Parameter(torch.Tensor(128)).cuda(),bias = Parameter(torch.Tensor(128)).cuda(),training=True, momentum=0.1, eps=1e-05),inplace=True) #56 * 56 * 64 depthwise
        x4_1 = F.batch_norm(self.conv9(x),running_mean=torch.randn(128).cuda(),running_var=torch.ones(128).cuda(),
            weight = Parameter(torch.Tensor(128)).cuda(),bias = Parameter(torch.Tensor(128)).cuda(),training=True, momentum=0.1, eps=1e-05)
        weight4 = torch.randn((128,64,1,1),requires_grad=True).cuda()
        weight4_flip = torch.flip(weight4,[1])
        weight4 = torch.cat((weight4,weight4_flip),1)
        x4_2 = F.batch_norm(F.conv2d(x,weight4,bias=None,padding=0,groups=1),running_mean=torch.randn(128).cuda(),running_var=torch.ones(128).cuda(),
            weight = Parameter(torch.Tensor(128)).cuda(),bias = Parameter(torch.Tensor(128)).cuda(),training=True, momentum=0.1, eps=1e-05)
        x = torch.cat((x4_1,x4_2),1)
        x = F.relu(x,inplace=True) # 28 * 28 * 256
        ################################################################################
        x = F.relu(F.batch_norm(self.conv10(x),running_mean=torch.randn(256).cuda(),running_var=torch.ones(256).cuda(),
            weight = Parameter(torch.Tensor(256)).cuda(),bias = Parameter(torch.Tensor(256)).cuda(),training=True, momentum=0.1, eps=1e-05),inplace=True) #56 * 56 * 64 depthwise
        x5_1 = F.batch_norm(self.conv11(x),running_mean=torch.randn(128).cuda(),running_var=torch.ones(128).cuda(),
            weight = Parameter(torch.Tensor(128)).cuda(),bias = Parameter(torch.Tensor(128)).cuda(),training=True, momentum=0.1, eps=1e-05)
        weight5 = torch.randn((128,128,1,1),requires_grad=True).cuda()
        weight5_flip = torch.flip(weight5,[1])
        weight5 = torch.cat((weight5,weight5_flip),1)
        x5_2 = F.batch_norm(F.conv2d(x,weight5,bias=None,padding=0,groups=1),running_mean=torch.randn(128).cuda(),running_var=torch.ones(128).cuda(),
            weight = Parameter(torch.Tensor(128)).cuda(),bias = Parameter(torch.Tensor(128)).cuda(),training=True, momentum=0.1, eps=1e-05)
        x = torch.cat((x5_1,x5_2),1)
        x = F.relu(x,inplace=True) # 28 * 28 * 256
        ################################################################################
        x = F.relu(F.batch_norm(self.conv12(x),running_mean=torch.randn(256).cuda(),running_var=torch.ones(256).cuda(),
            weight = Parameter(torch.Tensor(256)).cuda(),bias = Parameter(torch.Tensor(256)).cuda(),training=True, momentum=0.1, eps=1e-05),inplace=True) #56 * 56 * 64 depthwise
        x6_1 = F.batch_norm(self.conv13(x),running_mean=torch.randn(256).cuda(),running_var=torch.ones(256).cuda(),
            weight = Parameter(torch.Tensor(256)).cuda(),bias = Parameter(torch.Tensor(256)).cuda(),training=True, momentum=0.1, eps=1e-05)
        weight6 = torch.randn((256,128,1,1),requires_grad=True).cuda()
        weight6_flip = torch.flip(weight6,[1])
        weight6 = torch.cat((weight6,weight6_flip),1)
        x6_2 = F.batch_norm(F.conv2d(x,weight6,bias=None,padding=0,groups=1),running_mean=torch.randn(256).cuda(),running_var=torch.ones(256).cuda(),
            weight = Parameter(torch.Tensor(256)).cuda(),bias = Parameter(torch.Tensor(256)).cuda(),training=True, momentum=0.1, eps=1e-05)
        x = torch.cat((x6_1,x6_2),1)
        x = F.relu(x,inplace=True) # 14 * 14 * 512
        ##################################################################################
        ##################################################################################
        x = F.relu(F.batch_norm(self.conv14(x),running_mean=torch.randn(512).cuda(),running_var=torch.ones(512).cuda(),
            weight = Parameter(torch.Tensor(512)).cuda(),bias = Parameter(torch.Tensor(512)).cuda(),training=True, momentum=0.1, eps=1e-05),inplace=True) #56 * 56 * 64 depthwise
        x7_1 = F.batch_norm(self.conv15(x),running_mean=torch.randn(256).cuda(),running_var=torch.ones(256).cuda(),
            weight = Parameter(torch.Tensor(256)).cuda(),bias = Parameter(torch.Tensor(256)).cuda(),training=True, momentum=0.1, eps=1e-05)
        weight7 = torch.randn((256,256,1,1),requires_grad=True).cuda()
        weight7_flip = torch.flip(weight7,[1])
        weight7 = torch.cat((weight7,weight7_flip),1)
        x7_2 = F.batch_norm(F.conv2d(x,weight7,bias=None,padding=0,groups=1),running_mean=torch.randn(256).cuda(),running_var=torch.ones(256).cuda(),
            weight = Parameter(torch.Tensor(256)).cuda(),bias = Parameter(torch.Tensor(256).cuda()))
        x = torch.cat((x7_1,x7_2),1)
        x = F.relu(x,inplace=True) # 14 * 14 * 512
        ###################################################################################
        x = F.relu(F.batch_norm(self.conv16(x),running_mean=torch.randn(512).cuda(),running_var=torch.ones(512).cuda(),
            weight = Parameter(torch.Tensor(512)).cuda(),bias = Parameter(torch.Tensor(512)).cuda()),inplace=True) #56 * 56 * 64 depthwise
        x8_1 = F.batch_norm(self.conv17(x),running_mean=torch.randn(256).cuda(),running_var=torch.ones(256).cuda(),
            weight = Parameter(torch.Tensor(256)).cuda(),bias = Parameter(torch.Tensor(256)).cuda(),training=True, momentum=0.1, eps=1e-05)
        weight8 = torch.randn((256,256,1,1),requires_grad=True).cuda()
        weight8_flip = torch.flip(weight8,[1])
        weight8 = torch.cat((weight8,weight8_flip),1)
        x8_2 = F.batch_norm(F.conv2d(x,weight8,bias=None,padding=0,groups=1),running_mean=torch.randn(256).cuda(),running_var=torch.ones(256).cuda(),
            weight = Parameter(torch.Tensor(256)).cuda(),bias = Parameter(torch.Tensor(256)).cuda(),training=True, momentum=0.1, eps=1e-05)
        x = torch.cat((x8_1,x8_2),1)
        x = F.relu(x,inplace=True) # 14 * 14 * 512
        ####################################################################################
        x = F.relu(F.batch_norm(self.conv18(x),running_mean=torch.randn(512).cuda(),running_var=torch.ones(512).cuda(),
            weight = Parameter(torch.Tensor(512)).cuda(),bias = Parameter(torch.Tensor(512)).cuda(),training=True, momentum=0.1, eps=1e-05),inplace=True) #56 * 56 * 64 depthwise
        x9_1 = F.batch_norm(self.conv19(x),running_mean=torch.randn(256).cuda(),running_var=torch.ones(256).cuda(),
            weight = Parameter(torch.Tensor(256)).cuda(),bias = Parameter(torch.Tensor(256)).cuda(),training=True, momentum=0.1, eps=1e-05)
        weight9 = torch.randn((256,256,1,1),requires_grad=True).cuda()
        weight9_flip = torch.flip(weight9,[1])
        weight9 = torch.cat((weight9,weight9_flip),1)
        x9_2 = F.batch_norm(F.conv2d(x,weight9,bias=None,padding=0,groups=1),running_mean=torch.randn(256).cuda(),running_var=torch.ones(256).cuda(),
            weight = Parameter(torch.Tensor(256)).cuda(),bias = Parameter(torch.Tensor(256)).cuda(),training=True, momentum=0.1, eps=1e-05)
        x = torch.cat((x9_1,x9_2),1)
        x = F.relu(x,inplace=True) # 14 * 14 * 512
        #####################################################################################
        x = F.relu(F.batch_norm(self.conv20(x),running_mean=torch.randn(512).cuda(),running_var=torch.ones(512).cuda(),
            weight = Parameter(torch.Tensor(512)).cuda(),bias = Parameter(torch.Tensor(512)).cuda(),training=True, momentum=0.1, eps=1e-05),inplace=True) #56 * 56 * 64 depthwise
        x10_1 = F.batch_norm(self.conv21(x),running_mean=torch.randn(256).cuda(),running_var=torch.ones(256).cuda(),
            weight = Parameter(torch.Tensor(256)).cuda(),bias = Parameter(torch.Tensor(256)).cuda(),training=True, momentum=0.1, eps=1e-05)
        weight10 = torch.randn((256,256,1,1),requires_grad=True).cuda()
        weight10_flip = torch.flip(weight10,[1])
        weight10 = torch.cat((weight10,weight10_flip),1)
        x10_2 = F.batch_norm(F.conv2d(x,weight10,bias=None,padding=0,groups=1),running_mean=torch.randn(256).cuda(),running_var=torch.ones(256).cuda(),
            weight = Parameter(torch.Tensor(256)).cuda(),bias = Parameter(torch.Tensor(256)).cuda(),training=True, momentum=0.1, eps=1e-05)
        x = torch.cat((x10_1,x10_2),1)
        x = F.relu(x,inplace=True) # 14 * 14 * 512
        ######################################################################################
        x = F.relu(F.batch_norm(self.conv22(x),running_mean=torch.randn(512).cuda(),running_var=torch.ones(512).cuda(),
            weight = Parameter(torch.Tensor(512)).cuda(),bias = Parameter(torch.Tensor(512)).cuda(),training=True, momentum=0.1, eps=1e-05),inplace=True) #56 * 56 * 64 depthwise
        x11_1 = F.batch_norm(self.conv23(x),running_mean=torch.randn(256).cuda(),running_var=torch.ones(256).cuda(),
            weight = Parameter(torch.Tensor(256)).cuda(),bias = Parameter(torch.Tensor(256)).cuda(),training=True, momentum=0.1, eps=1e-05)
        weight11 = torch.randn((256,256,1,1),requires_grad=True).cuda()
        weight11_flip = torch.flip(weight11,[1])
        weight11 = torch.cat((weight11,weight11_flip),1)
        x11_2 = F.batch_norm(F.conv2d(x,weight11,bias=None,padding=0,groups=1),running_mean=torch.randn(256).cuda(),running_var=torch.ones(256).cuda(),
            weight = Parameter(torch.Tensor(256)).cuda(),bias = Parameter(torch.Tensor(256)).cuda(),training=True, momentum=0.1, eps=1e-05)
        x = torch.cat((x11_1,x11_2),1)
        x = F.relu(x,inplace=True) # 14 * 14 * 512
        #######################################################################################
        #######################################################################################
        x = F.relu(F.batch_norm(self.conv24(x),running_mean=torch.randn(512).cuda(),running_var=torch.ones(512).cuda(),
            weight = Parameter(torch.Tensor(512)).cuda(),bias = Parameter(torch.Tensor(512)).cuda(),training=True, momentum=0.1, eps=1e-05),inplace=True) #56 * 56 * 64 depthwise
        x12_1 = F.batch_norm(self.conv25(x),running_mean=torch.randn(512).cuda(),running_var=torch.ones(512).cuda(),
            weight = Parameter(torch.Tensor(512)).cuda(),bias = Parameter(torch.Tensor(512)).cuda(),training=True, momentum=0.1, eps=1e-05)
        weight12 = torch.randn((512,256,1,1),requires_grad=True).cuda()
        weight12_flip = torch.flip(weight12,[1])
        weight12 = torch.cat((weight12,weight12_flip),1)
        x12_2 = F.batch_norm(F.conv2d(x,weight12,bias=None,padding=0,groups=1),running_mean=torch.randn(512).cuda(),running_var=torch.ones(512).cuda(),
            weight = Parameter(torch.Tensor(512)).cuda(),bias = Parameter(torch.Tensor(512)).cuda(),training=True, momentum=0.1, eps=1e-05)
        x = torch.cat((x12_1,x12_2),1)
        x = F.relu(x,inplace=True) # 7 * 7 * 1024
        #######################################################################################
        x = F.relu(F.batch_norm(self.conv26(x),running_mean=torch.randn(1024).cuda(),running_var=torch.ones(1024).cuda(),
            weight = Parameter(torch.Tensor(1024)).cuda(),bias = Parameter(torch.Tensor(1024)).cuda(),training=True, momentum=0.1, eps=1e-05),inplace=True) #56 * 56 * 64 depthwise
        x13_1 = F.batch_norm(self.conv27(x),running_mean=torch.randn(512).cuda(),running_var=torch.ones(512).cuda(),
            weight = Parameter(torch.Tensor(512)).cuda(),bias = Parameter(torch.Tensor(512)).cuda(),training=True, momentum=0.1, eps=1e-05)
        weight13 = torch.randn((512,512,1,1),requires_grad=True).cuda()
        weight13_flip = torch.flip(weight13,[1])
        weight13 = torch.cat((weight13,weight13_flip),1)
        x13_2 = F.batch_norm(F.conv2d(x,weight13,bias=None,padding=0,groups=1),running_mean=torch.randn(512).cuda(),running_var=torch.ones(512).cuda(),
        weight = Parameter(torch.Tensor(512)).cuda(),bias = Parameter(torch.Tensor(512)).cuda(),training=True, momentum=0.1, eps=1e-05)
        x = torch.cat((x13_1,x13_2),1)
        x = F.relu(x,inplace=True) # 7 * 7 * 1024
        #######################################################################################
        x = F.avg_pool2d(x,7)
        #######################################################################################
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x
        #return F.log_softmax(x, dim=1)   
"""
     
def main():
    global args, best_prec1
    args = parser.parse_args()
    
    
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        if args.arch.startswith('mobilenet'):
            model = Net()
            print(model)
        else:
            model = models.__dict__[args.arch]()
    
    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        model = AlexNet()
        print(model)
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
    
    else:
        model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

   # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)
        

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        #target = target.cuda(async=True)
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
    

def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    file = open("./record.txt","a+")
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            #target = target.cuda(async=True)
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1, top5=top5))

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
        print_out = "top1:" + str(float(top1.avg)) + " " + "top5" + str(float(top5.avg)) + "\n"   #write_file
        file.write(print_out)
        file.close()
    return top1.avg


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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0,keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()