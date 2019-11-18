#ResNet
import numpy as np 
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torchvision import transforms as tfs
import torchvision.models as models

# densenet = models.densenet161(pretrained=True)
# pretrained_dict = densenet.state_dict()
def conv3x3(in_channel,out_channel,stride=1,padding=1,bias=False):
    return nn.Conv2d(in_channel,out_channel,3,stride=stride,padding=1,bias=False)   #返回的是用2D计算的卷积

class residual_block(nn.Module):
    def __init__(self, in_channel, out_channel, same_shape=True):
        super(residual_block, self).__init__()
        self.same_shape = same_shape
        stride=1 if self.same_shape else 2
        #print(stride)
        self.conv1 = conv3x3(in_channel, out_channel, stride=stride)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
        self.bn1 = nn.BatchNorm2d(out_channel)
        
        self.conv2 = conv3x3(out_channel, out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        if not self.same_shape:
            self.conv3 = nn.Conv2d(in_channel, out_channel, 1, stride=stride)


    def forward(self,x):
        out=self.conv1(x)
        #print('out1',out.shape)
        out=F.relu(self.bn1(out),True)  #inplace=true  计算覆盖  默认是false
        out=self.conv2(out)
        #print('out2',out.shape)
        out=F.relu(self.bn2(out),True)
        if not self.same_shape:
            x = self.conv3(x)
        return F.relu(x+out,True)
    
class Resnet(nn.Module):
    def __init__(self,in_channel,num_classes,verbose=False):
        super(Resnet,self).__init__()

        self.verbose=verbose
        self.conv1=nn.Conv2d(in_channel,64,7,2)
        self.layer1=nn.Conv2d(in_channel,64,7,2)
        self.layer2=nn.Sequential(
            nn.MaxPool2d(3,2),
            residual_block(64,64),
            residual_block(64,64)
        )
        self.layer3=nn.Sequential(
            residual_block(64,128,False),
            residual_block(128,128)
        )
        self.layer4=nn.Sequential(
            residual_block(128,256,False),
            residual_block(256,256)
        )
        self.layer5=nn.Sequential(
            residual_block(256,512,False),
            residual_block(512,512),
            nn.AvgPool2d(3)
        )
        self.classifier=nn.Linear(512,num_classes)


    def forward(self,x):
        x=self.layer1(x)
        if self.verbose:
            print('layer1:',x.shape)
        x=self.layer2(x)
        if self.verbose:
            print('layer2:',x.shape)
        x=self.layer3(x)
        if self.verbose:
            print('layer3:',x.shape)
        x=self.layer4(x)
        if self.verbose:
            print('layer4:',x.shape)
        x=self.layer5(x)
        if self.verbose:
            print('layer5:',x.shape)
        x=x.view(x.shape[0],-1)
        x=self.classifier(x)
        #print('out:',x.shape)
        return x



def train_tf(x):
    x=x.resize((96,96))
    x=x.convert('RGB')
    im_aug = tfs.Compose([
        tfs.RandomHorizontalFlip(),
        tfs.RandomCrop(96),
        tfs.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
        tfs.ToTensor(),
        tfs.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    x = im_aug(x)
    return x

def test_tf(x):
    x=x.resize((96,96))
    x=x.convert('RGB')
    im_aug = tfs.Compose([
        tfs.CenterCrop(96),
        tfs.ToTensor(),
        tfs.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    x = im_aug(x)
    return x

train_set = ImageFolder('/home/videostudy/liqianqian/cv_data/self_data/training',transform=train_tf)
#print(len(train_set))
train_data=torch.utils.data.DataLoader(train_set,batch_size=32,shuffle=True)
# print(train_data[0][0].shape)
test_set = ImageFolder('/home/videostudy/liqianqian/cv_data/self_data/testing',transform=test_tf)
test_data=torch.utils.data.DataLoader(test_set,batch_size=32,shuffle=False)


net=Resnet(3,22)
optimizer=torch.optim.SGD(net.parameters(),lr=1e-3)
criterion=nn.CrossEntropyLoss()





 
# resnet50 = models.resnet50(pretrained=True)
# print('resnet',resnet50)
# pretrained_dict =resnet50.state_dict() 
pretrain_dict = torch.load('/home/videostudy/liqianqian/cv_data/resnet50.pth')
# v_list=[]
# for k,v in pretrain_dict.items():
    # v_list=v_list.append(v)
    # print('pretrain',k,v.shape)
model_dict = net.state_dict() 
# for k,v in model_dict.items():
    # print('model',k,v.shape)
pretrain_dict =  {k: v for k, v in pretrain_dict.items() if v in model_dict} 
model_dict.update(pretrain_dict) 
net.load_state_dict(model_dict)  
# for k,v in model_dict.items():
    # print('after model:',k,v.shape)

from utils import train
train(net,train_data,test_data,20,optimizer,criterion)