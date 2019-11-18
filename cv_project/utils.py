from datetime import datetime

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable


def get_acc(output, label):
    total = output.shape[0]
    #print('output shape:',output.shape)
    _, pred_label = output.max(1)                   #得到预测输出 max函数返回最大值和最大值对应的索引,pred对应索引,返回out每一行最大值组成的一维数组
    num_correct = (pred_label == label).sum().item()
    return num_correct / total


def train(net, train_data, valid_data, num_epochs, optimizer, criterion):
    # if torch.cuda.is_available():
    #     net = net.cuda()
    prev_time = datetime.now()
    for epoch in range(num_epochs):
        train_loss = 0
        train_acc = 0
        net = net.train()
        for im, label in train_data:
            # if torch.cuda.is_available():
            #     im = Variable(im.cuda())  # (bs, 3, h, w)
            #     label = Variable(label.cuda())  # (bs, h, w)
            # else:
            # im = Variable(im)
            #print(im.size())
            #print('im:',im)
            # label = Variable(label)
            # forward
            # label=torch.LongTensor(label)
            output = net(im)
            loss = criterion(output, label)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += get_acc(output, label)

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)     #divmod返回商和余数，秒换算成小时
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)    #%02d：h,m,s不够两列就补上0
        if valid_data is not None:
            valid_loss = 0
            valid_acc = 0
            net = net.eval()
            for im, label in valid_data:
                # if torch.cuda.is_available():
                #     im = Variable(im.cuda(), volatile=True)
                #     label = Variable(label.cuda(), volatile=True)
                # else:
                with torch.no_grad():   #版本更新，使用with torch.no_grad()表示不需要反向求导
                    im = Variable(im)
                    label = Variable(label)
                    # label=torch.LongTensor(label)
                    output = net(im)
                    loss = criterion(output, label)
                    valid_loss += loss.item()
                    valid_acc += get_acc(output, label)
                epoch_str = (
                    "Epoch %d. Train Loss: %f, Train Acc: %f, Valid Loss: %f, Valid Acc: %f, "
                    % (epoch, train_loss / len(train_data),
                    train_acc / len(train_data), valid_loss / len(valid_data),
                    valid_acc / len(valid_data)))
        else:
            epoch_str = ("Epoch %d. Train Loss: %f, Train Acc: %f, " %
                         (epoch, train_loss / len(train_data),
                          train_acc / len(train_data)))
        prev_time = cur_time                        #到此处使得差为0
        print(epoch_str + time_str)





