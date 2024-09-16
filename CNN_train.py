import os,re,sys
import pandas as pd
import numpy as np 
import argparse

import pretrainedmodels
import pretrainedmodels.utils as utils
import torch
import torch.nn as nn
from torchvision import transforms

import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score

import time

from myDataset import DataSet
from tqdm import tqdm

# In[]
def ACC(pred,truth):
    pred_np = pred.argmax(1).cpu().detach().numpy()
    
    truth_np = truth.cpu().numpy()
    acc = accuracy_score(truth_np,pred_np)
    return acc


# In[]
parser = argparse.ArgumentParser(description='CNN')
parser.add_argument('--MODELNAME', default='nasnetamobile')
parser.add_argument('--EPOCHS', default=20)
parser.add_argument('--nb_classes', default=6)
parser.add_argument('--freeze_num', default=-5)
parser.add_argument('--gpu_list', default=[0])
parser.add_argument('--BATCH_SIZE', default= 32)
parser.add_argument('--dstDir', default='./checkpoint')

args = parser.parse_args()
# In[Set Initial Gpu]# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu")

print(device)
print(torch.cuda.get_device_name(0))
# In[build model]

model_name = args.MODELNAME # could be fbresnet152 or inceptionresnetv2
print(pretrainedmodels.pretrained_settings[model_name])
model_info = pretrainedmodels.pretrained_settings[model_name]
model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')


dim_feats = model.last_linear.in_features # =2048
model.last_linear = nn.Linear(dim_feats, args.nb_classes)

mp = list(model.parameters())
for para in list(model.parameters())[:args.freeze_num]: #15
    para.requires_grad=False 

if torch.cuda.device_count() > 1:
    print("Use", torch.cuda.device_count(), 'gpus')
    model = torch.nn.DataParallel(model, device_ids=args.gpu_list)
model.to(device)
model.train()

## 设置优化器
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9,weight_decay=1e-3)
loss_func = nn.CrossEntropyLoss()
#loss_func = nn.CrossEntropyLoss(weight=torch.tensor([1,1,1,1,1]).to(device))


# In[]
BATCH_SIZE = args.BATCH_SIZE
EPOCHS = args.EPOCHS

MODELPATH= os.path.join(args.dstDir,model_name)
if not os.path.exists(MODELPATH):
    os.makedirs(MODELPATH)
# In[main]

trainset = DataSet(model,img_list = './dataset/img_list.xlsx',
                 alignDir =  './dataset/nii_align',label_list='./dataset/clinicInfo/JSPH-info.xlsx',
                 GT_Flag = 'Gs-RP',Split_rate = 0.8 )
valset = DataSet(model,img_list = './dataset/img_list.xlsx',
                 alignDir =  './dataset/nii_align',label_list='./dataset/clinicInfo/JSPH-info.xlsx',
                 GT_Flag = 'Gs-RP',Split_rate = -0.2)


Train_loss=[]
Train_acc=[]
Val_acc=[]
for epoch in range(EPOCHS):
    start = time.time()
    model.train()
    trainset.df = trainset.df_ori
    n_batch = trainset.df.shape[0]//BATCH_SIZE
    pbar = tqdm(list(range(0,n_batch)))
    batch_i = 0
    for char in pbar:   
        x_train,x0_train,y_train = trainset.generate_img_batch(n_sample=BATCH_SIZE)   
        # X = torch.autograd.Variable(X,requires_grad=False)
        x= torch.FloatTensor(x_train).to(device)
        y= torch.LongTensor(y_train).to(device)
        
        pred = model(x)
        loss = loss_func(pred, y)  # 计算误差
        optimizer.zero_grad()  # 清除梯度
        loss.backward()
        optimizer.step()
        
        acc = ACC(pred,y) 
        batch_i += 1
        pbar.set_description('epoch:{},iter:{},loss:{},ACC={}'.format(epoch,batch_i,loss.data,acc))


    # Validate
    model.eval()
 
    valset.df = valset.df_ori
    n_batch = valset.df.shape[0]//BATCH_SIZE
    pbar = tqdm(list(range(0,n_batch)))
    batch_i = 0
    for char in pbar:   
        x,x0,y = trainset.generate_img_batch(n_sample=BATCH_SIZE)   
        # X = torch.autograd.Variable(X,requires_grad=False)
        x_ = torch.FloatTensor(x).to(device)
        y_ = torch.LongTensor(y).to(device)
        
        pred = model(x_)
        
        acc = ACC(pred,y_) 
        batch_i += 1
        pbar.set_description('epoch:{},iter:{},loss:{},ACC={}'.format(epoch,batch_i,loss.data,acc))
        Val_acc.append(acc)
    print('# Accuracy of Valset = {}.'.format(np.mean(Val_acc)))
