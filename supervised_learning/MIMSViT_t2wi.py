import argparse
import os

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from vit_pytorch import SimpleViT

from tqdm import tqdm

from sklearn.metrics import accuracy_score
import numpy as np

import time
import datetime
# In[Visualize]
from matplotlib import pyplot as plt
def showNii(img):
    if img.ndim==2:
        plt.imshow(img,cmap='gray')
        return
    
    for i in range(img.shape[0]):
        if img.ndim==3:
            plt.imshow(img[i,:,:],cmap='gray')
        else:
            plt.imshow(img[i,0,:,:],cmap='gray')
        plt.show()
    return

# In[Metrics]
def ACC(pred,truth):
    pred_np = pred.argmax(1).cpu().detach().numpy()
    
    truth_np = truth.cpu().numpy()
    acc = accuracy_score(truth_np,pred_np)
    return acc
# In[]
parser = argparse.ArgumentParser(description='MIMSViT_t2wi')
# Normal
parser.add_argument('--MODELNAME', default='ViT')
parser.add_argument('--EPOCHS', default=20000)
parser.add_argument('--freeze_num', default=None)
parser.add_argument('--gpu_list', default=[0])
parser.add_argument('--BATCH_SIZE', default=8)
parser.add_argument('--dstDir', default='./checkpoint')
# Model seting
parser.add_argument('--INPUT_SIZE', default=[1, 3, 800, 800])
parser.add_argument('--PATCH_SIZE', default=200) # (INPUT_SIZE//IMG_SIZE)**2
parser.add_argument('--NUM_CLASSES', default=6)

# Data
parser.add_argument('--IMG_SIZE', default=[128,128])
args = parser.parse_args()

# In[]
# df = pd.read_excel()

from myDataset2 import DataSet
datasetDir = r'G:\data\Prostate(in-house)\PCaDeepSet'
setName = 'trainset'
trainset = DataSet(args,ImgRootDir=r'G:\data\Prostate(in-house)\PCaDeepSet',
                  img_list = os.path.join(datasetDir,'tables',setName+'-seq_list.xlsx'),
                  label_list= os.path.join(datasetDir,'tables',setName+'_clinicInfo.xlsx'),
                  GT_Flag = 'both',Split_rate = -0.2)

setName = 'valset'
valset = DataSet(args,ImgRootDir=r'G:\data\Prostate(in-house)\PCaDeepSet',
                  img_list = os.path.join(datasetDir,'tables',setName+'-seq_list.xlsx'),
                  label_list= os.path.join(datasetDir,'tables',setName+'_clinicInfo.xlsx'),
                  GT_Flag = 'both',Split_rate = 0.05)



df = trainset.df_ori
df0 = valset.df_ori
# In[]
model = SimpleViT(
    image_size = args.INPUT_SIZE[2],
    patch_size = args.PATCH_SIZE,
    num_classes = args.NUM_CLASSES,
    dim = 1024,
    depth = 8,
    heads = 8,
    mlp_dim = 2048
)

# 设置优化器
optimizer = torch.optim.SGD(model.parameters(), lr=0.00001, momentum=0.9,weight_decay=1e-2)
loss_func = nn.CrossEntropyLoss()

# In[]
def training_loop(model,optimizer,datasets,args,writer):
    if not type(datasets) == list:
        trainset = datasets
    else:
        trainset = datasets[0]
        valset = datasets[1]
    # Initialze sets        
    trainset.df = trainset.df_ori
    valset.df = valset.df_ori
    
    # Starting
    n_batch = trainset.df.shape[0]//args.BATCH_SIZE
    pbar = tqdm(list(range(0,n_batch)))
    batch_i = 0
    ACCs = []
    LOSSes = []
    for char in pbar:   
        x_, y_ = trainset.generate_Singleimg_batch_3D(n_sample=args.BATCH_SIZE)   
        x_train,y_train = x_,y_
        x = torch.FloatTensor(x_).to(device)
        y = torch.LongTensor(y_).to(device)
        # forward
        pred = model(x)
        loss = loss_func(pred, y)  # 计算误差        
          
        acc = ACC(pred,y) 
        plot_pred = pred.argmax(1).cpu().detach().numpy()

        if not type(datasets) == list:    
            ACCs.append(acc)
            LOSSes.append(loss.data.cpu().numpy())
        else:
            try:
                x_, y_ = valset.generate_Singleimg_batch_3D(n_sample=args.BATCH_SIZE)   
            except:
                valset.df = valset.df_ori
                x_, y_ = valset.generate_img_batch_3D(n_sample=args.BATCH_SIZE)   
            x = torch.FloatTensor(x_).to(device)
            y = torch.LongTensor(y_).to(device)
            # forward
            pred = model(x)
            loss_val = loss_func(pred, y)  # 计算误差
            acc_val = ACC(pred,y) 
            
            LOSSes.append([loss.data.cpu().numpy(),loss_val.data.cpu().numpy()])
            ACCs.append([acc,acc_val])
            
            
        pbar.set_description('$ Traning loop, Epoch:{},iter:{},loss:{:.5f},ACC={:.2f},label={},plot_pred={} | Val,loss:{:.5f},ACC={:.2f}'
                              .format(args.train_epoch,batch_i,loss.data,acc,y_train,plot_pred,
                                      loss_val,acc_val))  
        try:
            n_iter = args.train_epoch*n_batch+batch_i
            writer.add_scalar('Loss/train', loss.data.cpu().numpy(), n_iter)
            writer.add_scalar('Loss/test', loss_val.data.cpu().numpy(), n_iter)
            writer.add_scalar('Accuracy/train', acc, n_iter)
            writer.add_scalar('Accuracy/test', acc_val, n_iter)
        except:
            0
          
        optimizer.zero_grad()  # 清除梯度
        loss.backward()
        optimizer.step()        
        batch_i += 1
    # Plot log
    if not type(datasets) == list:   
        print('### Traning loop, Trainset, ACC:{:.2f}, LOSS:{:.5f}.'.format(np.mean(ACCs,axis=0),np.mean(LOSSes,axis=0)))  
    else:
        print('### Traning loop, Trainset, ACC:{:.2f}, LOSS:{:.5f} | Valset, ACC:{:.2f}, LOSS:{:.5f}.'
              .format(np.mean(ACCs,axis=0)[0],np.mean(LOSSes,axis=0)[0],np.mean(ACCs,axis=0)[1],np.mean(LOSSes,axis=0)[1]))
    print('----------------------------------------')
    return model

def validation_loop(model,valSet,args,test_frac=1,setName='Val'):
    acc_L=[]
    valSet.df = valSet.df_ori
    n_batch = valSet.df.shape[0]//args.BATCH_SIZE
    n_batch = int(n_batch*test_frac)
    pbar = tqdm(list(range(0,n_batch)))
    batch_i = 0
    for char in pbar:   
        x,y = valSet.generate_Singleimg_batch_3D(n_sample=args.BATCH_SIZE)   
        x_ = torch.FloatTensor(x).to(device)
        y_ = torch.LongTensor(y).to(device)
        
        pred = model(x_)
        acc = ACC(pred,y_) 
        batch_i += 1
        pbar.set_description('$Evaluating epoch:{},iter:{},ACC={:.3f}'
                              .format(args.train_epoch,batch_i,acc))
        acc_L.append(acc)
    print('# Accuracy of {} = {:.3f}.'.format(setName,np.mean(acc_L)))
    return
# In[]

# Bring model to GPU
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

device = torch.device("cuda:0" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu")
print(device)
print(torch.cuda.get_device_name(0))

if torch.cuda.device_count() > 1:
    print("Use", torch.cuda.device_count(), 'gpus')
    model = torch.nn.DataParallel(model, device_ids=args.gpu_list)
model.to(device)

# Main
writer = SummaryWriter()
for args.train_epoch in range(args.EPOCHS):
    start = time.time()
    model.train()
    
    # Initialze sets        
    trainset.df = trainset.df_ori
    valset.df = valset.df_ori

    model = training_loop(model,optimizer,[trainset,valset],args,writer)   # training model
    writer.close()
    
    model.eval()
    if args.train_epoch%1==0:
        # validation_loop(model,trainset,args,test_frac=1,setName='train')
        validation_loop(model,valset,args,test_frac=1,setName='val')
        
        print('+++++++++++++++++++++++++++++++++++')
        # if np.mean(Val_acc)>0.80:
        #     # 保存整个模型  
        #     current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") # 获取当前时间
        #     filename = f"model_{epoch}_{current_time}.pth" # 生成文件名 
        #     torch.save(model, os.path.join('./checkpoint',filename))
