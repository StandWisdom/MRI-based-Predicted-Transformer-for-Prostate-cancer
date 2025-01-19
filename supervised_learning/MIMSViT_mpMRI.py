import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models
from torchinfo import summary

from vit_pytorch import SimpleViT

from tqdm import tqdm

from sklearn.metrics import accuracy_score
import numpy as np
import einops
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
parser = argparse.ArgumentParser(description='GleasonNet')
# Normal
parser.add_argument('--MODELNAME', default='ViT')
parser.add_argument('--RESUME', default=0)
parser.add_argument('--EPOCHS', default=20000)
parser.add_argument('--freeze_num', default=None)
parser.add_argument('--gpu_list', default=[0])
parser.add_argument('--BATCH_SIZE', default=16)
parser.add_argument('--dstDir', default='./checkpoint')
# Model seting
parser.add_argument('--NUM_CLASSES', default=6)
parser.add_argument('--base_lr', default=0.001)
# Data
args = parser.parse_args()

# In[]
# df = pd.read_excel()

from myDataset2 import DataSet
datasetDir = r'G:\data\Prostate(in-house)\PCaDeepSet'
setName = 'trainset'
trainset = DataSet(ImgRootDir=r'./data',
                  img_list = os.path.join(datasetDir,'tables',setName+'-seq_list.xlsx'),
                  label_list= os.path.join(datasetDir,'tables',setName+'_clinicInfo.xlsx'),
                  GT_Flag = 'both',Split_rate = 0.95)

setName = 'trainset'
valset = DataSet(ImgRootDir=r'./data',
                  img_list = os.path.join(datasetDir,'tables',setName+'-seq_list.xlsx'),
                  label_list= os.path.join(datasetDir,'tables',setName+'_clinicInfo.xlsx'),
                  GT_Flag = 'both',Split_rate = -0.05)



# In[]

class vision_net(nn.Module):
    def __init__(self,  **kwargs):
        super().__init__()
        
        model = models.mobilenet_v3_small() # without mpMRI foundation model. Can be customized based on your own pre-trained network
        if os.path.exists('./model/foundationModel.pth'):
            model.load_state_dict('./model/foundationModel.pth')
        self.myCNN = nn.Sequential(*list(model.children())[0]) #19 28
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.bn = nn.BatchNorm2d(576)  # 批归一化层

        # Hyper parameters
        self.batchsize = 4
        
    def encoding_cnn(self,input):
        input = einops.rearrange(input, 'b z c w h -> (b z) c w h ') 
        x = self.myCNN(input)
        out = self.relu(x)      
        # x = self.bn(x)
        # out = self.dropout(x)   
        return out
    
    def forward(self,input):
        x = self.encoding_cnn(input)
        out = einops.rearrange(x, '(b z) c w h -> b z c w h',b=self.batchsize) 
        return out
        
class myCNNViT_MM(nn.Module):
    def __init__(self,  **kwargs):
        super().__init__()
        # Hyper parameters
        self.batchsize = 4
        self.z_num = 16
        self.patch_size = 7
        
        # Layer        
        self.conv_0 = nn.Conv2d(576, 3, kernel_size=1, stride=1, padding=0) # 576-7 ; 48-13
        self.conv_1 = nn.Conv2d(576, 3, kernel_size=1, stride=1, padding=0)
        self.conv_2 = nn.Conv2d(576, 3, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5) # W.S
        self.pool = nn.AdaptiveAvgPool2d(1)

        # self.bn0 = nn.BatchNorm2d(576)  # 批归一化层
        self.bn1 = nn.BatchNorm2d(3)  # 批归一化层
        
        # full connected layer
        self.last_linear012 = nn.Linear(9216, 6) #576*16 
        # self.last_linear012 = nn.Linear(27648, 6) #576*16*3 27648
        
        # Net
        self.net_t2 = vision_net()
        self.net_adc = vision_net()
        self.net_dwi = vision_net()
        
        self.myViT_MM = SimpleViT(
            image_size = self.patch_size*4,
            patch_size = self.patch_size,
            num_classes = 6,
            dim = 2048,
            depth = 24,
            heads = 12,
            mlp_dim = 2048
        )
    
    def encoding_cnn_MM(self,input):
        # multi-modality
        input_sub = []
        for i in range(input.shape[1]//self.z_num):
            input_sub.append(input[:,i*self.z_num:(i+1)*self.z_num,])
        out0 = self.net_t2(input_sub[0])
        out1 = self.net_adc(input_sub[1])
        out2 = self.net_dwi(input_sub[2])
        return out0,out1,out2
    
    def VisionNet(self,x):
        x0_,x1_,x2_ = x[0],x[1],x[2]       
       
        x0 = einops.rearrange(x0_, 'b z c w h -> b (z c) w h') 
        x1 = einops.rearrange(x1_, 'b z c w h -> b (z c) w h') 
        x2 = einops.rearrange(x2_, 'b z c w h -> b (z c) w h') 
        
        # CNN branch of MM
        x0 = self.pool(x0)
        x1 = self.pool(x1)
        x2 = self.pool(x2)
        
        x0 = einops.rearrange(x0, 'b c w h -> b (c w h)') 
        x0 = self.relu(x0)
        x1 = einops.rearrange(x1, 'b c w h -> b (c w h)') 
        x1 = self.relu(x1)
        x2 = einops.rearrange(x2, 'b c w h -> b (c w h)') 
        x2 = self.relu(x2)
        
        # x012 = torch.cat((x0,x1,x2),1) # If you use the concat solution, you need to adjust the dimension of self.last_linear012
        x012 = x0+x1-x2
        out_MM = self.last_linear012(x012)

        return out_MM
    
    def forward(self,input):
        x0,x1,x2 = self.encoding_cnn_MM(input)
        out_MM = self.VisionNet([x0,x1,x2])
        
        # For ViT
        x0 = einops.rearrange(x0, 'b z c w h -> (b z) c w h ') 
        x1 = einops.rearrange(x1, 'b z c w h -> (b z) c w h ') 
        x2 = einops.rearrange(x2, 'b z c w h -> (b z) c w h ') 
        
        x0 = self.conv_0(x0)
        x0 = self.relu(x0)
        x1 = self.conv_1(x1)
        x1 = self.relu(x1)
        x2 = self.conv_2(x2)
        x2 = self.relu(x2)
        
        x0 = self.bn1(x0)
        x1 = self.bn1(x1)
        x2 = self.bn1(x2)
        
        x0 = einops.rearrange(x0, '(b z) c w h -> b z c w h',b=self.batchsize) 
        x1 = einops.rearrange(x1, '(b z) c w h -> b z c w h',b=self.batchsize) 
        x2 = einops.rearrange(x2, '(b z) c w h -> b z c w h',b=self.batchsize) 
        
        x012 = torch.cat((x0,x1,x2),1)  
        
        # Reshape
        x012 = einops.rearrange(x012, 'b (z1 z2) c w h -> b c (z1 w) (z2 h)', 
                                  z1=4, h=self.patch_size, w=self.patch_size) 
        
        # ViT
        out_MM_vit = self.myViT_MM(x012)  
        return [out_MM_vit, out_MM]
    

# build network
model = myCNNViT_MM()

## 设置优化器
optimizer = torch.optim.Adam([
        {"params":model.net_t2.parameters(),"lr":0.001},
        {"params":model.net_adc.parameters(),"lr":0.0001},
        {"params":model.net_dwi.parameters(),"lr":0.0001}],
        lr=0.01, #默认参数
    )

# In[]
# If you hold a checkpoint
if isinstance(args.RESUME,str):
    path_checkpoint = args.RESUME  # 断点路径
    checkpoint = torch.load(path_checkpoint)  # 加载断点

    model.load_state_dict(checkpoint['net'])  # 加载模型可学习参数

    optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
    start_epoch = checkpoint['epoch']  # 设置开始的epoch
else:
    start_epoch = 0

# In[]
# Loss function
def loss_similarity(x, y):
    x = F.normalize(x.detach(), dim=-1, p=2)
    y = F.normalize(y.detach(), dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

def loss_multiTask(x,y):
    loss_log = []
    # Consistency loss
    x_num = len(x)
    loss_consistency = 0
    for i in range(x_num):
        for j in range(x_num):
            if not i==j and i<j:
                loss_consistency += loss_similarity(x[i],x[j]).mean()
    loss_log.append(loss_consistency)
    # Classification loss
    loss_cls = 0
    loss_fuc = nn.CrossEntropyLoss()
    w = torch.Tensor([1,0.5])
    for i in range(x_num):
        loss_cls_ = loss_fuc(x[i],y)*w[i]
        loss_cls += loss_cls_
        loss_log.append(loss_cls_)
    
    loss_overall = loss_cls + 0.1*(loss_consistency)
    return loss_overall,loss_log

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.base_lr * (0.1 ** (epoch // 500000))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

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

# In[]

# Main
writer = SummaryWriter()
for args.train_epoch in range(start_epoch,args.EPOCHS):    
    # Initialze    
    start = time.time()
    model.train()    
    trainset.df = trainset.df_ori
    valset.df = valset.df_ori

    # Starting
    n_batch = trainset.df.shape[0]//args.BATCH_SIZE
    pbar = tqdm(list(range(0,n_batch)))
    batch_i = 0
    ACCs = []
    LOSSes = []
    for char in pbar:   
        x_, y_ = trainset.generate_MMimg_batch_3D(n_sample=args.BATCH_SIZE)   
        x_train,y_train = x_,y_
        x = torch.FloatTensor(x_).to(device)
        y = torch.LongTensor(y_).to(device)

        # train
        pred = model(x)
        loss,loss_log = loss_multiTask(pred, y)  # 计算误差        
        acc = ACC(pred[0],y) 
        acc1 = ACC(pred[1],y)
        plot_pred = pred[0].argmax(1).cpu().detach().numpy()
        
        '''
        # Validate
        x_, y_ = valset.generate_MMimg_batch_3D(n_sample=args.BATCH_SIZE)   
        x = torch.FloatTensor(x_).to(device)
        y = torch.LongTensor(y_).to(device)
        # forward
        pred = model(x)
        loss_val,loss_val_log = loss_multiTask(pred, y)  # 计算误差
        acc_val = ACC(pred[0],y) 
        
        LOSSes.append([loss.data.cpu().numpy(),loss_val.data.cpu().numpy()])
        ACCs.append([acc,acc_val])
        '''
        
        loss_val,loss_val_log,acc_val = 0,0,0
        LOSSes.append([loss.data.cpu().numpy(),loss_val])
        ACCs.append([acc,acc_val])
        
        # Plot log.
        if batch_i<10:
            pbar.set_description('$ Traning loop, Epoch:{},iter:{},loss:{:.5f},ACC={:.2f},label={},plot_pred={} | Val,loss:{:.5f},ACC={:.2f}'
                                  .format(args.train_epoch,batch_i,loss.data,acc,y_train,plot_pred,
                                          loss_val,acc_val))  
        else:
            acc_exp_train = np.mean(ACCs[batch_i-10:batch_i],axis=0)[0]
            acc_exp_val = np.mean(ACCs[batch_i-10:batch_i],axis=0)[1]
            pbar.set_description('$ Traning loop, Epoch:{},iter:{},loss:{:.5f},ACC={:.2f},label={},plot_pred={} | Val,loss:{:.5f},ACC={:.2f} | Train, exp ACC:{:.2f}, Val, exp ACC:{:.2f}'
                                  .format(args.train_epoch,batch_i,loss.data,acc,y_train,plot_pred,
                                          loss_val,acc_val,
                                          acc_exp_train,acc_exp_val))  
            
        try:
            # print('****')
            n_iter = args.train_epoch*n_batch+batch_i
            
            # ACC
            writer.add_scalar('Accuracy/train', acc, n_iter)         
            writer.add_scalar('Accuracy/train_mm', acc1, n_iter)                
            # writer.add_scalar('Accuracy/val', acc_val, n_iter)
            
            # LOSS
            writer.add_scalar('Loss/train', loss.data.cpu().numpy(), n_iter)
            # writer.add_scalar('Loss/val', loss_val.data.cpu().numpy(), n_iter)
            
            writer.add_scalar('Loss/train_ci', loss_log[0], n_iter)            
            # writer.add_scalar('Loss/val_ci', loss_val_log[0], n_iter)
            
            writer.add_scalar('Loss/train_ViT_mm', loss_log[1], n_iter)            
            # writer.add_scalar('Loss/val_ViT_t2', loss_val_log[1], n_iter)
            
            writer.add_scalar('Loss/train_CNN_mm', loss_log[2], n_iter)            
            # writer.add_scalar('Loss/val_CNN_mm', loss_val_log[2], n_iter)

            
        except:
            print('tensorboard error.')
        
        # adjust_learning_rate(optimizer,n_iter) # 动态学习率
        optimizer.zero_grad()  # 清除梯度
        loss.backward()
        optimizer.step()        
        batch_i += 1
        
        # 保存整个模型  
        MaxAccumulation = 40
        # print(len(ACCs),np.mean(np.array(ACCs)[-MaxAccumulation:]))
        if len(ACCs)>MaxAccumulation and np.mean(np.array(ACCs)[-MaxAccumulation:],axis=0)[0]>0.75:
            current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H") # 获取当前时间
            filename = f"{round(np.mean(ACCs[-MaxAccumulation:],axis=0)[0],4)}_model_{args.train_epoch}_{current_time}.pth" # 生成文件名 
            if not os.path.exists(os.path.join('./checkpoint',filename)):
                torch.save(model, os.path.join('./checkpoint',filename))
                print('Find model...')

    # Plot log
    print('### Traning loop, Trainset, ACC:{:.2f}, LOSS:{:.5f} | Valset, ACC:{:.2f}, LOSS:{:.5f}.'
          .format(np.mean(ACCs,axis=0)[0],np.mean(LOSSes,axis=0)[0],np.mean(ACCs,axis=0)[1],np.mean(LOSSes,axis=0)[1]))
    print('----------------------------------------')
            
    # checkpoint
    if args.train_epoch%5==0 and args.train_epoch != 0:
        checkpoint = {"net": model.state_dict(),
                    'optimizer':optimizer.state_dict(),
                    "epoch": args.train_epoch
                    }
        filename = f"checkpoint_{args.train_epoch}.pth" # 生成文件名 
        if not os.path.exists(os.path.join('./checkpoint',filename)):
            torch.save(checkpoint, os.path.join('./checkpoint',filename))
         

writer.close()
