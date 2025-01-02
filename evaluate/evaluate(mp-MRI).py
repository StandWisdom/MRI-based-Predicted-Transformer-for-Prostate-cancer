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
import pandas as pd
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
parser.add_argument('--freeze_num', default=None)
parser.add_argument('--gpu_list', default=[0])
parser.add_argument('--BATCH_SIZE', default=4)
parser.add_argument('--dstDir', default='./checkpoint')
# Model seting
parser.add_argument('--NUM_CLASSES', default=6)

# Data
args = parser.parse_args()

# In[]
from myDataset2 import DataSet
datasetDir = r'./data'
setName = 'testset'
valset = DataSet(ImgRootDir= r'./data/nii', # path of rawdata
                  img_list = os.path.join(datasetDir,'tables',setName+'-seq_list.xlsx'),
                  label_list= os.path.join(datasetDir,'tables',setName+'_clinicInfo.xlsx'),
                  GT_Flag = 'both',Split_rate = 1)


# In[]

class vision_net(nn.Module):
    def __init__(self,  **kwargs):
        super().__init__()
        
        model = models.mobilenet_v3_small()
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

        self.bn1 = nn.BatchNorm2d(3)  # 批归一化层
        
        # full connected layer
        self.last_linear012 = nn.Linear(9216, 6)  #576*16*3 27648
        
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

# In[]
modelPath = r'./model/0.9_model_55_2023-09-01_13.pth' # model path
model=torch.load(modelPath)

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
def result_to_df(y,pred):
    # 0,1,2,3 : out_MM_vit, out_MM
    pred0 = torch.softmax(pred[0].detach(),1)
    pred1 = torch.softmax(pred[1].detach(),1)

    
    pred = pred0.detach().argmax(1)
    pred = einops.repeat(pred,'b -> b 1')
    
    y = einops.repeat(y,'b -> b 1')
    res = torch.cat((pred,y,pred0,pred1),1)
    return res.cpu().numpy()


# In[]

model.eval()
test_frac = 1
acc_L=[]
results = []
# df
res_keys = ['out_MM_vit', 'out_MM']
res_colname = []
for res_key in res_keys:
    for gg in range(6):
        res_colname.append(res_key+str(gg))
    
df = pd.DataFrame([],columns=['pred','label']+res_colname)

n_batch = valset.df.shape[0]//args.BATCH_SIZE
n_batch = int(n_batch*test_frac)
pbar = tqdm(list(range(0,n_batch)))
batch_i = 0
for char in pbar:   
    x_,y_ = valset.generate_MMimg_batch_3D(n_sample=args.BATCH_SIZE)   
    x = torch.FloatTensor(x_).to(device)
    y = torch.LongTensor(y_).to(device)
    
    # forward
    pred = model(x)
    acc = ACC(pred[0],y) 
    plot_pred = pred[0].detach().argmax(1).cpu()    
    
    batch_i += 1
    acc_L.append(acc)
    pbar.set_description('$Evaluating iter:{},ACC={:.3f},mean_ACC={:.3f}'
                          .format(batch_i,acc,np.mean(acc_L)))
    
    #
    res = result_to_df(y,pred)
    df_t = pd.DataFrame(res,columns=['pred','label']+res_colname)
    df_t['ID'] = list(valset.batch_df['ID'])
    df = df.append(df_t)
    # break
    
print('# Accuracy of {} = {:.3f}.'.format(setName,np.mean(acc_L)))
current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") # 获取当前时间
df.to_excel(str(round(np.mean(acc_L),4))+'_'+current_time+' results.xlsx',index=0)

print('acc',accuracy_score(list(df['pred']),list(df['label'])))
print(modelPath)













