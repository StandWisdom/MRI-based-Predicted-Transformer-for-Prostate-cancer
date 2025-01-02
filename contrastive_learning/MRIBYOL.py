from byol_pytorch import BYOL
from torchvision import models
import os
import torch

from vit_pytorch import SimpleViT
from torchinfo import summary
import argparse
import time
from tqdm import tqdm

# In[]
def load_checkpoint(model,PATH):
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

# In[]
parser = argparse.ArgumentParser(description='RMRIOL')
# Normal
parser.add_argument('--MODELNAME', default='RMRIOL-Net')
parser.add_argument('--EPOCHS', default=20000)
parser.add_argument('--gpu_list', default=[0])
parser.add_argument('--BATCH_SIZE', default=4)
parser.add_argument('--ModelDir', default='./checkpoint/BYOL')
# Model seting
parser.add_argument('--INPUT_SIZE', default=[1, 3, 768, 1024])
parser.add_argument('--PATCH_SIZE', default=64) # (INPUT_SIZE//IMG_SIZE)**2
parser.add_argument('--NUM_CLASSES', default=6)

# Data
parser.add_argument('--IMG_SIZE', default=[128,128])
args = parser.parse_args()
# In[]
from myDataset2 import DataSet
datasetDir = r'G:\data\Prostate(in-house)\PCaDeepSet'
setName = 'PUTH'
trainset = DataSet(args,ImgRootDir=None,
                  img_list = os.path.join(datasetDir,'tables',setName+'-seq_list.xlsx'),
                  label_list= os.path.join(datasetDir,'tables',setName+'_clinicInfo.xlsx'),
                  GT_Flag = 'both',Split_rate = 1)

# In[]

# Bring model to GPU
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
device = torch.device("cuda:0" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu")


# In[]

model = SimpleViT(
    image_size = args.INPUT_SIZE[3],
    patch_size = args.PATCH_SIZE,
    num_classes = args.NUM_CLASSES,
    dim = 1024,
    depth = 12,
    heads = 48,
    mlp_dim = 2048
)

learner = BYOL(
    model,
    image_size = 128,
    hidden_layer = -2,
    use_momentum = False       # turn off momentum in the target encoder
)
learner.to(device)
learner.train()

optimizer = torch.optim.Adam(learner.parameters(), lr=3e-6)

# Main
for epoch in range(1,args.EPOCHS):
    start = time.time()
    model.train()
    trainset.df = trainset.df_ori
    n_batch = trainset.df.shape[0]//args.BATCH_SIZE
    pbar = tqdm(list(range(0,n_batch)))
    batch_i = 0
    for char in pbar:   
        x0,x1 = trainset.generate_img_batch_BYOL(n_sample=4)
        x0_ = torch.FloatTensor(x0).to(device)
        x1_ = torch.FloatTensor(x1).to(device)
        
        loss = learner([x0_,x1_])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_i += 1
        pbar.set_description('$Training epoch:{},iter:{},loss:{}'
                             .format(epoch,batch_i,loss))
        
    if epoch//50 == 0 and epoch!=0:
        # save your improved network
        filename = f"{epoch}_{loss}_improved-net.pt" # 生成文件名 
        torch.save(model.state_dict(), os.path.join(args.ModelDir,filename))
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            # 'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, os.path.join(args.ModelDir,filename))
        

















