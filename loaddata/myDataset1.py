import warnings
warnings.filterwarnings('ignore')

import os,re
import pandas as pd

import SimpleITK as sitk
from matplotlib import pyplot as plt
import numpy as np

import einops
#[Creat img_list]
import torchvision.transforms as transforms
import torch
import math
import shutil
#[generate batch for computing]
from PIL import Image
import cv2
from scipy import stats


# In[Plot]
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
        
def showNii_subplot(img, save_path=None):
    x_n = 8
    y_n = 4
    plt.figure(figsize=(x_n, y_n))
    # plt.subplots_adjust(wspace=0, hspace=0)
    for y in range(y_n):
        for x in range(x_n):
            idx = y * x_n + x
            if idx >= img.shape[0]:
                continue
            plt.subplot(y_n, x_n, idx + 1)
            if img.ndim==3:
                plt.imshow(img[idx,:,:],cmap='gray')
            else:
                plt.imshow(img[idx,0,:,:],cmap='gray')
            plt.axis('off')
    plt.show()
    # plt.savefig(save_path, dpi=300)
    # plt.close()

# In[Generate batch]
class TransformImage(object):

    def __init__(self, opts, solid_resize=[320,320], crop=False,
                 random_hflip=False, random_vflip=False,random_rotate=False,random_affine=False,
                 preserve_aspect_ratio=False,norm=False,to_tensor=False): #scale=0.875

        self.input_size = opts.IMG_SIZE
        
        # self.mean = opts.mean
        # self.std = opts.std
        
        # https://github.com/tensorflow/models/blob/master/research/inception/inception/image_processing.py#L294
        self.crop = crop
        self.random_hflip = random_hflip
        self.random_vflip = random_vflip

        tfs = []
        if solid_resize:
            tfs.append(transforms.Resize(solid_resize)) #[320,320]

        if crop==1:
            tfs.append(transforms.RandomCrop(max(self.input_size)))
        elif crop==2:
            tfs.append(transforms.CenterCrop(max(self.input_size)))

        if random_hflip:
            tfs.append(transforms.RandomHorizontalFlip())

        if random_vflip:
            tfs.append(transforms.RandomVerticalFlip())
            
        if random_affine:
            tfs.append(transforms.RandomAffine(degrees=30, translate=(0, 0.2), scale=(0.9, 1), shear=(6, 9), fillcolor=66))
                    
        tfs.append(transforms.Grayscale(num_output_channels=3))# should be listed above the random_rotate
        
        if random_rotate:
            tfs.append(transforms.RandomRotation(degrees=(-30,30)))
        
        if to_tensor:
            tfs.append(transforms.ToTensor())
        if to_tensor and norm:
            # tfs.append(transforms.Normalize(mean=self.mean, std=self.std))
            tfs.append(transforms.Normalize(mean=0.5, std=0.5))
            
        self.tf = transforms.Compose(tfs)

    def __call__(self, img):
        tensor = self.tf(img)
        return tensor
    
class AugmentImg():
    def __init__(self):
        return print('AugmentImg online.')
    
    def random_bright(self,img):
        scale_factor = (np.random.random()+0.5)
        img = np.clip(img * scale_factor, 0, 4096)      
        return img
    
    def histbalance(self,mri_image):
        # 创建一个空的数组，用于保存数据增强后的图像
        mri_image_equal = np.zeros_like(mri_image)
        
        # 对每个切片进行数据增强
        for i in range(mri_image.shape[0]):
            # 获取第i个切片
            slice = mri_image[i,:,:]
            
            # 计算切片的直方图和累积分布函数
            hist, bins = np.histogram(slice.ravel(), 4096, [0, 4096])
            cdf = np.cumsum(hist)
            
            # 构建numpy掩膜数组，忽略累积分布函数为0的像素值
            cdf_m = np.ma.masked_equal(cdf, 0)
            
            # 对累积分布函数进行线性映射，使其范围在0到255之间
            cdf_m = (cdf_m - cdf_m.min()) * 4096 / (cdf_m.max() - cdf_m.min())
            
            # 对被掩膜的元素赋值为0
            cdf = np.ma.filled(cdf_m, 0).astype('uint32')
            
            # 根据累积分布函数的映射关系，得到直方图均衡后的切片
            slice_equal = cdf[slice]
            
            # 将数据增强后的切片保存到新的数组中
            mri_image_equal[i,:,:] = slice_equal
        return mri_image_equal


class DataSet():
    def __init__(self,opts,ImgRootDir=None,
                 img_list = './dataset/xxx-seq_list.xlsx',
                 label_list='./ClinicInfo/xxx_clinicInfo.xlsx',
                 GT_Flag = 'Gs-RP', Split_rate = 0.8 
                  ):
        print('Creat dataset.')
        self.ImgRootDir = ImgRootDir
        self.df = pd.read_excel(img_list,sheet_name=0)
        self.tf_img = TransformImage(opts, crop=2,norm=False,to_tensor=False) 
        self.aug_img = AugmentImg()
        
        df_GT = pd.read_excel(label_list,sheet_name=0)
        self.GT_Flag = GT_Flag
        if  GT_Flag:
            # rebuilt label list
            self.df_GT = self.check_label(df_GT,GT_Flag)
        
        self.df['ID'] = self.df['ID'].astype(str)
        self.df_GT['ID'] = self.df_GT['ID'].astype(str)
        self.df = self.df[self.df['ID'].isin(self.df_GT['ID'])].reset_index(drop=True)# convert imglist and infolist
        
        if Split_rate>0:
            self.df = self.df[:round(self.df.shape[0]*Split_rate)]
        else:
            self.df = self.df[round(self.df.shape[0]*Split_rate):]   
        # Spit the dataset list
        self.df = self.df.sample(frac=1,replace=False)
        self.df_ori = self.df.copy()
        print(self.df.shape)
        
    # Check label
    def check_label(self,df,key):
        df['label'] = df['Gs-RP'].copy()
        L = []
        pattern = re.compile(r"(\d+)\+(\d+)")
        for i in range(df.shape[0]):
            idx = df.loc[i,'ID']
            nb = df.loc[i,'Gs-NB']
            rp = df.loc[i,'Gs-RP']
            if key=='both':
                try:
                    if pattern.search(str(rp)) and not 'nan' in str(rp):
                        L.append([idx,rp])
                    elif 'nan' not in str(nb) and (pattern.search(str(nb)) or int(nb)==0):
                        L.append([idx,nb])
                    else:
                        continue
                except:
                    print(type(nb))
                    print(rp,nb,'xxxxx')
                    continue
            elif key=='rp':
                try:
                    if pattern.search(rp) or int(nb)==0:
                        L.append([idx,rp])
                    else:
                        continue
                except:
                    continue
            else:
                print('Error label key.')
        df_label = pd.DataFrame(L,columns=['ID',key])
        if df_label.shape[0]<=0:
            print('Label convert error.')
        return df_label
    
    def load_src_img(self,path):
        return
    
    def Z_score_2D(self,image):
        mean = np.mean(image)
        std = np.std(image)
        z_scores = (image - mean) / std    
        return z_scores
    
    def imgForTorch(self,img,ViT_reshape=True):
        L_imgs = [] 
        c1_value = np.sqrt(16).astype(int) # For ViT hyperparameter
        
        for i_slice in range(4,4+c1_value**2):
            try:
                img_ = img[i_slice,:,:]
            except:
                # img_ = np.random.random([img.shape[1],img.shape[2]])
                img_ = np.zeros((img.shape[1],img.shape[2]))
            # 应用Z-score
            # img_ = self.Z_score_2D(img_)    
            img_ = img_.astype(np.float)
            try:
                img_ = Image.fromarray(img_)
            except:
                try:
                    img_ = Image.fromarray(img_.astype(np.float32))
                except:
                    img_ = Image.fromarray(np.zeros((128,128)))
            # transform from Torch
            input_tensor = self.tf_img.tf(img_)
            print(np.array(input_tensor).shape,type(input_tensor))
            if type(input_tensor) == torch.Tensor:
                L_imgs.append(input_tensor.numpy())
            else:
                L_imgs.append(einops.rearrange(np.array(input_tensor),'w h c -> c w h'))
            
        if ViT_reshape:
            img = einops.rearrange(np.array(L_imgs), '(c1 c2) z w h -> z (c1 w) (c2 h) ', 
                                  c1=c1_value, h=self.tf_img.input_size[0], w=self.tf_img.input_size[1])    
        else:
            img = np.array(L_imgs)
            print(img.shape)
        return img
    
    def imgForTorch2(self,img,ViT_reshape=True):
        img = np.array(img)    
        c_z,c_x,c_y = img.shape[0]//2,img.shape[1]//2,img.shape[2]//2
        box_size = 100
        c1_value = 4
        img = img[c_z-c1_value*2:c_z+c1_value*2,c_x-box_size:c_x+box_size,c_y-box_size:c_y+box_size]
        img = self.Z_score_2D(img)
        img = einops.repeat(img, 'z w h -> z 3 w h')  
        # For ViT
        if ViT_reshape:
            img = einops.rearrange(img, '(c1 c2) z w h -> z (c1 w) (c2 h) ', 
                                  c1=c1_value, h=box_size*2, w=box_size*2)  
        else:
            img = img         
            
        return img
    
    def generate_img_batch_forCNN(self,n_sample=8):     
        if self.df.shape[0] > n_sample:
            df_t = self.df.sample(n=n_sample,axis=0,replace=False)
        else:
            df_t = self.df.sample(n=n_sample,axis=0,replace=True)
            
        paths = list(df_t['Dir'].value_counts().index)
        t2_batch = []
        label_batch = []
        for t_path in paths:            
            if self.ImgRootDir is None or os.path.isabs(t_path):
                imgPath = t_path
            else:
                try:
                    if re.search('./',t_path):
                        imgPath = t_path.split('./')[-1]
                    imgPath = os.path.join(self.ImgRootDir,imgPath) 
                except:
                    continue
                
            p_ID = list(df_t[df_t['Dir']==t_path]['ID'])[0]          
                
            # 3D
            if not df_t.loc[df_t['Dir']==t_path,'t2_tra'].isna().values[0]:
                path_t2 = os.path.join(imgPath,df_t.loc[df_t['Dir']==t_path,'t2_tra'].values[0])
            elif not df_t.loc[df_t['Dir']==t_path,'t2_tra_fs'].isna().values[0]:
                path_t2 = os.path.join(imgPath,df_t.loc[df_t['Dir']==t_path,'t2_tra_fs'].values[0])
            else:
                path_t2 = None
            
            # Load .nii/.NIFTI/.dcm
            if path_t2:
                itk_img_t2 = sitk.ReadImage(path_t2)
                img = sitk.GetArrayFromImage(itk_img_t2) 
            else:
                img = np.zeros((24,128,128))
            showNii_subplot(img)
            img = self.aug_img.histbalance(img)
            showNii_subplot(img)
            t2_batch.append(self.imgForTorch2(img,ViT_reshape=True))  
                
            # Label
            label = self.df_GT[self.df_GT['ID']==p_ID][self.GT_Flag].values[0]
            label_batch.append(self.Gs2GG(label))

        self.df = self.df.drop(df_t.index)   
        return np.array(t2_batch).astype(np.float32),np.array(label_batch)   
            
    # Read batch
    def generate_img_batch_forViT(self,n_sample=8):     
        if self.df.shape[0] > n_sample:
            df_t = self.df.sample(n=n_sample,axis=0,replace=False)
        else:
            df_t = self.df.sample(n=n_sample,axis=0,replace=True)
            
        paths = list(df_t['Dir'].value_counts().index)
        t2_batch = []
        label_batch = []
        for t_path in paths:            
            if self.ImgRootDir is None or os.path.isabs(t_path):
                imgPath = t_path
            else:
                try:
                    if re.search('./',t_path):
                        imgPath = t_path.split('./')[-1]
                    imgPath = os.path.join(self.ImgRootDir,imgPath) 
                except:
                    continue
                
            p_ID = list(df_t[df_t['Dir']==t_path]['ID'])[0]          
                
            # 3D
            if not df_t.loc[df_t['Dir']==t_path,'t2_tra'].isna().values[0]:
                path_t2 = os.path.join(imgPath,df_t.loc[df_t['Dir']==t_path,'t2_tra'].values[0])
            elif not df_t.loc[df_t['Dir']==t_path,'t2_tra_fs'].isna().values[0]:
                path_t2 = os.path.join(imgPath,df_t.loc[df_t['Dir']==t_path,'t2_tra_fs'].values[0])
            else:
                path_t2 = None
            
            # Load .nii/.NIFTI/.dcm
            if path_t2:
                itk_img_t2 = sitk.ReadImage(path_t2)
                img = sitk.GetArrayFromImage(itk_img_t2) 
            else:
                img = np.zeros((24,128,128))
                
            t2_batch.append(self.imgForTorch(img))  
                
            # Label
            label = self.df_GT[self.df_GT['ID']==p_ID][self.GT_Flag].values[0]
            label_batch.append(self.Gs2GG(label))

        self.df = self.df.drop(df_t.index)   
        return np.array(t2_batch),np.array(label_batch)   
    
    # Read batch
    def generate_img_batch_mpMRI(self,n_sample=8):     
        if self.df.shape[0] > n_sample:
            df_t = self.df.sample(n=n_sample,axis=0,replace=False)
        else:
            df_t = self.df.sample(n=n_sample,axis=0,replace=True)
            
        paths = list(df_t['Dir'].value_counts().index)
        mri_batch = []
        label_batch = []
        for t_path in paths: 
            sub_batch = []
            if self.ImgRootDir is None or os.path.isabs(t_path):
                imgPath = t_path
            else:
                try:
                    if re.search('./',t_path):
                        imgPath = t_path.split('./')[-1]
                    imgPath = os.path.join(self.ImgRootDir,imgPath) 
                except:
                    continue

            p_ID = list(df_t[df_t['Dir']==t_path]['ID'])[0]          
                
            # 3D get .nii path
            if not df_t.loc[df_t['Dir']==t_path,'t2_tra'].isna().values[0]:
                path_t2 = os.path.join(imgPath,df_t.loc[df_t['Dir']==t_path,'t2_tra'].values[0])
            elif not df_t.loc[df_t['Dir']==t_path,'t2_tra_fs'].isna().values[0]:
                path_t2 = os.path.join(imgPath,df_t.loc[df_t['Dir']==t_path,'t2_tra_fs'].values[0])
            else:
                path_t2 = None

            if not df_t.loc[df_t['Dir']==t_path,'adc'].isna().values[0]:
                path_adc = os.path.join(imgPath,df_t.loc[df_t['Dir']==t_path,'adc'].values[0])
            else:
                path_adc = None
            
            dwi_B = ['dwi_2000','dwi_1500','dwi_750']
            for b_value in dwi_B:
                if not df_t.loc[df_t['Dir']==t_path,b_value].isna().values[0]:
                    path_highB = os.path.join(imgPath,df_t.loc[df_t['Dir']==t_path,b_value].values[0])
                    break
                else:
                    path_highB = None
            
            # Load .nii/.NIFTI/.dcm
            if path_t2:
                itk_img = sitk.ReadImage(path_t2)
                img = sitk.GetArrayFromImage(itk_img) 
            else:
                img = np.random.random((20,128,128))
            sub_batch.append(self.imgForTorch(img,Flag_reshape=False))
            
            if path_adc:
                itk_img = sitk.ReadImage(path_adc)
                img = sitk.GetArrayFromImage(itk_img) 
            else:
                img = np.random.random((20,128,128))
            sub_batch.append(self.imgForTorch(img,Flag_reshape=False))
                
            if path_highB:
                itk_img = sitk.ReadImage(path_highB)
                img = sitk.GetArrayFromImage(itk_img) 
            else:
                img = np.random.random((20,128,128))
            sub_batch.append(self.imgForTorch(img,Flag_reshape=False))

            # Convert input
            # print(np.array(sub_batch).shape)
            img = einops.rearrange(np.array(sub_batch), 'd z c w h -> (d z) c w h')  
            # print(img.shape)            
            # print(img.shape)
            img = einops.rearrange(img, 'b c w h -> b w h c') 
            img = einops.rearrange(img, '(b1 b2) w h c -> (b1 w) (b2 h) c',  b1=6) 
            img = einops.rearrange(img, 'w h c-> c w h') 
            # img = np.array(sub_batch)
            # print(img.shape)
            mri_batch.append(img)
                
            # Label
            label = self.df_GT[self.df_GT['ID']==p_ID][self.GT_Flag].values[0]
            label_batch.append(self.Gs2GG(label))

        self.df = self.df.drop(df_t.index)   
        return np.array(mri_batch),np.array(label_batch)   
    
    # Read batch
    def generate_img_batch_BYOL(self,n_sample=8,p_RandEchoSeq=0.5):     
        if self.df.shape[0] > n_sample:
            df_t = self.df.sample(n=n_sample,axis=0,replace=False)
        else:
            df_t = self.df.sample(n=n_sample,axis=0,replace=True)
            
        paths = list(df_t['Dir'].value_counts().index)
        mri_batch = []
        label_batch = []
        for t_path in paths:             
            if self.ImgRootDir is None or os.path.isabs(t_path):
                imgPath = t_path
            else:
                try:
                    if re.search('./',t_path):
                        imgPath = t_path.split('./')[-1]
                    imgPath = os.path.join(self.ImgRootDir,imgPath) 
                except:
                    continue

            p_ID = list(df_t[df_t['Dir']==t_path]['ID'])[0]          
                
            # 3D get .nii path
            if not df_t.loc[df_t['Dir']==t_path,'t2_tra'].isna().values[0]:
                path_t2 = os.path.join(imgPath,df_t.loc[df_t['Dir']==t_path,'t2_tra'].values[0])
            elif not df_t.loc[df_t['Dir']==t_path,'t2_tra_fs'].isna().values[0]:
                path_t2 = os.path.join(imgPath,df_t.loc[df_t['Dir']==t_path,'t2_tra_fs'].values[0])
            else:
                path_t2 = None

            if not df_t.loc[df_t['Dir']==t_path,'adc'].isna().values[0]:
                path_adc = os.path.join(imgPath,df_t.loc[df_t['Dir']==t_path,'adc'].values[0])
            else:
                path_adc = None
            
            dwi_B = ['dwi_2000','dwi_1500','dwi_750']
            for b_value in dwi_B:
                if not df_t.loc[df_t['Dir']==t_path,b_value].isna().values[0]:
                    path_highB = os.path.join(imgPath,df_t.loc[df_t['Dir']==t_path,b_value].values[0])
                    break
                else:
                    path_highB = None
            
            # Load .nii/.NIFTI/.dcm   
            # x0
            sub_batch = []
            if path_t2:
                itk_img = sitk.ReadImage(path_t2)
                img = sitk.GetArrayFromImage(itk_img) 
            else:
                img = np.random.random((20,128,128))
            sub_batch.append(self.imgForTorch(img,Flag_reshape=False))
            
            if path_adc:
                itk_img = sitk.ReadImage(path_adc)
                img = sitk.GetArrayFromImage(itk_img) 
            else:
                img = np.random.random((20,128,128))
            sub_batch.append(self.imgForTorch(img,Flag_reshape=False))
             
            if path_highB:
                itk_img = sitk.ReadImage(path_highB)
                img = sitk.GetArrayFromImage(itk_img) 
            else:
                img = np.random.random((20,128,128))
            sub_batch.append(self.imgForTorch(img,Flag_reshape=False))

            # Convert input
            img = einops.rearrange(np.array(sub_batch), 'd z c w h -> (d z) c w h')  
            img = einops.rearrange(img, 'b c w h -> b w h c') 
            img = einops.rearrange(img, '(b1 b2) w h c -> (b1 w) (b2 h) c',  b1=6) 
            img = einops.rearrange(img, 'w h c-> c w h') 
            mri_batch.append(img)
            
            # x1
            sub_batch = []
            if path_t2:
                itk_img = sitk.ReadImage(path_t2)
                img = sitk.GetArrayFromImage(itk_img) 
            else:
                img = np.random.random((20,128,128))
            sub_batch.append(self.imgForTorch(img,Flag_reshape=False))
            
            if p_RandEchoSeq>0:
                BYOL_Flag = np.random.rand() # Add random seed
            else:
                BYOL_Flag = -1
            if path_adc and BYOL_Flag<p_RandEchoSeq:
                itk_img = sitk.ReadImage(path_adc)
                img = sitk.GetArrayFromImage(itk_img) 
            else:
                img = np.random.random((20,128,128))
            sub_batch.append(self.imgForTorch(img,Flag_reshape=False))
            
            BYOL_Flag = np.random.rand()    
            if path_highB and BYOL_Flag<p_RandEchoSeq:
                itk_img = sitk.ReadImage(path_highB)
                img = sitk.GetArrayFromImage(itk_img) 
            else:
                img = np.random.random((20,128,128))
            sub_batch.append(self.imgForTorch(img,ViT_reshape=False))

            # Convert input
            img = einops.rearrange(np.array(sub_batch), 'd z c w h -> (d z) c w h')  
            img = einops.rearrange(img, 'b c w h -> b w h c') 
            img = einops.rearrange(img, '(b1 b2) w h c -> (b1 w) (b2 h) c',  b1=6) 
            img = einops.rearrange(img, 'w h c-> c w h') 
            label_batch.append(img)

        self.df = self.df.drop(df_t.index)   
        return np.array(mri_batch),np.array(label_batch) 


    # For GT    
    def Gs2GG(self,gs):
        x = str(gs) 
        # print(x)
        try:
            if int(gs)==0:
                x_=0
        except:
            if '3+4' in x:
                x_ = 2
            elif '4+3' in x:
                x_ = 3
            elif ('4+4' in x) or ('5+3' in x) or ('3+5' in x):
                x_ = 4
            elif ('4+5' in x) or ('5+5' in x) or ('5+4' in x):
                x_ = 5
            elif ('3+3' in x) or ('3+2' in x) or ('2+3' in x) or ('2+4' in x) or\
                ('4+2' in x) or ('5+1' in x) or ('1+5' in x) or ('2+2' in x) or\
                ('1+3' in x) or ('3+1' in x) or ('1+4' in x) or ('4+1' in x)\
                ('2+1' in x) or ('1+2' in x) or ('1+1' in x):
                x_ = 1
            else:
                print(x,'label error')
        return x_
    
# In[]   

# In[]
import argparse
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Testing Dataset.')
    parser.add_argument('--IMG_SIZE', default=[128,128])
    args = parser.parse_args()

    datasetDir = r'G:\data\Prostate(in-house)\PCaDeepSet'
    setName = 'PUTH'
    set0 = DataSet(args,ImgRootDir=r'G:\data\Prostate(in-house)\PCaDeepSet',
                      img_list = os.path.join(datasetDir,'tables',setName+'-seq_list.xlsx'),
                      label_list= os.path.join(datasetDir,'tables',setName+'_clinicInfo.xlsx'),
                      GT_Flag = 'both',Split_rate = 1 )
    
    df = set0.df_GT
    df0 = set0.df_ori
    batch_size = 16
    
    for i in range(df.shape[0]//batch_size):
        x,label = set0.generate_img_batch_forCNN(n_sample=1)
        print(i,'-----')

