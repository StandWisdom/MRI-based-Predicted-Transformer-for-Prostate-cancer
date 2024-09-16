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

import torchio as tio

# In[Plot]
def subplot_Array(arr,keywords=None,save_path=None):            
    if keywords=='MedicalImage':
        x_n = 5
        y_n = 12
        plt.figure(figsize=(x_n, y_n))
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        for y in range(y_n):
            for x in range(x_n):
                idx = y * x_n + x
                if idx >= arr.shape[0]:
                    continue
                plt.subplot(y_n, x_n, idx + 1)
                if arr.ndim==3:
                    plt.imshow(arr[idx,:,:],cmap='gray')
                else:
                    plt.imshow(arr[idx,0,:,:],cmap='gray')
                plt.axis('off')
        plt.show()
        # plt.savefig(save_path, dpi=300)
        # plt.close()
        
def subplot_batch_Array(data,BatchMode=1):
    '''
    shape of array: [b,z,c,w,h]
    '''
    if BatchMode != 1:
        subplot_Array(data,keywords='MedicalImage',save_path=None)
    else:
        for batch_i in range(data.shape[0]):
            subplot_Array(data[batch_i],keywords='MedicalImage',save_path=None)
    
def plot_hist(data):    
    # 绘制直方图
    plt.figure()
    plt.title('Hist')
    plt.xlabel('grayscale value')
    plt.ylabel('Frequency')
    plt.hist(data, bins=50)
    plt.show()
    
# In[Generate batch]
    
class AugmentImg():
    def __init__(self,histNorm=None):
        L_tioTF = [] # list of tio transformer
        
        # Spatial transform
        # spatial_transforms = {
        #     tio.RandomAnisotropy(downsampling=(2,5)):1,
        #     tio.RandomElasticDeformation(num_control_points=(5,5,5),locked_borders=2,image_interpolation='bspline'):1,
        #     tio.RandomBiasField(coefficients=(0.1,0.2)):1,
        #     # tio.RandomAffine(scales=(1,1),degrees=5,):1,
        # }
        # L_tioTF.append(tio.OneOf(spatial_transforms, p=1))
        
        # Noise
        # L_tioTF.append(tio.RandomBlur(std=0.5,p=0.5))
        # L_tioTF.append(tio.RandomNoise(std=(0,50),p=0.5))
        
        # Norm
        if not histNorm is None:
            histNorm_imglist = list(pd.read_excel(histNorm).iloc[:,0])
            landmarks = tio.HistogramStandardization.train(histNorm_imglist)
            L_tioTF.append(tio.HistogramStandardization({'default_image_name': landmarks}))            
        L_tioTF.append(tio.RescaleIntensity(out_min_max=(-1, 1)))
        
        self.transform = tio.Compose(L_tioTF)
        return print('AugmentImg online.')
    
    def tf_img(self,x):
        x = einops.repeat(x, 'z w h -> 1 w h z')
        x_tf = self.transform(x)
        x_tf = einops.repeat(x_tf, 'c w h z -> (c z) w h')
        return x_tf

    def histbalance(self,mri_image,gray_level=65535):
        mri_image_equal = np.zeros_like(mri_image)# 创建一个空的数组，用于保存数据增强后的图像
        
        # 对每个切片进行数据增强
        for i in range(mri_image.shape[0]): 
            slice = mri_image[i,:,:]
            if np.max(slice)>gray_level:
                slice[slice>gray_level]=gray_level
            slice = slice.astype(np.uint16)# 获取第i个切片
                
            hist, bins = np.histogram(slice.ravel(), gray_level, [0, gray_level])# 计算切片的直方图和累积分布函数
            cdf = np.cumsum(hist)
            cdf_m = np.ma.masked_equal(cdf, 0)   # 构建numpy掩膜数组，忽略累积分布函数为0的像素值                  
            cdf_m = (cdf_m - cdf_m.min()) * gray_level / (cdf_m.max() - cdf_m.min()) # 对累积分布函数进行线性映射，使其范围在0到255之间               
            cdf = np.ma.filled(cdf_m, 0).astype('uint16')      # 对被掩膜的元素赋值为0      
            slice_equal = cdf[slice]# 根据累积分布函数的映射关系，得到直方图均衡后的切片
            
            # 将数据增强后的切片保存到新的数组中
            mri_image_equal[i,:,:] = slice_equal
        return mri_image_equal


class DataSet():
    def __init__(self,ImgRootDir=None,
                 img_list = './dataset/xxx-seq_list.xlsx',
                 label_list='./ClinicInfo/xxx_clinicInfo.xlsx',
                 GT_Flag = 'Gs-RP', Split_rate = 0.8):
        print('Creat dataset.')
        self.ImgRootDir = ImgRootDir
        self.df = pd.read_excel(img_list,sheet_name=0)
        self.aug_img = AugmentImg(histNorm=r'./Norm/Normalizationlist-1.xlsx') # r'./Norm/Normalizationlist-1.xlsx'
        
        if not label_list:
            df_GT = pd.DataFrame([],columns=['ID','ID_hosp','Name','Age','PSA',\
                                             'tPSA','fPSA','PIRADS','GS-NB','Gs-RP'])
            df_GT['ID']= self.df['ID']
            df_GT['GS-NB'] = np.zeros(self.df.shape[0])
            df_GT['GS-RP'] = np.zeros(self.df.shape[0])
        else:  
            df_GT = pd.read_excel(label_list,sheet_name=0)
            
        self.GT_Flag = GT_Flag
        if  GT_Flag:
            # rebuilt label list
            self.df_GT = self.check_label(df_GT,GT_Flag)
        
        self.df['ID'] = self.df['ID'].astype(str)
        self.df_GT['ID'] = self.df_GT['ID'].astype(str)
        # print(self.df[~self.df['ID'].isin(self.df_GT['ID'])])
        self.df = self.df[self.df['ID'].isin(self.df_GT['ID'])].reset_index(drop=True)# convert imglist and infolist

        if Split_rate>0:
            self.df = self.df[:round(self.df.shape[0]*Split_rate)]
        else:
            self.df = self.df[round(self.df.shape[0]*Split_rate):]   
        # Spit the dataset list
        # self.df = self.df.sample(frac=1,replace=False)
        self.df_ori = self.df.copy()
        print('df.shape',self.df.shape)
        self.batch_i = 0
        
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
    
    def img_preprocessing(self,img):  
        '''
        dim_img is [z,w,h]
        '''
        L_tio = []
        z_bias = 2
        wh_bias = 10
        if img.shape[1]!=img.shape[2]:
            max_edge = np.max(img[0,:,:].shape)
            L_tio.append(tio.CropOrPad((img.shape[0],max_edge,max_edge)))  
        if img.shape[1]<2*self.box_size+wh_bias*2 or\
            img.shape[2]<2*self.box_size+wh_bias*2 or\
            img.shape[0]<=self.c1_value**2+z_bias*2:     
            if img.shape[0]<self.c1_value**2+z_bias*2:   
                L_tio.append(tio.Resize((self.c1_value**2+z_bias*2,320,320)))     
            else:
                L_tio.append(tio.Resize((img.shape[0],320,320)))     
 
            arr_transform = tio.Compose(L_tio)
            img = einops.repeat(img, 'z w h -> 1 z w h')
            img = arr_transform(img)
            img = einops.repeat(img, 'c z w h -> (c z) w h')
            
        c_z = img.shape[0]//2+np.random.randint(-z_bias,z_bias)
        c_x,c_y = img.shape[1]//2+np.random.randint(-wh_bias,wh_bias),img.shape[2]//2+np.random.randint(-wh_bias,wh_bias)
        img = img[c_z-self.c1_value*2:c_z+self.c1_value*2,c_x-self.box_size:c_x+self.box_size,c_y-self.box_size:c_y+self.box_size]
        return img
    
    def imgForTorch(self,img,ViT_reshape=True):
        img = np.array(img)                    

        img = self.img_preprocessing(img) 
        # Preprocessing
        try:
            img = self.aug_img.histbalance(img,gray_level=65535)
        except:
            0
        img = self.aug_img.tf_img(img) # torchio
        img = self.Z_score_2D(img)
        
        img = einops.repeat(img, 'z w h -> z 3 w h')  
        # For ViT
        if ViT_reshape:
            img = einops.rearrange(img, '(c1 c2) z w h -> z (c1 w) (c2 h) ', 
                                  c1=self.c1_value, h=self.box_size*2, w=self.box_size*2)  
        else:
            img = img                 
        return img
    
    def generate_img_batch_3D(self,n_sample=8,keyword=None,batchlist=None):     
        if batchlist is None or batchlist == 'ergodic':
            if self.df.shape[0] < n_sample:
               # 如果数量不足1个batchsize 直接循环
               self.df = self.df_ori
            if batchlist == 'ergodic':
                df_t = self.df.iloc[self.batch_i*n_sample:(self.batch_i+1)*n_sample]
                print(self.batch_i,n_sample)
            else:
                df_t = self.df.sample(n=n_sample,axis=0,replace=False) # Random select           
            self.batch_df = df_t.copy()
        elif 'mm' == batchlist :
            df_t = self.batch_df
        else:
            df_t = batchlist

        # get inputs    
        paths = list(df_t['Dir'].value_counts().index) # 去除重复路径，获取单个影像路径
        img_batch = []
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
            if 't2' in keyword:
                if not df_t.loc[df_t['Dir']==t_path,keyword].isna().values[0]:
                    path = os.path.join(imgPath,df_t.loc[df_t['Dir']==t_path,keyword].values[0])
                elif not df_t.loc[df_t['Dir']==t_path,keyword+'_fs'].isna().values[0]:
                    path = os.path.join(imgPath,df_t.loc[df_t['Dir']==t_path,keyword+'_fs'].values[0])
                else:
                    path = None
            elif 'adc' in keyword: # Random
                if not df_t.loc[df_t['Dir']==t_path,keyword].isna().values[0] and np.random.rand()>0:
                    path = os.path.join(imgPath,df_t.loc[df_t['Dir']==t_path,keyword].values[0])
                else:
                    path = None
            elif 'dwi' in keyword:
                dwi_B = ['dwi_2000','dwi_1500','dwi_750']
                for b_value in dwi_B:
                    if not df_t.loc[df_t['Dir']==t_path,b_value].isna().values[0] and np.random.rand()>0:
                        path = os.path.join(imgPath,df_t.loc[df_t['Dir']==t_path,b_value].values[0])
                        break
                    else:
                        path = None
            else:
                print('Unknown seq.',keyword)
                
            # if ('adc' in keyword or 'dwi' in keyword) and np.random.rand()<0.3:
            #     path = None

            # Load .nii/.NIFTI/.dcm
            self.box_size = 100
            self.c1_value = 4
            if path is not None:
                itk_img = sitk.ReadImage(path)
                img = sitk.GetArrayFromImage(itk_img) 
                self.missImg_Flag = 0
                try:
                    img_tf = self.imgForTorch(img,ViT_reshape=False)
                except:
                    # img_tf = np.zeros((self.c1_value**2,3,self.box_size*2,self.box_size*2))               
                    img_tf = np.random.rand(self.c1_value**2,3,self.box_size*2,self.box_size*2)
                    print('path',path,img.shape)
            else:
                # img_tf = np.zeros((self.c1_value**2,3,self.box_size*2,self.box_size*2))
                img_tf = np.random.rand(self.c1_value**2,3,self.box_size*2,self.box_size*2)
                self.missImg_Flag = 1

            img_batch.append(img_tf)  
                
            # Label
            label = self.df_GT[self.df_GT['ID']==p_ID][self.GT_Flag].values[0]
            label_batch.append(self.Gs2GG(label))
        
        if batchlist is None:
            self.df = self.df.drop(df_t.index)   

        return np.array(img_batch).astype(np.float32),np.array(label_batch) 

    def generate_MMimg_batch_3D(self,n_sample=8):  
        # Multi-modality imaging
        img0,label = self.generate_img_batch_3D(n_sample=n_sample,keyword='t2_tra',batchlist=None) # 'ergodic'
        img1,_ = self.generate_img_batch_3D(n_sample=n_sample,keyword='adc',batchlist='mm')
        img2,_ = self.generate_img_batch_3D(n_sample=n_sample,keyword='dwi',batchlist='mm')
        img = np.concatenate((img0, img1,img2),axis=1)
        self.batch_i +=1
        return img,label
    
    def generate_Singleimg_batch_3D(self,n_sample=8):  
        # Multi-modality imaging
        img,label = self.generate_img_batch_3D(n_sample=n_sample,keyword='t2_tra',batchlist=None) # 'ergodic'
        # img = einops.rearrange(img, '(b z) c w h -> b z c w h',b=8) 
        # img = einops.rearrange(img, 'b (z1 z2) c w h -> b c (z1 w) (z2 h) ', 
        #                       z1=4, h=200, w=200) 
        self.batch_i +=1
        return img,label

    # For GT    
    def Gs2GG(self,gs):
        x = str(gs) 
        # print(x)
        try:
            if int(x)==0:
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
import argparse
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Testing Dataset.')
    # parser.add_argument('--IMG_SIZE', default=[128,128])
    # args = parser.parse_args()

    datasetDir = r'G:\data\Prostate(in-house)\PCaDeepSet'
    setName = 'JSPH'
    set0 = DataSet(ImgRootDir=r'G:\data\Prostate(in-house)\PCaDeepSet',
                      img_list = os.path.join(datasetDir,'tables',setName+'-seq_list.xlsx'),
                      label_list= os.path.join(datasetDir,'tables',setName+'_clinicInfo.xlsx'),
                      GT_Flag = 'both',Split_rate = 1 )
    
    # df = set0.df_ori
    # df.to_excel(setName+'-trainlist.xlsx',index=0)
    
    # df = set0.df_GT
    # df0 = set0.df_ori
    # batch_size = 4
    # n_batch = df0.shape[0]//batch_size
    # set0.batch_i = 0
    # for i in range(set0.batch_i,n_batch):
    #     x,label = set0.generate_MMimg_batch_3D(n_sample=batch_size)
    #     print(set0.batch_i,'-----',x.shape,label.shape,set0.df.shape)
    #     print('------------------------------------')
    #     # Plot
    #     # subplot_batch_Array(x,BatchMode=1)
    #     # plot_hist(x[2,15,0,])
    #     # break
