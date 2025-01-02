import os,re
import numpy as np
import SimpleITK as sitk
import shutil  
import pandas as pd

import warnings
warnings.filterwarnings('ignore')


# In[]
def dcm2nii():
    '''
    This function is used to transfer multiple unidentified .dcm files according to the sequence name.
    '''
    
    f=open(r'./result.txt','w')  #保存为txt
    rootDir = r'G:\data\dcm' # Your source folder, which contains all the mp-MRI files of your patients. The patient folders are usually .dcm files.
    dstDir = r'G:\data\nii' # Your target folder, output as .nii or nii.gz format file 
    ids = os.listdir(rootDir)
    i = 0
    for idx in ids[i:]:  

        idx_new = idx

        print('{}-{}-------------------{}'.format(i,idx,idx_new),file=f)
        print('{}-{}-------------------{}'.format(i,idx,idx_new))
        
        # Creat folder
        dstPath = os.path.join(dstDir,idx_new)
        if os.path.exists(dstPath):
            shutil.rmtree(dstPath)  
            os.makedirs(dstPath)
        else:
            os.makedirs(dstPath)  
            
        # Read dcm
        dir_tmp = next(os.walk(os.path.join(rootDir, idx)))[1]
        if len(dir_tmp)==0:
            srcPath = os.path.join(rootDir,idx)   
        elif len(dir_tmp)==1:
            srcPath = os.path.join(rootDir,idx,dir_tmp[0])   
        else:
            print('error dir')
            break
       
        # 初始化SimpleITK的序列读取器
        reader = sitk.ImageSeriesReader()
        series_IDs = reader.GetGDCMSeriesIDs(srcPath)
        file_reader = sitk.ImageFileReader()
        print(srcPath,reader)
        break
        
        if series_IDs:
            for series in series_IDs:
                series_file_names =  reader.GetGDCMSeriesFileNames(srcPath, series) # 根据一个单张的dcm文件，读取这个series的metedata，即可以获取这个序列的描述符    
                # Read DICOM information
                file_reader.SetFileName(series_file_names[0])
                file_reader.ReadImageInformation()
                series_description = file_reader.GetMetaData('0008|103e')  
                series_description = series_description.replace('/','_')   # Reformat the description
                # print(series_description)
                
                # Eliminate unuseful sequences
                if not (re.search('t2',series_description) or re.search('T2',series_description) 
                        or re.search('t1',series_description) or re.search('T1',series_description)
                        or re.search('adc',series_description) or re.search('ADC',series_description)
                        or re.search('dwi',series_description) or re.search('diff',series_description) or re.search('DWI',series_description)
                        ):
                    print('skip1:{}'.format(series_description),file=f)
                    continue  
                if re.search('PosDisp',series_description) or re.search('OGep2d',series_description)\
                    or re.search('CA',series_description) or re.search('ZOOMit',series_description)\
                    or re.search('_vibe_',series_description) or re.search('_ND',series_description)\
                    or re.search('_F',series_description) or re.search('opp',series_description)\
                    or re.search('_W',series_description) or re.search('NEW_',series_description)\
                    or re.search('microV',series_description) or re.search('quick3d',series_description)\
                    or re.search('MVXD',series_description) or re.search('spair',series_description)\
                    or re.search('SPAIR',series_description) or re.search('dark-fluid',series_description)\
                    or re.search('ms_',series_description):
                    print('skip2:{}'.format(series_description))
                    continue   
                
                # Save as .nii
                # For multi-b DWI
                if (re.search('diff',series_description) or re.search('dwi_',series_description)) and not \
                    (re.search('adc',series_description) or re.search('ADC',series_description)): 
                    l = []
                    for k in range(len(series_file_names)):
                        file_reader.SetFileName(series_file_names[k])
                        file_reader.ReadImageInformation()
                        if re.search('dwi_',series_description) and not re.search('epi_dwi_',series_description):
                            sequence_name = 'ep_b'+file_reader.GetMetaData('0018|9087').replace('*','')
                        else:
                            sequence_name = file_reader.GetMetaData('0018|0024').replace('*','')
                        l.append([series_file_names[k],sequence_name])
                        
                    df_seq = pd.DataFrame(l,columns=['path','seq_name'])
                    b_types = list(df_seq.loc[:,'seq_name'].value_counts().index)
                    for b_type in b_types:
                        df_t = df_seq[df_seq['seq_name']==b_type]
                        dicom_itk = sitk.ReadImage(df_t['path'])
                        save_path = os.path.join(dstPath,series_description+'-'+b_type + '.nii.gz')
                        print("正在保存序列：{},{}".format(series_description,b_type))
                        # print("正在保存序列：{},{}".format(series_description,b_type),file=f)
                        sitk.WriteImage(dicom_itk, save_path)                   
                else:                            
                    dicom_itk = sitk.ReadImage(series_file_names)
                    save_path = os.path.join(dstPath,series_description + '.nii.gz')
                    print("正在保存序列：{}".format(series_description))
                    # print("正在保存序列：{}".format(series_description),file=f)
                    sitk.WriteImage(dicom_itk, save_path)
                    
        print('-----------------------------------------------------',file=f)
        print('-----------------------------------------------------')
        i +=1
        # break
    f.close()
    return

# dcm2nii()            
# In[]

def gen_dataset_list(rootDir = './Dataset/nii',out_path='./dataset/seq_list.xlsx',):
    
    ids = next(os.walk(rootDir))[1]
    df = pd.DataFrame([],columns=['ID','Dir',\
                                  't2_tra','t2_tra_fs','t2_cor','t2_cor_fs','t2_sag','t2_sag_fs',\
                                    'dwi_50','dwi_750','dwi_1500','dwi_2000',\
                                        'adc','t1_tra','t1_cor','t1_sag'])
    
    i=0
    for idx in ids:
        i +=1
        seqs = next(os.walk(os.path.join(rootDir, idx)))[2]
        df.loc[i,'ID'] = idx
        df.loc[i,'Dir'] = os.path.join(rootDir,idx)
        
        for seq in seqs:
            if (re.search('t2',seq) or re.search('T2',seq)) and (re.search('tra',seq) or re.search('Ax ',seq) or re.search('AX ',seq)):
                if re.search('_fs_',seq) or re.search('-fs',seq) or re.search('FS A',seq):
                    df.loc[i,'t2_tra_fs'] = seq
                else:
                    df.loc[i,'t2_tra'] = seq
            elif (re.search('t2',seq) or re.search('T2',seq)) and (re.search('sag',seq) or re.search('Sag',seq)):
                if re.search('_fs_',seq) or re.search('-fs',seq) or re.search('FS A',seq):
                    df.loc[i,'t2_sag_fs'] = seq
                else:
                    df.loc[i,'t2_sag'] = seq
            elif (re.search('t2',seq) or re.search('T2',seq)) and (re.search('cor',seq) or re.search('Cor',seq)):
                if re.search('_fs_',seq) or re.search('-fs',seq) or re.search('FS A',seq):
                    df.loc[i,'t2_cor_fs'] = seq
                else:
                    df.loc[i,'t2_cor'] = seq
            elif (re.search('dwi',seq) or re.search('ep_b',seq) or re.search('epi_',seq) or re.search('DWI ',seq)) \
                and not (re.search('ADC',seq) or re.search('adc',seq)):    
                if re.search('b50',seq) or re.search('b0',seq) and not re.search('dwi_b0_50_',seq):
                    df.loc[i,'dwi_50'] = seq   
                elif re.search('b750',seq) or re.search('b700',seq) or re.search('b800',seq) or re.search('b1000',seq):
                    df.loc[i,'dwi_750'] = seq            
                elif re.search('b1500',seq) or re.search('b1400',seq):
                    df.loc[i,'dwi_1500'] = seq
                elif re.search('b2000',seq):
                    df.loc[i,'dwi_2000'] = seq
                else:
                    df.loc[i,'dwi_1500'] = seq
            elif (re.search('ADC',seq) or re.search('adc',seq)) and not (re.search('EADC',seq) or re.search('tumor',seq)):
                df.loc[i,'adc'] = seq
            elif re.search('t1',seq) or re.search('T1',seq):
                if re.search('tra',seq):
                    df.loc[i,'t1_tra'] = seq
                elif re.search('cor',seq):
                    df.loc[i,'t1_cor'] = seq
                elif re.search('sag',seq):
                    df.loc[i,'t1_sag'] = seq
            elif  re.search('T2.nrrd',seq):
                df.loc[i,'t2_tra_fs'] = seq
    # Save   
    df.to_excel(out_path,index=0)     
    return 
      
gen_dataset_list(rootDir=r'./x',
                 out_path=r'./x-seq_list.xlsx')            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    






