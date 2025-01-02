# Import Modules #
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_auc_score

plt.rcParams['font.family'] = ['Arial']
plt.rcParams['font.size'] = 6

def bootstrap_results(y_truth, y_pred,num_bootstraps = 1000):
    n_bootstraps = num_bootstraps
    rng_seed = 42  # control reproducibility
    y_pred=y_pred
    y_true=y_truth
    rng = np.random.RandomState(rng_seed)
    tprs=[]
    fprs=[]
    aucs=[]
    threshs=[]
    base_thresh = np.linspace(0, 1, 101)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(y_pred), len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            continue
        fpr, tpr, thresh = metrics.roc_curve(y_true[indices],y_pred[indices])
        thresh=thresh[1:]
        thresh=np.append(thresh,[0.0])
        thresh=thresh[::-1]
        fpr = np.interp(base_thresh, thresh, fpr[::-1])
        tpr = np.interp(base_thresh, thresh, tpr[::-1])
        tprs.append(tpr)
        fprs.append(fpr)
        threshs.append(thresh)
    tprs = np.array(tprs)
    mean_tprs = tprs.mean(axis=0)  
    fprs = np.array(fprs)
    mean_fprs = fprs.mean(axis=0)
    
    return base_thresh, mean_tprs, mean_fprs

def check_cat(prob,thresholds):
    cat=0
    for i,v in enumerate(thresholds):
        if prob>v:
            cat=i
    return cat

def make_cat_matrix(ref, new, indices, thresholds):
    num_cats=len(thresholds)
    mat=np.zeros((num_cats,num_cats))
    for i in indices:
        row,col=check_cat(ref[i],thresholds),check_cat(new[i],thresholds)
        mat[row,col]+=1
    return mat
        
def nri(y_truth,y_ref, y_new,risk_thresholds):
    event_index = np.where(y_truth==1)[0]
    nonevent_index = np.where(y_truth==0)[0]
    event_mat=make_cat_matrix(y_ref,y_new,event_index,risk_thresholds)
    nonevent_mat=make_cat_matrix(y_ref,y_new,nonevent_index,risk_thresholds)
    events_up, events_down = event_mat[0,1:].sum()+event_mat[1,2:].sum()+event_mat[2,3:].sum(),event_mat[1,:1].sum()+event_mat[2,:2].sum()+event_mat[3,:3].sum()
    nonevents_up, nonevents_down = nonevent_mat[0,1:].sum()+nonevent_mat[1,2:].sum()+nonevent_mat[2,3:].sum(),nonevent_mat[1,:1].sum()+nonevent_mat[2,:2].sum()+nonevent_mat[3,:3].sum()
    nri_events = (events_up/len(event_index))-(events_down/len(event_index))
    nri_nonevents = (nonevents_down/len(nonevent_index))-(nonevents_up/len(nonevent_index))
    return nri_events, nri_nonevents, nri_events + nri_nonevents 


def track_movement(ref,new, indices):
    up, down = 0,0
    for i in indices:
        ref_val, new_val = ref[i],new[i]
        if ref_val<new_val:
            up+=1
        elif ref_val>new_val:
            down+=1
    return up, down

def category_free_nri(y_truth,y_ref, y_new):
    event_index = np.where(y_truth==1)[0]
    nonevent_index = np.where(y_truth==0)[0]
    events_up, events_down = track_movement(y_ref, y_new,event_index)
    nonevents_up, nonevents_down = track_movement(y_ref, y_new,nonevent_index)
    nri_events = (events_up/len(event_index))-(events_down/len(event_index))
    nri_nonevents = (nonevents_down/len(nonevent_index))-(nonevents_up/len(nonevent_index))
    return nri_events, nri_nonevents, nri_events + nri_nonevents 


def area_between_curves(y1,y2):
    diff = y1 - y2 # calculate difference
    posPart = np.maximum(diff, 0) 
    negPart = -np.minimum(diff, 0) 
    posArea = np.trapz(posPart)
    negArea = np.trapz(negPart)
    return posArea,negArea,posArea-negArea

def plot_idi(y_truth, ref_model, new_model, save=False): 
    ref_fpr, ref_tpr, ref_thresholds = metrics.roc_curve(y_truth, ref_model)
    new_fpr, new_tpr, new_thresholds = metrics.roc_curve(y_truth, new_model)
    base, mean_tprs, mean_fprs=bootstrap_results( y_truth, new_model,100)
    base2, mean_tprs2, mean_fprs2=bootstrap_results( y_truth, ref_model,100)
    is_pos,is_neg, idi_event=area_between_curves(mean_tprs,mean_tprs2)
    ip_pos,ip_neg, idi_nonevent=area_between_curves(mean_fprs2,mean_fprs)
    print('IS positive', round(is_pos,2),'IS negative',round(is_neg,2),'IDI events',round(idi_event,2))
    print('IP positive', round(ip_pos,2),'IP negative',round(ip_neg,2),'IDI nonevents',round(idi_nonevent,2))
    print('IDI =',round(idi_event+idi_nonevent,2))
    plt.figure(figsize=(3,3),dpi=600)
    ax=plt.axes()
    lw = 2
    plt.plot(base, mean_tprs, '#1B1919',linewidth=1, alpha = 0.5, label='Events New (MRI-PTPCa)' )
    plt.plot(base, mean_fprs, '#CB181E',linewidth=1, alpha = 0.5, label='Nonevents New (MRI-PTPCa)')
    plt.plot(base2, mean_tprs2, '#1B1919',linewidth=1, alpha = 0.7, linestyle='--',label='Events Reference (PI-RADS)' )
    plt.plot(base2, mean_fprs2, '#CB181E',linewidth=1, alpha = 0.7,  linestyle='--', label='Nonevents Reference (PI-RADS)')
    plt.fill_between(base, mean_tprs,mean_tprs2, color='#1B1919',alpha = 0.1, label='Integrated Sensitivity (area = %0.2f)'%idi_event)
    plt.fill_between(base, mean_fprs,mean_fprs2, color='#CB181E', alpha = 0.1, label='Integrated Specificity (area = %0.2f)'%idi_nonevent)
    
    #''' #TODO: comment out if not for breast birads
    ### BIRADS Thresholds ###
    # plt.axvline(x=0.0909,color='#09488F',linewidth=0.75,linestyle='--',alpha=.5,label='MRI-PTPCa 0/1 Border (0.5)')
    # plt.axvline(x=0.2727,color='#2473B5',linewidth=0.75,linestyle='--',alpha=.5,label='MRI-PTPCa 1/2 Border (1.5)')
    # plt.axvline(x=0.4545,color='#448FC6',linewidth=0.75,linestyle='--',alpha=.5,label='MRI-PTPCa 2/3 Border (2.5)')
    # plt.axvline(x=0.6363,color='#6DACD5',linewidth=0.75,linestyle='--',alpha=.5,label='MRI-PTPCa 3/4 Border (3.5)')
    # plt.axvline(x=0.8181,color='#C7DCF1',linewidth=0.75,linestyle='--',alpha=.5,label='MRI-PTPCa 4/5 Border (4.5)')
    plt.axvline(x=0.1,color='#09488F',linewidth=0.5,linestyle='--',alpha=.5,label='MRI-PTPCa 0/1 Border (0.5)')
    plt.axvline(x=0.28,color='#2473B5',linewidth=0.5,linestyle='--',alpha=.5,label='MRI-PTPCa 1/2 Border (1.5)')
    plt.axvline(x=0.46,color='#448FC6',linewidth=0.5,linestyle='--',alpha=.5,label='MRI-PTPCa 2/3 Border (2.5)')
    plt.axvline(x=0.64,color='#6DACD5',linewidth=0.5,linestyle='--',alpha=.5,label='MRI-PTPCa 3/4 Border (3.5)')
    plt.axvline(x=0.82,color='#C7DCF1',linewidth=0.5,linestyle='--',alpha=.5,label='MRI-PTPCa 4/5 Border (4.5)')
    
    def nri_annotation(plt, threshold):
        x_pos = base[threshold]
        x_offset=0.02
        x_offset2=x_offset
        text_y_offset=0.01
        text_y_offset2=text_y_offset
        
        if threshold==2:
            text_y_offset=0.04
            text_y_offset2=0.04
            x_offset2=0.05
            print(x_pos+x_offset, (np.mean([mean_tprs2[threshold], mean_tprs[threshold]])+text_y_offset),
                    x_pos, (np.mean([mean_tprs2[threshold], mean_tprs[threshold]])))
            
        text_y_events=np.mean([mean_tprs2[threshold], mean_tprs[threshold]])+text_y_offset
        text_y_nonevents=np.mean([mean_fprs[threshold], mean_fprs2[threshold]])+text_y_offset2
        if abs(mean_tprs[threshold]-mean_tprs2[threshold])>0.03:
            plt.annotate('', xy=(x_pos+0.02, mean_tprs2[threshold+1]), xycoords='data', xytext=(x_pos+0.02, 
                                mean_tprs[threshold]), textcoords='data', arrowprops=dict(arrowstyle='|-|',color='#1B1919',lw=0.5))
            plt.annotate('NRI$_{events}$=%0.2f'%(mean_tprs[threshold]-mean_tprs2[threshold]), 
                          xy=(x_pos+x_offset, text_y_events), xycoords='data',
                          xytext=(x_pos+x_offset, text_y_events),
                          textcoords='offset points', fontsize=6)
        if abs(mean_fprs2[threshold]-mean_fprs[threshold])>0.03:
            plt.annotate('', xy=(x_pos+0.02, mean_fprs[threshold]), xycoords='data', xytext=(x_pos+0.02, 
                                 mean_fprs2[threshold]), textcoords='data', arrowprops=dict(arrowstyle= '|-|',color='#CB181E', lw=0.5))
            plt.annotate('NRI$_{nonevents}$=%0.2f'%(mean_fprs2[threshold]-mean_fprs[threshold]), 
                         xy=(x_pos+x_offset2, text_y_nonevents), xycoords='data',
                         xytext=(x_pos+x_offset2, text_y_nonevents), 
                         textcoords='offset points', fontsize=6)
        print('Threshold =',round(x_pos,2),'NRI events =',round(mean_tprs[threshold]-mean_tprs2[threshold],4),
              'NRI nonevents =',round(mean_fprs2[threshold]-mean_fprs[threshold],4),'Total =',
              round((mean_tprs[threshold]-mean_tprs2[threshold])+(mean_fprs2[threshold]-mean_fprs[threshold]),4))
        
    nri_annotation(plt,9)
    nri_annotation(plt,27)
    nri_annotation(plt,45)
    nri_annotation(plt,63)
    nri_annotation(plt,81)
    #'''
    plt.xlim([0.0, 1.10])
    plt.ylim([0.0, 1.10])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xlabel('Calculated Risk', fontsize=6)
    plt.ylabel('Sensitivity (black), 1 - Specificity (red)', fontsize=6)
    plt.legend(loc="upper right", fontsize=6)
    plt.legend(loc=0, fontsize=6,  bbox_to_anchor=(0,0,1.2,.9))
    plt.gca().set_aspect('equal', adjustable='box')
    if save:
        plt.savefig(save,dpi=600, bbox_inches='tight')
    # look=95
    plt.show()

# In[]
# 导入所需的库
from scipy.stats import norm
# 定义一个函数，计算 IDI 和 NRI
def idinri(y_true, y_pred1, y_pred2):
    # y_true 是真实的标签，y_pred1 是第一个模型的预测概率，y_pred2 是第二个模型的预测概率
    # 首先，计算两个模型的 AUC
    auc1 = roc_auc_score(y_true, y_pred1)
    auc2 = roc_auc_score(y_true, y_pred2)
    # 然后，计算 IDI，即两个模型的 AUC 之差
    idi = auc2 - auc1
    # 接着，计算 NRI，即两个模型在事件和非事件中的预测概率的变化
    # 创建一个数据框，存储真实标签和两个模型的预测概率
    df = pd.DataFrame({'y_true': y_true, 'y_pred1': y_pred1, 'y_pred2': y_pred2})
    # 根据真实标签，分为事件组和非事件组
    event = df[df['y_true'] == 1]
    nonevent = df[df['y_true'] == 0]
    # 计算事件组中，第二个模型比第一个模型预测概率更高的比例
    event_nri = (event['y_pred2'] > event['y_pred1']).mean()
    # 计算非事件组中，第二个模型比第一个模型预测概率更低的比例
    nonevent_nri = (nonevent['y_pred2'] < nonevent['y_pred1']).mean()
    # 计算 NRI，即两个比例之和
    nri = event_nri + nonevent_nri
    # 计算 IDI 和 NRI 的 Z 统计量和 P 值
    idi_z = idi / np.sqrt((auc1 * (1 - auc1) + auc2 * (1 - auc2)) / len(y_true))
    idi_p = norm.sf(abs(idi_z)) * 2 # 双尾检验
    nri_z = nri / np.sqrt((event_nri * (1 - event_nri) + nonevent_nri * (1 - nonevent_nri)) / len(y_true))
    nri_p = norm.sf(abs(nri_z)) * 2 # 双尾检验
    # 返回 IDI 和 NRI 的值以及 P 值
    return idi, nri, idi_p, nri_p

# In[]
# 导入所需的库
import scipy.stats as st

# 定义一个类，实现 delongTest
class delongTest:
    def __init__(self, preds1, preds2, label, threshold=0.05):
        # preds1 是第一个模型的预测概率
        # preds2 是第二个模型的预测概率
        # label 是真实的标签
        # threshold 是显著性水平，默认为 0.05
        self.preds1 = preds1
        self.preds2 = preds2
        self.label = label
        self.threshold = threshold
        # self.show_result() # 显示结果
    
    def auc(self, X, Y):
        # 计算 AUC 值
        return 1 / (len(X) * len(Y)) * sum([self.kernel(x, y) for x in X for y in Y])
    
    def kernel(self, X, Y):
        # 计算 Mann-Whitney 统计量
        return 0.5 if Y == X else int(Y < X)
    
    def structural_components(self, X, Y):
        # 计算结构分量
        V10 = [1 / (len(Y)+0.001) * sum([self.kernel(x, y) for y in Y]) for x in X]
        V01 = [1 / (len(X)+0.001) * sum([self.kernel(x, y) for x in X]) for y in Y]
        return V10, V01
    
    def get_S_entry(self, V_A, V_B, auc_A, auc_B):
        # 计算协方差矩阵 S 的元素
        return 1 / (len(V_A)+0.001 - 1) * sum([(a - auc_A) * (b - auc_B) for a, b in zip(V_A, V_B)])
    
    def z_score(self, var_A, var_B, covar_AB, auc_A, auc_B):
        # 计算 Z 统计量
        return (auc_A - auc_B) / ((var_A + var_B - 2 * covar_AB)**0.5 + 1e-8)
    
    def group_preds_by_label(self, preds, actual):
        # 根据真实标签分组预测概率
        X = [p for (p, a) in zip(preds, actual) if a]
        Y = [p for (p, a) in zip(preds, actual) if not a]
        return X, Y
    
    def compute_z_p(self):
        # 计算 Z 统计量和 P 值
        X_A, Y_A = self.group_preds_by_label(self.preds1, self.label)
        X_B, Y_B = self.group_preds_by_label(self.preds2, self.label)
        
        V_A10, V_A01 = self.structural_components(X_A, Y_A)
        V_B10, V_B01 = self.structural_components(X_B, Y_B)
        
        auc_A = self.auc(X_A, Y_A)
        auc_B = self.auc(X_B, Y_B)
        
        # 计算协方差矩阵 S 的元素（covar_AB = covar_BA）
        var_A = (self.get_S_entry(V_A10, V_A10, auc_A, auc_A) / len(V_A10) +
                 self.get_S_entry(V_A01, V_A01, auc_A, auc_A) / len(V_A01))
        var_B = (self.get_S_entry(V_B10, V_B10, auc_B, auc_B) / len(V_B10) +
                 self.get_S_entry(V_B01, V_B01, auc_B, auc_B) / len(V_B01))
        covar_AB = (self.get_S_entry(V_A10, V_B10, auc_A, auc_B) / len(V_A10) +
                    self.get_S_entry(V_A01, V_B01, auc_A, auc_B) / len(V_A01))
        
        # 双尾检验
        z = self.z_score(var_A, var_B, covar_AB, auc_A, auc_B)
        p = st.norm.sf(abs(z)) * 2
        return z, p
    
    def show_result(self):
        # 显示结果
        z, p = self.compute_z_p()
        print(f"Z 统计量 = {z:.4f}, P 值 = {p:.4f}")
        if p < self.threshold:
            print("两个 ROC 曲线的差异具有统计显著性")
        else:
            print("两个 ROC 曲线的差异没有统计显著性")

# In[]
import warnings
warnings.filterwarnings('ignore')
if __name__ == '__main__':
    df_all = pd.read_excel(r'features & predictions.xlsx')    

    df_stat = pd.DataFrame([])
    h_names = ['PUTH-p','JSPH','BJFH-XC','BJFH-TZ',
                # 'AHQU-C','AHQU-E','AHQU-W',
                # 'PICAI',
               'JSPH-prospective','all'] 
    # 'PUTH-p','JSPH','BJFH-XC','BJFH-TZ',
    # 'AHQU-C','AHQU-E','AHQU-W',
    # 'PICAI','JSPH-prospective','all'
    
    for h_name in h_names:
        if h_name == 'all':
            df = df_all[df_all['HospName'].isin(h_names[:-1])].copy().reset_index(drop=True)
        else:
            df = df_all[df_all['HospName']==h_name].copy().reset_index(drop=True)
            
        # select
        df = df.dropna(subset='GG-NB').reset_index(drop=True)
        # df = df.dropna(subset='GG-RP').reset_index(drop=True)    
        
        # df['PSA'] = df['PSA'].fillna(value=0) 
        # df = df[(df['PSA'] > 4) & (df['PSA'] < 10)].reset_index(drop=True)
        
        df['PIRADS'] = df['PIRADS'].fillna(value=3)
        df = df[(df['PIRADS']<3)].reset_index(drop=True)
        
        # df = df[(df['GG-NB']!=0)].reset_index(drop=True) # 限制为前列腺癌
        
        # compare target
        df['gt'] = df['GG-NB'].apply(lambda x:1 if x>=2 else 0) # 注意修改
        df['factor_a'] = df['PIRADS'].copy()/5
        df['factor_b'] = df['deepMRI2GGGscore'].copy()/5.5
        
        # Plot idi nri pic
        savePath = os.path.join(r'H:\我的论文\【前列腺癌辅助诊断-多中心】\画图\result stat\pic\cspca compare',h_name+'_idi_nri.pdf')
        plot_idi(df['gt'],df['factor_a'],df['factor_b'],save=savePath)
        
        
        # Boot strap
        L_nri = []
        L_idi = []
        L_delong = []
        for i in range(200):
            df_samp = df.sample(frac=1,replace=True)
            # 调用函数，计算 IDI 和 NRI 及其 P 值
            try:
                idi, nri, idi_p, nri_p = idinri(df_samp['gt'],df_samp['factor_a'],df_samp['factor_b'])
                L_nri.append(nri)
                L_idi.append(idi)
            except:
                continue
            
            # Delong Test
            # z, p = delongTest(df_samp['label'],df_samp['pred']/5,df_samp['deepMRI2GGGscore']/5)
            
        mean_value = round(np.mean(L_idi),3)
        mean_std = round(np.std(L_idi),3)
        idi_95 = str(mean_value)+'['+str(round(mean_value-mean_std,3))+'-'+str(round(mean_value+mean_std,3))+']'
        
        mean_value = round(np.mean(L_nri),3)
        mean_std = round(np.std(L_nri),3)
        nri_95 = str(mean_value)+'['+str(round(mean_value-mean_std,3))+'-'+str(round(mean_value+mean_std,3))+']'   
        
        # Delong Test
        try:
            test_delong = delongTest(df['gt'],df['factor_a'],df['factor_b'])
            z,p_delong = test_delong.compute_z_p()
        except:
            p_delong = np.inf
        
        # _, _, idi_p, nri_p = idinri(df['label'],df['pred']/5,df['deepMRI2GGGscore']/5)
        # Intergrate result
        df_t = pd.DataFrame([p_delong,idi_95,idi_p,nri_95,nri_p])
        df_stat = pd.concat([df_stat,df_t],axis=1)
        
# Save
df_stat.to_excel('nri_idi_p.xlsx',index=0)











