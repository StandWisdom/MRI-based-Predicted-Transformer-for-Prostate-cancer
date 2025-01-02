# 导入matplotlib.pyplot模块
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# In[]
def plot_up_down_grading(stat,save):
    # Set
    plt.rcParams['font.family'] = ['Arial']
    plt.rcParams['font.size'] = 6
    plt.rcParams["axes.labelsize"] = 6
    
    # 定义矩形的宽度和高度
    width = 0.45
    height = [0.45,0.3,0.15]
    
    # 定义矩形的颜色和透明度
    colors = ["#09488F", "#CB181E"]
    alphas = [0.2,0.3,0.8]
    
    # 创建一个画布
    fig, ax = plt.subplots(1,1,figsize=(20,2),dpi=600)
    
    # 循环绘制三个矩形
    for i in range(3):
        #绘制downgrading矩形
        data = stat[0][0]
        data_ = stat[0][1]
        # 绘制downgrading矩形
        x = width-data[i]*width
        y = 0
        rect = plt.Rectangle((x, y), width * data[i], height[i], facecolor=colors[0],fill=True, alpha=alphas[i],
                             edgecolor=None,linewidth=0.5)
        ax.add_patch(rect)
        
        bias_txt = 0.04 #
        plt.text(x=x+bias_txt/10, # 文字的x坐标
             y=height[i]-bias_txt, # 文字的y坐标
             s = '%d (%0.1f)'%(data_[i],data[i]*100),
             fontsize=8, # 文字大小
             fontweight='regular',
             color='black', # 文字颜色
             ha='left', # 水平对齐方式，可选'left', 'right', 'center'
             va='top', # 垂直对齐方式，可选'top', 'bottom', 'center', 'baseline'
            )
        
        #绘制upgrading矩形
        data = stat[1][0]
        data_ = stat[1][1]
        x,y = 1-width, 0
        rect = plt.Rectangle((x, y), width * data[i], height[i], facecolor=colors[1],fill=True, alpha=alphas[i],
                              edgecolor=None,linewidth=0.5)
        ax.add_patch(rect)
        
        plt.text(x=data[i]*width+x-bias_txt/10, # 文字的x坐标
             y=height[i]-bias_txt, # 文字的y坐标
             s = '%d (%0.1f)'%(data_[i],data[i]*100),
             fontsize=8, # 文字大小
             fontweight='regular',
             color='black', # 文字颜色
             ha='right', # 水平对齐方式，可选'left', 'right', 'center'
             va='top', # 垂直对齐方式，可选'top', 'bottom', 'center', 'baseline'
             # bbox={'facecolor':'yellow', 'edgecolor':'blue', 'alpha':0.5, 'pad':10} # 文字背景框属性，可选
            )
        
        
    # 设置坐标轴的范围和刻度
    plt.axis('off')
    if save:
        plt.savefig(save,dpi=600)
    # 显示图形
    plt.show()
    return 0


def stat_Up_Down_grading(df):
    stat = []
    data = [] # rate
    data_ = [] # num
    # All down
    totalNum = df.shape[0]
    diffNum = df[df['score']>df['goal']].shape[0]
    rate = round(diffNum/totalNum,3)
    data.append(rate)
    data_.append(diffNum)
    # down <=2
    diffNum = df[(df['score']>df['goal']) & (df['goal']<=2)].shape[0]
    rate = round(diffNum/totalNum,3)
    data.append(rate)
    data_.append(diffNum)
    # down <=1
    diffNum = df[(df['score']>df['goal']) & (df['goal']==1)].shape[0]
    rate = round(diffNum/totalNum,3)
    data.append(rate)
    data_.append(diffNum)
    stat.append([data,data_])
    
    data = [] # rate
    data_ = [] # num
    # ALL up
    diffNum = df[df['score']<df['goal']].shape[0]
    rate = round(diffNum/totalNum,3)
    data.append(rate)
    data_.append(diffNum)
    # up>=2
    diffNum = df[(df['score']<df['goal']) & (df['goal']>=2)].shape[0]
    rate = round(diffNum/totalNum,3)
    data.append(rate)
    data_.append(diffNum)
    # up>=3
    diffNum = df[(df['score']<df['goal']) & (df['goal']>=3)].shape[0]
    rate = round(diffNum/totalNum,3)
    data.append(rate)
    data_.append(diffNum)
    stat.append([data,data_])
    return stat


# In[]

# 定义一个自定义的百分比格式函数
def pct_format(pct):
    return ('%.1f%%' % pct) if pct > 0 else ''

def plot_pie(stat,save):
    numbers = stat[0][1]
    # Set
    plt.rcParams['font.family'] = ['Arial']
    plt.rcParams['font.size'] = 6
    plt.rcParams["axes.labelsize"] = 6
    
    labels = ['Concordance',
              'grade group=1', '2',
              '3','4','5'] # 扇形的标签
    colors=['#C9A063',
            '#e5e5e5','#d2d2d2','#bfbfbf','#9f9f9f','#7c7c7c'           
            ]

    # 创建一个画布
    
    fig, ax = plt.subplots(1,1,figsize=(1.5,1.5),dpi=600)

    # 绘制饼图
    plt.pie(stat[0][0], labels=labels, colors=colors,
            autopct=lambda x: f"{x*sum(numbers)/100:.0f}({x:.1f})", # 设置百分比和人数格式，使用lambda函数计算人数
            # autopct=pct_format, # 设置百分比和人数格式，使用lambda函数计算人数
            # wedgeprops={'linewidth': 0.5, 'edgecolor': 'black'},
            startangle=0)
    plt.axis('equal') # 设置坐标轴比例为相等，使饼图为圆形
    # plt.title('扇形图示例') # 设置标题
    if save:
        # print(save)
        plt.savefig(save,dpi=600)
    plt.show() # 显示图形
    return 0

def stat_Up_Down_grading2(df):
    stat = []
    data = [] # rate
    data_ = [] # num
    totalNum = df.shape[0]
    
    # CI
    diffNum = df[df['score']==df['goal']].shape[0] # GG-NB score
    rate = round(diffNum/totalNum,3)
    data.append(rate)
    data_.append(diffNum)
    # GGG
    for i in range(1,6):
        # diffNum = df[(df['score']!=df['goal'])&(df['score']==i)].shape[0]
        diffNum = df[(df['score']!=df['goal'])&(df['goal']==i)].shape[0]
        rate = round(diffNum/totalNum,3)
        data.append(rate)
        data_.append(diffNum)
        # print(diffNum,rate)
    
    stat.append([data,data_])
    return stat

# In[]
def joint(df):
    l=[]
    for i in range(df.shape[0]):
        a = int(list(df['GG-NB'])[i])
        b = int(list(df['deepMRI2GGGscore'])[i])
        if (a==3 or a==4) and b<a: # biopsy
            l.append(b)
        else:
            l.append(a)

        # if a>b:
        #     l.append(a-1)
        # else:
        #     l.append(a)
    return l
# In[]

import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    df_all = pd.read_excel(r'./results/features & predictions.xlsx')    

    df_stat = pd.DataFrame([])
    h_names = [
                # 'PUTH-p','JSPH','BJFH-XC','BJFH-TZ',
                # 'AHQU-C','AHQU-E','AHQU-W',
                # 'PICAI',
                'JSPH-prospective'
               # 'all'
               ] 
    # 'PUTH-p','JSPH','BJFH-XC','BJFH-TZ',
    # 'AHQU-C','AHQU-E','AHQU-W',
    # 'PICAI','JSPH-prospective','all'
    
    df_stat = pd.DataFrame([])
    for h_name in h_names:
        
        if h_name == 'all':
            df = df_all[df_all['HospName'].isin(h_names[:-1])].copy().reset_index(drop=True)
        else:
            df = df_all[df_all['HospName']==h_name].copy().reset_index(drop=True)
        
        # select
        # screen
        df = df.dropna(subset='GG-RP')
        df = df.dropna(subset='GG-NB')
        df = df[df['GG-RP']!=0] 
        df = df[df['GG-NB']!=0] #deepMRI2GGGscore
        df['PIRADS'] = df['PIRADS'].fillna(value=3)

        # 
        
        df['goal'] = df['GG-RP'].copy()
        df['deepMRI2GGGscore'] = df['deepMRI2GGGscore'].apply(lambda x:round(x))
        df['score'] = df['deepMRI2GGGscore'].apply(lambda x:int(x))
        
         
        # Plot and Save
        stat = stat_Up_Down_grading(df)
        # savePath = None
        savePath = os.path.join(r'H:\我的论文\【前列腺癌辅助诊断-多中心】\画图\result stat\pic\GGG compare',
                                h_name+'_U&D.pdf')
        plot_up_down_grading(stat,savePath)
        
        rate = np.array(stat[0][0]+stat[1][0])*100
        rate = rate.round(1)
        num = stat[0][1]+stat[1][1]
        
        df_t = pd.DataFrame(num).astype(str)+' ('+pd.DataFrame(rate).astype(str)+')'       
        df_stat = pd.concat([df_stat,df_t],axis=1)
        
        # Bootstrap
        l_stat = []
        for i in range(1):
            df_smp = df.sample(frac=1,replace=True)
            stat = stat_Up_Down_grading(df_smp)
            l_stat.append(stat[0][0]+stat[1][0])
            
        mean = np.mean(l_stat,axis=0).round(3)
        std = np.std(l_stat,axis=0).round(3)
        mean_a = mean-std
        mean_b = mean+std
        
        df_t = pd.DataFrame(mean).astype(str)+' ['+pd.DataFrame(mean_a).astype(str)+'-'+pd.DataFrame(mean_b).astype(str)+']'       
        df_stat = pd.concat([df_stat,df_t],axis=1)
        break
    
        # fan 
        stat = stat_Up_Down_grading2(df)
        print('***********')
        # savePath = None
        savePath = os.path.join(r'H:\我的论文\【前列腺癌辅助诊断-多中心】\画图\result stat\pic\GGG compare',
                                h_name+'_pie.pdf')
        plot_pie(stat,savePath)

        # break

    df_stat.to_excel('up_down_grad.xlsx',index=0)






