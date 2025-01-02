import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os,re
import numpy as np

from sklearn.metrics import roc_auc_score,roc_curve,classification_report,auc,\
                            accuracy_score,confusion_matrix
from matplotlib.backends.backend_pdf import PdfPages

from scipy import stats
np.seterr(divide='ignore', invalid='ignore')

class DrawPic():
    def __init__(self,colorsBar=None,linewidth=1,linestyle=None,lineNum=99):
        print('Creat DrawPic.')
        if colorsBar:
            self.colors=colorsBar
        else:
            self.colors=['red','blue','green','yellow','pink','gray','black','oranage']       
            
        self.lineWidth=linewidth
        
        if linestyle:
            self.lineStyle=linestyle
        else:
            self.lineStyle=[]
            for i in range(lineNum):
                self.lineStyle.append('-')

    def results_stat(self,df,flag=None,cutoff=None):
        # if not re.search('*',flag):   
        if df['score'].dtypes is np.dtype('float'):
            # for continuous
            y_true = np.array(df['goal']).astype(np.float64)
            y_score = np.array(df['score']).astype(np.float64)
            fpr, tpr, ths = roc_curve(y_true,y_score)
            if cutoff is None:
                cutoff,roc_Idx = self.Youden_cutof(fpr, tpr, ths)
                # print(cutoff)
            else:
                # print(ths)
                roc_Idx = np.where(ths>=cutoff)[0][-1]
                cutoff = cutoff
            
            if flag =='AUC':
                return auc(fpr, tpr)
            
            if flag =='SEN':
                return tpr[roc_Idx]          

            if flag =='SPE':
                return 1-fpr[roc_Idx]
            
            if flag =='FPR':
                return fpr[roc_Idx]
            
            if flag =='FNR':
                return 1-tpr[roc_Idx]
            
            if flag =='LR+':
                return tpr[roc_Idx]/fpr[roc_Idx]+0.001
            
            if flag =='LR-':
                return fpr[roc_Idx]/tpr[roc_Idx]+0.001
            
            #
            if flag =='SEN_SPE>0.95':
                return tpr[np.where(fpr<0.05)[0][-1]]
            
            if flag =='SPE_SEN>0.95':
                return 1-fpr[np.where(tpr>=0.95)[0][0]]
            
            if flag =='FPR_FNR<0.05':
                return fpr[np.where(((1-tpr))<0.05)[0][0]]
            
            if flag =='FNR_FPR<0.05':
                return (1-tpr)[np.where(fpr<0.05)[0][-1]]       
            
            if 'PPV>0.95' in flag:
                cp = np.where(y_true==1)[0].shape[0]
                cn = np.where(y_true==0)[0].shape[0]
                tp = cp*tpr
                # fn = cp*(1-tpr)
                fp = cn*fpr
                # tn = cn*(1-fpr)
                
                ppv = tp/(tp+fp)
                # npv = tn/(fn+tn)
                idx_PPVBest = np.where(ppv>0.95)[0][-1]
                
                if 'SEN_PPV>0.95':
                    return tpr[idx_PPVBest]
                elif 'SPE_PPV>0.95':
                    return (1-fpr)[idx_PPVBest]
                
            
                
            # PPV NPV ACCs
            y_pred = np.where(y_score>=cutoff,1,0)
            rep = classification_report(y_true,y_pred,output_dict=True)
            
            if flag =='PPV':   
                return rep['1.0']['precision']
            
            if flag =='NPV':   
                return rep['0.0']['precision']
            
            if flag=='ACC':
                return accuracy_score(y_true,y_pred)
                
        else:
            print('Require float columns of the score')
            
    def results_stat_class(self,df,flag=None,true_labels=None):        
        y_true = np.array(df['goal']).astype(np.int0)
        y_pred = np.array(df['score']).astype(np.int0)
        
        rep = classification_report(y_true,y_pred,output_dict=True,
                                    labels = true_labels)
        if flag == 'accuracy':
            return rep[flag]
        else:
            key = flag.split('_')[0]
            key_type = flag.split('_')[1]
            return rep[key][key_type]
   
              
    def bootstrap(self,df,flag=None,class_flag=0,cutoff=None):
        res_L = []
        for f in flag: 
            L=[]
            i=0
            err_flag = 0
            while(i<1000):
                mydf = df.sample(frac=1,replace=True)
                try:
                    if not isinstance(class_flag,list) and class_flag==0:
                        value = self.results_stat(mydf,f,cutoff)
                    else:
                        value = self.results_stat_class(mydf,f,true_labels=class_flag)
                        
                    if np.isnan(value):
                        continue
                    else:
                        L.append(value)
                        i +=1
                        err_flag = 0 # 用于减少采样时没有足够丰富样本的比例
                except:
                    err_flag +=1
                    if err_flag>20:
                        L.append(-1)
                        break
                                       
  
            mean = np.mean(L,axis=0)
            std = np.std(L,axis=0)
            mean_a = mean-std
            mean_b = mean+std
            
            if mean_a>1:  mean_a=1.0
            if mean_b>1:  mean_b=1.0
            print('{}={} [{}-{}]'.format(f,round(mean,3),round(mean_a,3),round(mean_b,3)))
            print('--------------------------------------')   
            res_str = '{} [{}-{}]'.format(round(mean,3),round(mean_a,3),round(mean_b,3))
            res_L.append([f,res_str])
        return res_L
    
    
    def Youden_cutof(self,fpr, tpr, ths):
        return ths[np.argmax(tpr-fpr)],np.argmax(tpr-fpr)
    
    def convert_score2pred(self,df,cutOff=None):
        if cutOff is None:
            y_true = np.array(df['goal']).astype(np.float64)
            y_score = np.array(df['score']).astype(np.float64)
            fpr, tpr, ths = roc_curve(y_true,y_score)
            youden_value,youden_idx = self.Youden_cutof(fpr, tpr, ths)
            
            y_pred = np.where(y_score>youden_value,1,0)
            # if not 'pred_tf' in df.columns:
        else:
            y_score = np.array(df['score']).astype(np.float64)
            y_pred = np.where(y_score>=cutOff,1,0)
        df['pred_tf']=y_pred
        return df
    
    
    def draw_ROC(self,x,save_Dir=None,plot_title='ROC',merge=0,add_axes =1,cmapId=None):     
        plt.rcParams['font.family'] = ['Arial']
        plt.rcParams['font.size'] = 6
        
        if merge:
            # fig = plt.figure(figsize=(5,5),dpi=600)
            fig, ax = plt.subplots(1,1,figsize=(2.5,2.5),dpi=600)
            plt.title(plot_title)
            plt.xlabel("1-Specificity")
            plt.ylabel("Sensitivity")      
            plt.xlim(-0.05, 1.05)
            plt.ylim(-0.05,1.05) 
            
            y_trues,y_preds = [],[]
            tprs,fprs = [],[]
            auc_values = []

            for i in range(len(x)):
                y_true = np.array(x[i]['goal']).astype(np.float64)
                y_pred = np.array(x[i]['score']).astype(np.float64)
                fpr, tpr, ths = roc_curve(y_true,y_pred)
                y_trues.extend(list(y_true))
                y_preds.extend(list(y_pred))
                fprs.extend(fpr)
                tprs.extend(tpr)
                
                auc_value = auc(fpr, tpr)
                auc_values.append(auc_value)
                plt.plot(fpr, tpr, color=self.colors[i],linestyle=self.lineStyle[i],\
                         linewidth=self.lineWidth,label = 'AUC='+str(round(auc_value,3)))
                # plt.legend(loc = 'lower right')
                print('AUC='+str(round(auc_value,3)))
                
                cutoff,roc_Idx = self.Youden_cutof(fpr, tpr, ths)
                print('youden index',i,cutoff)
            
            # shwo mean roc
            auc_mean = round(np.mean(auc_values[:-1]),3)
            print(auc_mean)
            # auc_mean = 0.5
            plt.plot(sorted(fprs), sorted(tprs), color='black',linestyle='--',\
                      linewidth=self.lineWidth,label = 'AUC='+str(auc_mean))
            # plt.legend(loc = 'lower right')     
            
            ## 放大的图片
            if add_axes ==1:
                inset_ax = fig.add_axes([0.4, 0.2, 0.45, 0.45],facecolor="white")
                ## 放大的图
                for i in range(len(x)):
                    y_true = np.array(x[i]['goal']).astype(np.float64)
                    y_pred = np.array(x[i]['score']).astype(np.float64)
                    fpr, tpr, ths = roc_curve(y_true,y_pred)
                    # 放大
                    inset_ax.plot(fpr, tpr,color=self.colors[i],linewidth = self.lineWidth,linestyle = self.lineStyle[i],
                                  label = 'AUC='+str(round(auc_value,3)))
                    
                    inset_ax.set_xlim([-0.02,0.2])
                    inset_ax.set_ylim([0.85,1.02])
                    
                    # inset_ax.set_xlim([-0.02,0.4])
                    # inset_ax.set_ylim([0.6,1.02])
                    # inset_ax.grid()

            # Save
            if save_Dir:
                save_path = os.path.join(save_Dir,'roc_merge.pdf')
                self.save_as_pdf(save_path)
                
            plt.show()            
            return 0
    
    def draw_cm(self,y_true,y_pred,palette='Blues',ticklabels=[]):#['Benign','GGG1','GGG2','GGG3','GGG4','GGG5']
        plt.rcParams['font.family'] = ['Arial']
        plt.rcParams['font.size'] = 6
        plt.rcParams["axes.labelsize"] = 6
        
        fig, ax = plt.subplots(1,1,figsize=(3,3),dpi=600)
        
        y_true = np.array(y_true).astype(np.int)
        y_pred = np.array(y_pred).astype(np.int)
        conf_mat = confusion_matrix(y_true,y_pred)
        conf_mat = np.round((conf_mat/np.sum(conf_mat)),3)
        # .20g .2%
        sns.heatmap(conf_mat,annot=True,fmt='.1%',
                    xticklabels=ticklabels[0],
                    yticklabels=ticklabels[1],
                    cmap=palette) #画热力图
        ax.set_title('confusion matrix') #标题
        ax.set_xlabel('True label') #x轴
        ax.set_ylabel('Predicted label') #y轴
        plt.show()
        return
    
    def draw_cm_plus(self,y_true,y_pred,plot_labels=['Predicted GGG','True GGG'],ticklabels=[],saveLog=None):
        # Creat canvas
        plt.rcParams['font.family'] = ['Arial']
        plt.rcParams['font.size'] = 6
        plt.rcParams["axes.labelsize"] = 6
        figure, axes = plt.subplots(2, 2, figsize=(4,4),dpi=600)
        figure.tight_layout()
        
        y_true = np.array(y_true).astype(np.int)
        y_pred = np.array(y_pred).astype(np.int)
        # 横坐标是真实类别数，纵坐标是预测类别数
        cf_matrix = confusion_matrix(y_true, y_pred)
         
        # 混淆矩阵
        ax = sns.heatmap(cf_matrix, annot=True, fmt='g', ax=axes[0][0],annot_kws={'size':6},
                         xticklabels=ticklabels[0], yticklabels=ticklabels[1],
                         cmap='Blues')
        # ax.title.set_text("Confusion Matrix")
        ax.set_xlabel(plot_labels[0])
        ax.set_ylabel(plot_labels[1])
        # if not saveLog is None:
        #     plt.savefig(os.path.join(saveLog[0], saveLog[1]+"_cf_matrix.pdf"))
        # plt.show()
         
        # 混淆矩阵 - 百分比
        cf_matrix = confusion_matrix(y_true, y_pred)
        ax = sns.heatmap(cf_matrix / np.sum(cf_matrix), annot=True, ax=axes[0][1], fmt='.1%',
                         xticklabels=ticklabels[0], yticklabels=ticklabels[1],
                         cmap='Blues')
        # ax.title.set_text("Confusion Matrix (percent)")
        ax.set_xlabel(plot_labels[0])
        ax.set_ylabel(plot_labels[1])
        # if not saveLog is None:
        #     plt.savefig(os.path.join(saveLog[0], saveLog[1]+"_cf_matrix_p.pdf"))
        # plt.show()
         
        # 召回矩阵，行和为1
        sum_true = np.expand_dims(np.sum(cf_matrix, axis=1), axis=1)
        precision_matrix = cf_matrix / sum_true
        ax = sns.heatmap(precision_matrix, annot=True, fmt='.1%', ax=axes[1][0],
                         xticklabels=ticklabels[0], yticklabels=ticklabels[1],
                         cmap='Blues')
        # ax.title.set_text("Precision Matrix")
        ax.set_xlabel(plot_labels[0])
        ax.set_ylabel(plot_labels[1])
        # if not saveLog is None:
        #     plt.savefig(os.path.join(saveLog[0], saveLog[1]+"_recall.pdf"))
        # plt.show()
         
        # 精准矩阵，列和为1
        sum_pred = np.expand_dims(np.sum(cf_matrix, axis=0), axis=0)
        recall_matrix = cf_matrix / sum_pred
        ax = sns.heatmap(recall_matrix, annot=True, fmt='.1%', ax=axes[1][1],
                         xticklabels=ticklabels[0], yticklabels=ticklabels[1],
                         cmap='Blues')
        # ax.title.set_text("Recall Matrix")
        ax.set_xlabel(plot_labels[0])
        ax.set_ylabel(plot_labels[1])
        # if not saveLog is None:
        #     plt.savefig(os.path.join(saveLog[0], saveLog[1]+"_precision.pdf"))
        # plt.show()
         
        # 绘制4张图
        plt.autoscale(enable=True)
        plt.subplots_adjust(wspace=0.2, hspace=0.2)        
        if not saveLog is None:
            plt.savefig(os.path.join(saveLog[0], saveLog[1]+"_all.pdf"), bbox_inches='tight', pad_inches=0.2)        
        plt.show()
         
        # F1矩阵
        figure, axes = plt.subplots(1, 1, figsize=(2.5,2.5),dpi=600)
        a = 2 * precision_matrix * recall_matrix
        b = precision_matrix + recall_matrix
        f1_matrix = np.divide(a, b, out=np.zeros_like(a), where=(b != 0))
        ax = sns.heatmap(f1_matrix, annot=True, fmt='.1%',
                         xticklabels=[1,2,3,4,5], yticklabels=[1,2,3,4,5],
                         cmap='Blues')
        # ax.title.set_text("F1 Matrix")
        ax.set_xlabel(plot_labels[0])
        ax.set_ylabel(plot_labels[1])
        if not saveLog is None:
            plt.savefig(os.path.join(saveLog[0], saveLog[1]+ "_f1.pdf"))
        plt.show() 
        return

    def save_as_pdf(self,save_path):
        pdf = PdfPages(save_path)
        pdf.savefig()
        pdf.close()
        return
        

# In[]
if __name__ == '__main__':
    
    dp = DrawPic(colorsBar=None,linewidth=1,linestyle=None,lineNum=1)
    
    Dir = './results'
    File = 'features & predictions.xlsx'
    df = pd.read_excel(os.path.join(Dir,File))

    # Preprocess
    df = df.rename(columns={'TRG':'goal','Nuclear':'pred'})
    # df = dp.convert_score2pred(df)
    
    df['goal'] = df['goal'].apply(lambda x:1 if x>2 else 0 )
    df['pred'] = df['pred'].apply(lambda x:1 if x>2 else 0 )
    
    dp.draw_ROC([df],save_Dir=None,plot_title=None)
    dp.bootstrap(df,flag=['AUC','ACC','SEN','SPE','PPV','NPV'])
    dp.bootstrap(df,flag=['ACC-','PPV-','NPV-'])
    
    dp.draw_ConfusionMatrix(df['goal'],df['pred'],save_path=None)
    
    dp.results_stat(df,'FNR_FPR>0.95')
    

        
        
        
        