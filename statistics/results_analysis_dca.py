import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def calculate_net_benefit_model(thresh_group, y_pred_score, y_label):
    net_benefit_model = np.array([])
    for thresh in thresh_group:
        y_pred_label = y_pred_score > thresh
        tn, fp, fn, tp = confusion_matrix(y_label, y_pred_label).ravel()
        n = len(y_label)
        net_benefit = (tp / n) - (fp / n) * (thresh / ((1 - thresh)+0.00001))
        net_benefit_model = np.append(net_benefit_model, net_benefit)
    return net_benefit_model


def calculate_net_benefit_all(thresh_group, y_label):
    net_benefit_all = np.array([])
    tn, fp, fn, tp = confusion_matrix(y_label, y_label).ravel()
    total = tp + tn
    for thresh in thresh_group:
        net_benefit = (tp / total) - (tn / total) * (thresh / ((1 - thresh)+0.0001))
        net_benefit_all = np.append(net_benefit_all, net_benefit)
    return net_benefit_all


def plot_DCA(ax, thresh_group, net_benefit_model, net_benefit_all):
    #Plot
    ax.plot(thresh_group, net_benefit_model, color = 'crimson', label = 'MRI-PTPCA',
                  linewidth=0.75)
    ax.plot(thresh_group, net_benefit_all, color = 'black',label = 'All diagnosed as CSPCa',
                  linewidth=0.75)
    ax.plot((0, 1), (0, 0), color = 'black', linestyle = ':', label = 'All diagnosed as non-CSPCa',
                  linewidth=0.75)

    #Fill，显示出模型较于treat all和treat none好的部分
    y2 = np.maximum(net_benefit_all, 0)
    y1 = np.maximum(net_benefit_model, y2)
    ax.fill_between(thresh_group, y1, y2, color = 'crimson', alpha = 0.2)

    #Figure Configuration， 美化一下细节
    ax.set_xlim(0,1)
    ax.set_ylim(-0.05, net_benefit_model.max() + 0.15)#adjustify the y axis limitation #net_benefit_model.min() - 0.15
    ax.set_xlabel(
        xlabel = 'Threshold Probability', 
        fontdict= {'family': 'Arial', 'fontsize': 6}
        )
    ax.set_ylabel(
        ylabel = 'Net Benefit', 
        fontdict= {'family': 'Arial', 'fontsize': 6}
        )
    ax.grid('major')
    ax.spines['right'].set_color((0.8, 0.8, 0.8))
    ax.spines['top'].set_color((0.8, 0.8, 0.8))
    ax.legend(loc = 'upper right')

    return ax

def draw_DCA(x,thresh_group = np.arange(0,6,0.02),savePath=None):
    plt.rcParams['font.family'] = ['Arial']
    plt.rcParams['font.size'] = 6
    fig, ax = plt.subplots(1,1,figsize=(2.5,2.5),dpi=600)
    #
    y_label = np.array(x['goal']).astype(np.float64)
    y_pred_score = np.array(x['score']).astype(np.float64)
    
    net_benefit_model = calculate_net_benefit_model(thresh_group, y_pred_score, y_label)
    net_benefit_all = calculate_net_benefit_all(thresh_group, y_label)
    # draw
    ax = plot_DCA(ax, thresh_group, net_benefit_model, net_benefit_all)
    if savePath is not None:
        fig.savefig(savePath, dpi = 600)
    plt.show()    

