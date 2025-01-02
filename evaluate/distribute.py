import numpy as np
import matplotlib.pyplot as plt

# 导入numpy和scipy库
import numpy as np
import scipy.stats as stats

import scipy.integrate as integrate


def find_std_norm(mean=0,density=0.1):
    # 根据均值和密度函数求解正态分布的标准差
    std = stats.norm.pdf(mean) / density # 使用概率密度函数的性质求解标准差
    return std

# 定义一个函数，接受一个正态分布函数列表作为参数，返回一个合成的分布函数
def combine_normal_distributions(normal_distributions):
    # 定义一个合成的分布函数，接受一个自变量x作为参数，返回一个因变量y
    def combined_distribution(x):
        # 初始化y为0
        y = 0
        # 遍历正态分布函数列表，对每个函数求值，并累加到y上
        for normal_distribution in normal_distributions:
            y += normal_distribution(x)
        # 返回y
        return y
    # 返回合成的分布函数
    return combined_distribution

# 定义一个函数，接受一个概率密度函数作为参数，返回一个概率累积函数
def pdf_to_cdf(pdf):
    # 定义一个概率累积函数，接受一个自变量x作为参数，返回一个因变量y
    def cdf(x):
        # 调用scipy.integrate.quad函数，传入概率密度函数、负无穷和x，得到积分值和误差
        result, error = integrate.quad(pdf, -np.inf, x) #-np.inf
        # 返回积分值，即概率累积值
        return result
    # 返回概率累积函数
    return cdf

def joint_distribution_func(pred_score):
    pred = np.argmax(pred_score)
    # 定义N个正态分布函数，分别具有不同的均值和标准差
    normal_distribution_0 = lambda x: stats.norm.pdf(x, loc=0, scale=find_std_norm(mean=0,density=pred_score[0]))
    normal_distribution_1 = lambda x: stats.norm.pdf(x, loc=1, scale=find_std_norm(mean=0,density=pred_score[1]))
    normal_distribution_2 = lambda x: stats.norm.pdf(x, loc=2, scale=find_std_norm(mean=0,density=pred_score[2]))
    normal_distribution_3 = lambda x: stats.norm.pdf(x, loc=3, scale=find_std_norm(mean=0,density=pred_score[3]))
    normal_distribution_4 = lambda x: stats.norm.pdf(x, loc=4, scale=find_std_norm(mean=0,density=pred_score[4]))
    normal_distribution_5 = lambda x: stats.norm.pdf(x, loc=5, scale=find_std_norm(mean=0,density=pred_score[5]))
    
    # 将三个正态分布函数放入一个列表中
    normal_distributions = [normal_distribution_0,
                            normal_distribution_1, normal_distribution_2, 
                            normal_distribution_3,normal_distribution_4,
                            normal_distribution_5]
    
    # 调用combine_normal_distributions函数，传入正态分布函数列表，得到合成的分布函数
    combined_distribution = combine_normal_distributions(normal_distributions)
    
    # 调用pdf_to_cdf函数，传入示例的概率密度函数，得到概率累积函数
    example_cdf = pdf_to_cdf(combined_distribution)
    
    # 使用概率累积函数，传入一个x值，得到y值，并打印结果
    # 使用二分法
    x = np.arange(pred-0.4,pred+0.5,0.1)
    y = []
    for x_value in x :
        y_value = example_cdf(x_value)/6 # y值
        y.append(y_value)
        if y_value >0.5:
            break    
        
    try:
        x_value_best = round(x[np.where(np.array(y)>0.5)[0][0]],2)
    except:
        if pred==0:
            x_value_best = 0
        else:
            x_value_best = pred-0.4
    
    return x_value_best

if __name__ == '__main__':
    pred_score = [0.01,0.08,0.05,0.04,0.04,0.77]
    predscore = [0.01,0.01,0.01,0.01,0.7,0.3]
    
    res = joint_distribution_func(predscore)
    print(res)



'''   

# 定义一个x轴的范围，从-6到6，间隔为0.01
x_range = np.arange(-20, 20, 1)
# 定义一个y轴的空列表，用于存储合成分布函数的值
y_range = []
# 遍历x轴的范围，对每个x值，调用合成分布函数，得到y值，并追加到y轴的列表中
for x in x_range:
    # y = combined_distribution(x)
    y_range.append(combined_distribution(x))

plt.figure()
plt.plot(x_range,y_range)
# xticks = np.arange(-10, 10, 1)
# plt.xticks(xticks) # 设置x轴刻度和标签
plt.show()
'''



