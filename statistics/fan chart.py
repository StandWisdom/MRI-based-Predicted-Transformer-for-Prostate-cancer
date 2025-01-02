# 导入matplotlib库
import matplotlib.pyplot as plt

# Set
plt.rcParams['font.family'] = ['Arial']
plt.rcParams['font.size'] = 6
plt.rcParams["axes.labelsize"] = 6

# 设置数据和标签
data = [80, 5,5, 5, 5] # 扇形的比例


labels = ['Consistent evaluation', 'B', 'C', 'D','e'] # 扇形的标签
colors=[ '#959595',
          '#09488F','#2473B5','#448FC6','#6DACD5','#C7DCF1','#EEF1FA',
          '#FFD878','#F8B04E','#F08743','#EC694C','#CB181E',
          # '#1B1919'
          ] # standard cmap (主图用改)
colors_L=[ ['#959595'],
          ['#09488F','#2473B5','#448FC6','#6DACD5','#C7DCF1','#EEF1FA'],
          ['#FFD878','#F8B04E','#F08743','#EC694C','#CB181E'],
          # '#1B1919'
          ] # standard cmap (主图用改)

# 创建一个画布
fig, ax = plt.subplots(1,1,figsize=(2,2),dpi=600)

# 绘制饼图
plt.pie(data, labels=labels, colors=colors, autopct='%d (%0.1f)'%(50,80))
plt.axis('equal') # 设置坐标轴比例为相等，使饼图为圆形
# plt.title('扇形图示例') # 设置标题
plt.show() # 显示图形
