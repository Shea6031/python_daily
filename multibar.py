#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 09:59:44 2018

@author: xj2sgh
"""
"""
https://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.xticks
https://segmentfault.com/a/1190000004103325
https://zhuanlan.zhihu.com/p/25128216
"""
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

names = ['A', 'B'] # 姓名
subjects = ['Chinese', 'Math', 'English'] # 科目
scores = [[65, 90, 75], [85, 80, 90]] # 成绩

def draw_multibar(name, subjects, value):
    font_size = 10 # 字体大小
    fig_size = (8, 6) # 图表大小
    size = len(name)
    n = len(subjects)
    x = np.arange(size)
    
    total_width = 0.8
    width = total_width/n
    
    # 设置柱形图宽度
    x = x - (total_width - width) / 2
    # 更新字体大小
    mpl.rcParams['font.size'] = font_size
    # 更新图表大小
    mpl.rcParams['figure.figsize'] = fig_size
    # X轴标题
    index = np.arange(len(name))
    plt.xticks(index , name)
    # Y轴范围
    plt.ylim(ymax=100, ymin=0)

    # 绘制
    #value = np.array(value)
    for i in range(n):
        rects1 = plt.bar(x+i*width, value[:,i],  width=width, label=subjects[i])
        add_labels(rects1)
    #rects2 = plt.bar(x+width, value[:,1],  width=width, label=subjects[1])
   # add_labels(rects2)
    #rects3 = plt.bar(x+2*width, value[:,2],  width=width, label=subjects[2])
   # add_labels(rects3)
    # 图表标题
    plt.title('Models Compare')
    # 图例显示在图表下方
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.03), fancybox=True, ncol=5)
    
    # 图表输出到本地
    #plt.savefig('scores_par.png')

# 添加数据标签
def add_labels(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height, height, ha='center', va='bottom')
        # 柱形图边缘用白色填充，纯粹为了美观
        rect.set_edgecolor('white')


scores = np.array(scores)
draw_multibar(names,subjects, scores)