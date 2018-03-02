#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 09:10:10 2018

@author: xj2sgh
"""
import matplotlib.pyplot as plt
from pylab import *                                 #支持中文
mpl.rcParams['font.sans-serif'] = ['SimHei']



names = ['AutoForecastModel', 'DoubleExpSmoothModel', 'MovingAverageModel', 'MulLinearRegression',\
        'NaiveForecastingModel','OlympicModel','PolyRegressionModel','RegressionModel','SimpleExpSmootModel',
        'TripleExpSmoothModel','WeightedMovingAverage','SpectralSmoother']

x = range(len(names))
y = [120, 30, 10, 1, 1, 1, 1, 1, 10, 10, 10, 30]

plt.plot(x, y, marker='o', mec='r', mfc='w',label=u'time(s)')

plt.legend()  # 让图例生效
plt.xticks(x, names, rotation=90)
plt.margins(0)
plt.subplots_adjust(bottom=0.15)
plt.xlabel("models") #X轴标签
plt.ylabel(u"time(s)") #Y轴标签
plt.title("Model time using") #标题

plt.show()
