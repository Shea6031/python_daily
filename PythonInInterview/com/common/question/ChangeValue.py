# -*- coding: utf-8 -*-
'''
title：Change two varialbles' value
Created on  2018-03-07
@author: xj2sgh
'''
a = 8
b =10
print('befor change:',a, b)

(a,b) = (b,a)
print('after change:',a,b)