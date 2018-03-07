# -*- coding: utf-8 -*-
'''
title：Global and Local varialble
Created on  2018-03-07
@author: xj2sgh
'''
num = 9

def f1():
    global num
    num = 20

def f2():
    print(num)

f2()
f1()
f2()