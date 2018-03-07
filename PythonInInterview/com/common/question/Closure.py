# -*- coding: utf-8 -*-
'''
title：闭包
Created on  2018-03-07
@author: xj2sgh
'''

def mulby(num):
    def gn(val):
        return num*val
    return gn
zw = mulby(7)
print(zw(9))