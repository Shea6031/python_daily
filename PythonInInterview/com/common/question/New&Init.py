# -*- coding: utf-8 -*-
'''
title：New和Init
Created on  2018-03-07
@author: xj2sgh
'''
class B():
    def fn(self):
        print('B fn')
    def __init__(self):
        print('B INIT')

class A():
    def fn(self):
        print('A fn')

    def __new__(cls, a):
        print('NEW',a)
        if a>10:
            return super(A,cls).__new__(cls)
        return B()

    def __init__(self,a):
        print('INIT',a)

a1 = A(5)
a1.fn()

a2 = A(20)
a2.fn()