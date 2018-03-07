# -*- coding: utf-8 -*-
'''
title：DefaultMthod
Created on  2018-03-07
@author: xj2sgh
'''
class A():
    def __init__(self,a,b):
        self.a1 = a
        self.b1 = b
        print('init')

    def mydefault(self):
        print('default')

    def __getattr__(self, name):
        return self.mydefault

a1 = A(10,20)
a1.fn1()
a1.fn2()
a1.fn3()
class B():
    def __init__(self,a,b):
        self.a1 = a
        self.b1 = b
        print('init')

    def mydefault(self,*args):
        print('default '+str(args[0]))

    def __getattr__(self, name):
        print('other fn: ',name)
        return self.mydefault

b1 = B(10,20)
b1.fn1(33)
b1.fn2('hello')
b1.fn3(10)
