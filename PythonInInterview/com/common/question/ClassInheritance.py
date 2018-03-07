# -*- coding: utf-8 -*-
'''
title：类继承
Created on  2018-03-07
@author: xj2sgh
'''

class A(object):
    def show(self):
        print('base show')


class B(A):
    def show(self):
        print('derived show B')

obj = B()
obj.show()
obj.__class__= A #指定类
obj.show()

#调用未绑定的父类构造方法
class C(A):
    def __init__(self):
        A.__init__(self)
    def show(self):
        print('derived show C')

objC = C()
objC.show()
objC.__class__= A #指定类
objC.show()
#使用super函数
class D(A):
    def __init__(self):
        super(D,self).__init__()
    def show(self):
        print('derived show D')

objD = D()
objD.show()
objD.__class__= A #指定类
objD.show()