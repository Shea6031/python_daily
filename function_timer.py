#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 11:12:50 2018

@author: xj2sgh
"""

import time
from functools import wraps

def fn_timer(function):
    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        print("Total time running %s:%s seconds" %(function.__name__, str(t1-t0)))
        return result
    return function_timer

@fn_timer
def sumfunction():
    sum = 0
    for i in range(10):
      sum+=i
    print(sum)
    
if __name__ =="__main__":
    sumfunction()