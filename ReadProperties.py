#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 21:51:35 2018

@author: xj2sgh

Parser the ini file
"""
import sys
import configparser


def properties(*args, **kwargs):
    if (len(args) == 0 | len(kwargs) == 0):
        print("You should input some files!")
        sys.exit()
    conf = configparser.ConfigParser(allow_no_value=True) #allow_no_value=True允许非键值对
    conf.read(args[0])
    section = conf.sections()

    for s in section:
        option = conf.options(s)
        for o in option:
            print(o,conf.get(s,o))



if __name__ == "__main__":
    properties("dbconf.ini")
