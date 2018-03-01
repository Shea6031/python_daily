#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 16:47:14 2018

@author: xj2sgh
"""



import configparser

class ConfigUtils:
    def __init__(self, path):
        self.path = path
        self.config = configparser.ConfigParser()
        self.config.read(self.path)
        

    def getConf(config):
       # filepath = '/Users/xj2sgh/Documents/PythonDaily/config'
        filename = 'sample_config.ini'
        config.read(filename)
        return config

    def writConf():
        pass
    
    def readConf():
        pass
    
    def changeConf():
        pass