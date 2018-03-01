#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 16:47:14 2018

@author: xj2sgh
"""
"""
参考：
https://docs.python.org/3/library/configparser.html
https://my.oschina.net/flymaxty/blog/222748
https://www.cnblogs.com/shellshell/p/6947049.html
"""
"""
1、config=ConfigParser.ConfigParser()  
创建ConfigParser实例  
  
2、config.sections()  
返回配置文件中节序列  
  
3、config.options(section)  
返回某个项目中的所有键的序列  
  
4、config.get(section,option)  
返回section节中，option的键值  
  
5、config.add_section(str)  
添加一个配置文件节点(str)  
  
6、config.set(section,option,val)  
设置section节点中，键名为option的值(val)  
  
7、config.read(filename)  
读取配置文件  
  
8、config.write(obj_file)  
写入配置文件  

"""

import configparser
import sys


class ConfigUtils:
    def __init__(self, path):
        self.path = path
        self.config = configparser.ConfigParser()
        try:
            self.config.read(self.path)
        except:
            print("There is no config")
            sys.exit
    
    def __del__(self):
        with open(self.path,'w') as fh:
            self.config.write(fh)

        

    def addSections(self,sections):
        try:
            return self.config.add_section(sections)
        except configparser.DuplicateSectionError:
            print("Section "+sections+" already exists")
            
    def set(self,section,option,value):
        try:
            self.config.set(section,option,value)
        except configparser.DuplicateSectionError:
            print("Set failed!")
    
    def readConf(self):
        sections = self.config.sections() #返回配置文件中节序列
        result = []
        try:
           for s in sections:
               options = self.config.options(s)
               for o in options:
                   result.append(s+':'+o+'='+(self.config.get(s,o)))
        except configparser.DuplicateSectionError:
                print("The config is null")
        return(result)
    
    def removeSections(self,section):
        try:
            return self.config.remove_section(section)
        except configparser.DuplicateSectionError:  
            print(section+" is not exists")
        
    
    
if __name__ == "__main__":
    configname = "example.ini"
    cf = ConfigUtils(configname)
    print(cf.readConf())
    