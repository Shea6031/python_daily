#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 22:38:34 2018

@author: xj2sgh
"""

import urllib
from urllib import parse
# request way
response = urllib.request.urlopen("https://www.zhihu.com/question/28661987")
print (response.read())

print("####################################")
#post way
values = {"username":"xj11306031@126.com","password":"SHEA7781683"}
data = parse.urlencode(values).encode('utf-8')
print(data)
url = "https://reg.163.com/logins.jsp"
response = urllib.request.urlopen(url,data)
print (response.read())
