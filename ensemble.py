#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:quincyqiang 
@license: Apache Licence 
@file: ensemble.py 
@time: 2020/9/13 12:35 上午
@description:
"""
import pandas as pd
lgb = pd.read_csv('result/lgb_auc0.7388451174739599.csv')
xgb = pd.read_csv('result/xgb_0.8070824999999999.csv')
ctb = pd.read_csv('result/catboost0.80661375.csv')
sub = xgb.copy()
sub['isDefault'] = (lgb['isDefault'].rank()**(0.7)*xgb['isDefault'].rank()**(0.15) * ctb['isDefault'].rank()**(0.15))/200000
sub['isDefault'] = sub['isDefault'].round(2)
sub.to_csv("result/submission.csv",index=False)
