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

# lgb = pd.read_csv('result/lgb_auc0.7388451174739599.csv')
# xgb = pd.read_csv('result/xgb_0.8070824999999999.csv')
# ctb = pd.read_csv('result/catboost0.80752375.csv')
# sub = xgb.copy()
# sub['isDefault'] = (lgb['isDefault'].rank()**(0.4)*xgb['isDefault'].rank()**(0.3) * ctb['isDefault'].rank()**(0.3))/200000
# sub['isDefault'] = sub['isDefault'].round(2)
# sub.to_csv("result/submission.csv",index=False)


# lgb = pd.read_csv('result/lgb_acc0.80779auc0.7403928902618715.csv')
# # xgb = pd.read_csv('result/xgb_0.8070824999999999.csv')
# ctb = pd.read_csv('result/catboost0.8077625000000002.csv')
# sub = lgb.copy()
# sub['isDefault'] = (lgb['isDefault'].rank()**(0.68) * ctb['isDefault'].rank()**(0.32))/200000
# sub['isDefault'] = sub['isDefault'].round(2)
# sub.to_csv("result/submission.csv",index=False) # 0.7405


lgb = pd.read_csv('result/lgb_acc0.8079175auc0.7404045216502018.csv')
xgb = pd.read_csv('result/xgb_0.8075875.csv')
ctb = pd.read_csv('result/catboost0.807885.csv')
sub = lgb.copy()
sub['isDefault'] = (lgb['isDefault'].rank() ** (0.4) * xgb['isDefault'].rank() ** (0.3) * ctb['isDefault'].rank() ** (
    0.3)) / 200000

sub['isDefault'] = sub['isDefault'].round(2)
sub.to_csv("result/submission.csv", index=False)
