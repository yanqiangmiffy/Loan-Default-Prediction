# -*- coding: utf-8 -*-
"""
@author: quincyqiang
@software: PyCharm
@file: gen_feas.py
@time: 2020/9/2 23:36
@description：
"""
import pickle
import multiprocessing
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
# import nltk
from sklearn.cluster import KMeans
# from nltk.corpus import stopwords
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestRegressor
from gensim.models.word2vec import Word2Vec
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from tqdm import tqdm
import os
import datetime

tqdm.pandas()


def load_data():
    train = pd.read_csv('data/train.csv')
    train_size = len(train)
    test = pd.read_csv('data/testA.csv')
    data = pd.concat([train, test], axis=0).reset_index(drop=True)

    # 非数值列
    def employmentLength_to_int(s):
        if pd.isnull(s):
            return s
        else:
            return np.int8(s.split()[0])

    cate_list = data.select_dtypes(include=['object', 'category']).columns.values.tolist()
    data['employmentLength'].replace(to_replace='10+ years', value='10 years', inplace=True)
    data['employmentLength'].replace('< 1 year', '0 years', inplace=True)
    data['employmentLength'] = data['employmentLength'].apply(employmentLength_to_int)
    # 对earliesCreditLine进行预处理
    data['earliesCreditLine'] = data['earliesCreditLine'].apply(lambda s: int(s[-4:]))

    # 设置特征
    numerical_fea = list(data.select_dtypes(exclude=['object']).columns)
    data[numerical_fea] = data[numerical_fea].fillna(data[numerical_fea].median())
    data['issueDate'] = pd.to_datetime(data['issueDate'], format='%Y-%m-%d')
    startdate = datetime.datetime.strptime('2007-06-01', '%Y-%m-%d')
    # 构造时间特征
    data['issueDateDT'] = data['issueDate'].apply(lambda x: x - startdate).dt.days

    # 通过除法映射到间隔均匀的分箱中，每个分箱的取值范围都是loanAmnt/1000
    data['loanAmnt_bin1'] = np.floor_divide(data['loanAmnt'], 1000)
    ## 通过对数函数映射到指数宽度分箱
    data['loanAmnt_bin2'] = np.floor(np.log10(data['loanAmnt']))
    data['loanAmnt_bin3'] = pd.qcut(data['loanAmnt'], 10, labels=False)

    for col in cate_list:
        data[col] = data[col].fillna(value=data[col].mode().values[0])
        lb = LabelEncoder()
        data[col] = lb.fit_transform(data[col])

    # ================== 添加转化率特征 ===================
    # 部分类别特征
    cat_list = ['grade', 'subGrade', 'employmentTitle',
                'homeOwnership', 'verificationStatus', 'purpose',
                'postCode', 'regionCode',
                'applicationType', 'initialListStatus', 'title']

    data['ID'] = data.index
    data['fold'] = data['ID'] % 5
    data.loc[data['isDefault'].isnull(), 'fold'] = 5
    target_feat = []
    for i in tqdm(cat_list):
        target_feat.extend([i + '_mean_last_1'])
        data[i + '_mean_last_1'] = None
        for fold in range(6):
            data.loc[data['fold'] == fold, i + '_mean_last_1'] = data[data['fold'] == fold][i].map(
                data[(data['fold'] != fold) & (data['fold'] != 5)].groupby(i)['isDefault'].mean()
            )
        data[i + '_mean_last_1'] = data[i + '_mean_last_1'].astype(float)

    label = 'isDefault'
    numerical_fea.remove(label)

    no_fea = ['id', 'policyCode',
              'isDefault', 'ID', 'fold',
              ]
    features = [fea for fea in data.columns if fea not in no_fea]
    train = data[:train_size]
    test = data[train_size:]

    print(features, len(features))

    return train, train['isDefault'], test, features
