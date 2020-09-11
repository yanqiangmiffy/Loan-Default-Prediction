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
from tqdm import tqdm
import warnings
import datetime
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

tqdm.pandas()


def load_data():
    train = pd.read_csv('data/train.csv')
    train_size = len(train)
    test = pd.read_csv('data/testA.csv')
    data = pd.concat([train, test], axis=0).reset_index(drop=True)

    # ========== 数据处理 ===================
    numerical_fea = list(data.select_dtypes(exclude=['object']).columns)
    label = 'isDefault'
    numerical_fea.remove(label)
    category_fea = list(filter(lambda x: x not in numerical_fea, list(data.columns)))
    # 按照平均数填充数值型特征
    data[numerical_fea] = data[numerical_fea].fillna(data[numerical_fea].median())
    # 按照众数填充类别型特征
    data['employmentLength'] = data['employmentLength'].fillna(data['employmentLength'].mode()[0])
    # 删除列
    del data['policyCode']

    # ================ 时间特征提取 ==================
    # employmentLength对象类型特征转换到数值
    def employmentLength_to_int(s):
        if s == '10+ years':
            return 10
        elif s == '< 1 year':
            return 0
        elif pd.isnull(s):
            return s
        else:
            return np.int8(s.split()[0])

    tqdm.pandas(desc="remove postfix", postfix=None)
    data['employmentLength_years'] = data['employmentLength'].progress_apply(lambda x: employmentLength_to_int(x))
    # issueDate：贷款发放的月份
    data['issueDate'] = pd.to_datetime(data['issueDate'], format='%Y-%m-%d')
    startdate = datetime.datetime.strptime('2007-06-01', '%Y-%m-%d')  # 最小日期
    # 构造时间特征
    tqdm.pandas(desc="issueDate_start_lag", postfix=None)
    data['issueDate_start_lag'] = data['issueDate'].progress_apply(lambda x: x - startdate).dt.days
    data['issueDate_start_lag2year'] = data['issueDate_start_lag'] / 365

    data['issueDate_year'] = data['issueDate'].dt.year
    data['issueDate_month'] = data['issueDate'].dt.month
    data['issueDate_hour'] = data['issueDate'].dt.hour
    data['issueDate_week'] = data['issueDate'].dt.dayofweek
    data['issueDate_day'] = data['issueDate'].dt.day
    # earliesCreditLine 借款人最早报告的信用额度开立的月份
    tqdm.pandas(desc="earliesCreditLine", postfix=None)

    def ym(row):
        x = row.earliesCreditLine
        return x.split('-')[0], x.split('-')[1]

    data[['earliesCreditLine_month', 'earliesCreditLine_year']] = data.progress_apply(lambda x: ym(x), axis=1,
                                                                                      result_type="expand")
    data['earliesCreditLine_year'] = data['earliesCreditLine_year'].astype(int)

    # ================ 类别特征等级编码 ==================
    data['earliesCreditLine_month'] = data['earliesCreditLine_month'].map({'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4,
                                                                           'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8,
                                                                           'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12})

    data['grade'] = data['grade'].map({'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7})

    tqdm.pandas(desc="subGrade_value", postfix=None)
    data['subGrade_value'] = data['subGrade'].progress_apply(lambda x: int(x[1]))

    for fea in ['subGrade', 'employmentLength', 'issueDate', 'earliesCreditLine']:
        lb = LabelEncoder()
        data[fea] = lb.fit_transform(data[fea])

    # =============== 长尾分布特征处理 ================
    cat_list = [i for i in train.columns if i not in ['id', 'isDefault', 'policyCode']]
    for i in tqdm(cat_list, desc="长尾分布特征处理"):
        if data[i].nunique() > 3:
            data['{}_count'.format(i)] = data.groupby(['{}'.format(i)])['id'].transform('count')
    # ===================== 分箱特征 ===============
    amount_feas = ['loanAmnt', 'interestRate', 'installment', 'annualIncome', 'dti',
                   'ficoRangeLow', 'ficoRangeHigh', 'openAcc', 'revolBal', 'revolUtil', 'totalAcc']
    for fea in tqdm(amount_feas, desc="分箱特征"):
        # 通过除法映射到间隔均匀的分箱中，每个分箱的取值范围都是loanAmnt/100
        data['{}_bin1'.format(fea)] = np.floor_divide(data[fea], 100)
        ## 通过对数函数映射到指数宽度分箱
        data['{}_bin2'.format(fea)] = np.floor(np.log10(data[fea]))
        # 分位数分箱
        data['{}_bin3'.format(fea)] = pd.qcut(data[fea], 10, labels=False)

    # ==================== 特征交互 ==================
    # 其他衍生变量 mean 和 std
    for item in tqdm(['n0', 'n1', 'n2', 'n2.1', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'n10', 'n11', 'n12', 'n13', 'n14'],
                     desc="其他衍生变量 mean 和 std"):
        data['grade_to_mean_' + item] = data['grade'] / data.groupby([item])['grade'].transform('mean')
        data['grade_to_std_' + item] = data['grade'] / data.groupby([item])['grade'].transform('std')

    # 类别特征nunique特征
    nuni_feat = ['grade', 'subGrade', 'employmentTitle',
                 'homeOwnership', 'verificationStatus', 'purpose',
                 'postCode', 'regionCode',
                 'applicationType', 'initialListStatus', 'title']
    multi_feat = ['grade', 'subGrade', 'employmentTitle',
                  'homeOwnership', 'verificationStatus', 'purpose',
                  'postCode', 'regionCode',
                  'applicationType', 'initialListStatus', 'title']
    for i in tqdm(nuni_feat, desc="类别特征nunique特征"):
        for j in multi_feat:
            if i != j:
                data['nuni_{0}_{1}'.format(i, j)] = data[i].map(data.groupby(i)[j].nunique())

    # ===================== 五折转化率特征 ====================
    # cat_list = ['grade', 'subGrade', 'employmentTitle',
    #             'homeOwnership', 'verificationStatus', 'purpose',
    #             'postCode', 'regionCode',
    #             'applicationType', 'initialListStatus', 'title']

    data['ID'] = data.index
    data['fold'] = data['ID'] % 5
    data.loc[data['isDefault'].isnull(), 'fold'] = 5
    target_feat = []
    for i in tqdm(cat_list, desc="5折转化率特征"):
        target_feat.extend([i + '_mean_last_1'])
        data[i + '_mean_last_1'] = None
        for fold in range(6):
            data.loc[data['fold'] == fold, i + '_mean_last_1'] = data[data['fold'] == fold][i].map(
                data[(data['fold'] != fold) & (data['fold'] != 5)].groupby(i)['isDefault'].mean()
            )
        data[i + '_mean_last_1'] = data[i + '_mean_last_1'].astype(float)
    no_fea = ['id', 'policyCode',
              'isDefault', 'ID', 'fold',
              ]
    features = [fea for fea in data.columns if fea not in no_fea]
    train = data[:train_size]
    test = data[train_size:]

    print(features, len(features))

    return train, train['isDefault'], test, features


