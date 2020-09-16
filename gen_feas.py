# -*- coding: utf-8 -*-
"""
@author: quincyqiang
@software: PyCharm
@file: gen_feas.py
@time: 2020/9/2 23:36
@description：
"""
import time
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


def w2v_transform(X, word2vec, length):
    # length = len(base_col[3:])
    return np.array([np.hstack([
        np.mean([word2vec[w]
                 for w in words if w in word2vec] or
                [np.zeros(length)], axis=1)
        , np.max([word2vec[w]
                  for w in words if w in word2vec] or
                 [np.zeros(length)], axis=1)
    ]) for words in X

    ])


def get_w2v(data_frame, feat, length):
    model = Word2Vec(data_frame[feat].values, size=length, window=20, min_count=1,
                     workers=10, iter=10)
    return model


def w2v_feat(data):
    tr_w2v = get_w2v(data[['rid']], 'rid', 50)
    vect = w2v_transform(data.rid.values, tr_w2v.wv, 50)
    for i in range(vect.shape[1]):
        data['w2vn' + str(i)] = vect[:, i]
    return data


def load_data():
    train = pd.read_csv('data/train.csv')
    train_size = len(train)
    test = pd.read_csv('data/testA.csv')
    if not os.path.exists('data/data_v1.pkl'):
        start_time = time.time()
        print(f'generate features started at {time.ctime()}')
        # 去除重复列
        del train['n2.1']
        del test['n2.1'], test['n2.2'], test['n2.3']
        data = pd.concat([train, test], axis=0).reset_index(drop=True)
        # 删除列
        del data['policyCode']

        print("data.shape:", data.shape)
        # ========== 数据处理 ===================

        numerical_fea = list(data.select_dtypes(exclude=['object']).columns)
        numerical_fea.remove('isDefault')
        numerical_fea.remove('id')
        # category_fea = list(filter(lambda x: x not in numerical_fea, list(data.columns)))

        # 按照中位数填充数值型特征
        loss_numerical_feas = ['revolUtil', 'pubRecBankruptcies', 'dti']
        data[loss_numerical_feas] = data[loss_numerical_feas].fillna(data[loss_numerical_feas].median())
        # 按照众数填充类别型特征
        loss_categorical_feas = ['employmentLength', 'employmentTitle', 'title', 'postCode']
        for cate_fea in loss_categorical_feas:
            data[cate_fea] = data[cate_fea].fillna(data[cate_fea].mode()[0])
        # 匿名特征
        n_feas = ['n0', 'n1', 'n2', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'n10', 'n11', 'n12', 'n13', 'n14']
        # 使用 -1 填充匿名特征 方面后续统计确实特征
        data[loss_numerical_feas] = data[loss_numerical_feas].fillna(-1)

        # ================ 时间特征提取 ==================
        # employmentLength对象类型特征转换到数值 雇佣年限
        def employmentLength_to_int(s):
            if s == '10+ years':
                return 10
            elif s == '< 1 year':
                return 0
            elif pd.isnull(s):
                return s
            else:
                return np.int8(s.split()[0])

        tqdm.pandas(desc="employmentLength_to_int", postfix=None)
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
        # data['issueDate_hour'] = data['issueDate'].dt.hour
        data['issueDate_week'] = data['issueDate'].dt.dayofweek
        data['issueDate_day'] = data['issueDate'].dt.day

        # earliesCreditLine 借款人最早报告的信用额度开立的月份
        tqdm.pandas(desc="earliesCreditLine", postfix=None)
        month_maps = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4,
                      'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8,
                      'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}

        def ym(x):
            month, year = x.split('-')
            month = month_maps[month]
            return year + '-' + str(month)

        data['earliesCreditLine'] = data['earliesCreditLine'].progress_apply(lambda x: ym(x))
        data['earliesCreditLine'] = pd.to_datetime(data['earliesCreditLine'], format='%Y-%m')
        data['earliesCreditLine'].value_counts()
        data['earliesCreditLine_year'] = data['earliesCreditLine'].dt.year
        data['earliesCreditLine_month'] = data['earliesCreditLine'].dt.month
        # 间隔特征
        ## 贷款日期与就业年限的差值
        data['issueDate_employmentLength_years_lag'] = data['issueDate_year'] - data['employmentLength_years']
        ## 贷款日期与就业年限的差值
        data['issueDate_earliesCreditLine_daylag'] = (data['issueDate'] - data['earliesCreditLine']).dt.days  # 天数间隔
        data['issueDate_earliesCreditLine_yearlag'] = (data['issueDate'] - data[
            'earliesCreditLine']).dt.days / 365  # 年间隔

        # 地区编码
        data['province'] = data['postCode'].apply(lambda x: str(x)[:2])

        # ================ 类别特征等级编码 ==================
        data['grade'] = data['grade'].map({'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7})
        tqdm.pandas(desc="subGrade_value", postfix=None)
        data['subGrade_value'] = data['subGrade'].progress_apply(lambda x: int(x[1]))

        for fea in ['subGrade', 'employmentLength', 'issueDate', 'earliesCreditLine', 'province']:
            lb = LabelEncoder()
            data[fea] = lb.fit_transform(data[fea])

        # =============== 长尾分布特征处理 ================
        # cat_list = [i for i in train.columns if i not in ['id', 'isDefault', 'policyCode']]
        cat_list = [i for i in data.columns if i not in ['id', 'isDefault', 'policyCode']]

        for i in tqdm(cat_list, desc="长尾分布特征处理"):
            if data[i].nunique() > 3:
                data['{}_count'.format(i)] = data.groupby(['{}'.format(i)])['id'].transform('count')  # 计数特征
                # data[i + '_rank'] = data.groupby(i)['id'].transform('rank')  # 排序特征

        # ===================== amount_feas 分箱特征 ===============
        amount_feas = ['loanAmnt', 'interestRate', 'installment', 'annualIncome', 'dti',
                       'ficoRangeLow', 'ficoRangeHigh', 'openAcc', 'revolBal', 'revolUtil', 'totalAcc']
        for fea in tqdm(amount_feas, desc="分箱特征"):
            # 通过除法映射到间隔均匀的分箱中，每个分箱的取值范围都是loanAmnt/1000
            data['{}_bin1'.format(fea)] = np.floor_divide(data[fea], 1000)
            ## 通过对数函数映射到指数宽度分箱
            data['{}_bin2'.format(fea)] = np.floor(np.log10(data[fea]))
            # 分位数分箱
            data['{}_bin3'.format(fea)] = pd.qcut(data[fea], 10, labels=False)

        # ===================== amount_feas 基本聚合特征 ===============
        for f in tqdm(amount_feas, desc="amount_feas 基本聚合特征"):
            for cate in ['grade', 'subGrade', 'employmentTitle', 'employmentLength', 'purpose', 'postCode',
                         'regionCode',
                         'earliesCreditLine', 'issueDate', 'issueDate_year', 'issueDate_day']:
                # for cate in ['employmentTitle', 'issueDate', 'issueDate_year', 'issueDate_day']:  # 围绕重要特征issueDate构建聚合特征
                if f != cate:
                    # data['{}_{}_medi'.format(cate, f)] = data.groupby(cate)[f].transform('median')
                    data['{}_{}_mean'.format(cate, f)] = data.groupby(cate)[f].transform('mean')
                    # data['{}_{}_max'.format(cate, f)] = data.groupby(cate)[f].transform('max')
                    # data['{}_{}_min'.format(cate, f)] = data.groupby(cate)[f].transform('min')
                    # data['{}_{}_std'.format(cate, f)] = data.groupby(cate)[f].transform('std')

        # =================== amount_feas 基本交叉特征  =============================
        for f1 in tqdm(amount_feas, desc="amount_feas 基本交叉特征"):
            for f2 in amount_feas:
                if f1 != f2:
                    data['{}_{}_ratio'.format(f1, f2)] = data[f1].values / data[f2].values
                    # data['{}_{}_multi'.format(f1, f2)] = data[f1].values * data[f2].values
                    # data['{}_{}_add'.format(f1, f2)] = data[f1].values + data[f2].values
                    # data['{}_{}_diff'.format(f1, f2)] = data[f1].values - data[f2].values

        # ==================== 匿名特征 ==================
        # 其他衍生变量 mean 和 std
        for item in tqdm(n_feas,
                         desc="其他衍生变量 mean 和 std"):
            data['grade_to_mean_' + item] = data['grade'] / data.groupby([item])['grade'].transform('mean')
            data['grade_to_std_' + item] = data['grade'] / data.groupby([item])['grade'].transform('std')

        for f in tqdm(n_feas, desc="n_feas 基本聚合特征"):
            for cate in ['grade', 'subGrade', 'employmentTitle', 'employmentLength', 'purpose', 'postCode',
                         'regionCode',
                         'earliesCreditLine', 'issueDate', 'issueDate_year', 'issueDate_day']:
                # for cate in ['employmentTitle', 'issueDate', 'issueDate_year', 'issueDate_day']:  # 围绕重要特征issueDate构建聚合特征
                if f != cate:
                    # data['{}_{}_medi'.format(cate, f)] = data.groupby(cate)[f].transform('median')
                    data['{}_{}_mean'.format(cate, f)] = data.groupby(cate)[f].transform('mean')
                    # data['{}_{}_max'.format(cate, f)] = data.groupby(cate)[f].transform('max')
                    # data['{}_{}_min'.format(cate, f)] = data.groupby(cate)[f].transform('min')
                    # data['{}_{}_std'.format(cate, f)] = data.groupby(cate)[f].transform('std')
        # 匿名特征信息提取
        data['nmean'] = data[n_feas].mean(1)
        data['ntd'] = data[n_feas].std(1)
        data['nsum'] = data[n_feas].sum(1)
        data['x_cross'] = ''
        for i in tqdm(range(0, len(n_feas)), desc="匿名特征信息提取"):
            data['x_cross'] = data['x_cross'].astype(str).values + '_' + data[n_feas[i]].astype(str).values
        lbl = LabelEncoder()
        data['x_cross'] = lbl.fit_transform(data['x_cross'].astype(str))

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

        # 缺失值统计特征
        # 缺失值统计，统计存在缺失值的特征，构造缺失值相关计数特征
        for i in tqdm(n_feas, desc="缺失值统计"):
            a = data.loc[data[i] == -1]
            e = a.groupby(['grade'])['id'].count().reset_index(name=i + '_grade_count')
            data = data.merge(e, on='grade', how='left')

            d = a.groupby(['subGrade'])['id'].count().reset_index(name=i + '_subGrade_count')
            data = data.merge(d, on='subGrade', how='left')

            m = a.groupby(['issueDate'])['id'].count().reset_index(name=i + '_issueDate_count')
            data = data.merge(m, on='issueDate', how='left')

            data['gradeloss_' + i] = data[i + '_grade_count'] / data['grade_count']
            data['subGradeloss_' + i] = data[i + '_subGrade_count'] / data['subGrade_count']
            data['issueDateloss_' + i] = data[i + '_issueDate_count'] / data['issueDate_count']
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
        # =============== word2vec=============
        data['rid'] = data.apply(lambda x: [i + 'n' + str(x[i]) for i in n_feas], axis=1)
        data = w2v_feat(data)
        del data['rid']
        # data = g2v_feat(data)

        # def oneHot(df, columns):
        #     for col in columns:
        #         pf = pd.get_dummies(df[col])
        #         pf = pf.astype('float')
        #         columns = {_: col + "_" + str(_) for _ in pf.columns}
        #         pf.rename(columns=columns, inplace=True)
        #         df = pd.concat([df, pf], axis=1)
        #         # df.drop(col, axis=1, inplace=True) #326
        #     return df
        #
        # data = oneHot(data, ['job', 'ethnic', 'gender', 'loanProduct'])
        data.to_pickle("data/data_v1.pkl")
        end_time = time.time()

        print("generate features cost time:{}".format(end_time - start_time))
    else:
        data = pd.read_pickle('data/data_v1.pkl')
    no_fea = ['id', 'policyCode',
              'isDefault', 'ID', 'fold',
              ]
    features = [fea for fea in data.columns if fea not in no_fea]
    train = data[:train_size]
    test = data[train_size:]

    print(len(features), features)
    del data
    return train, train['isDefault'], test, features

train, train['isDefault'], test, features=load_data()

