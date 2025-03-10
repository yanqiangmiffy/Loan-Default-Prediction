# -*- coding: utf-8 -*-
"""
@author: quincyqiang
@software: PyCharm
@file: lgb.py
@time: 2020/9/2 23:33
@description：
"""
import time
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from gen_feas import load_data


def train_model_classification(X, X_test, y, params, num_classes=2,
                               folds=None, model_type='lgb',
                               eval_metric='logloss', columns=None,
                               plot_feature_importance=False,
                               model=None, verbose=10000,
                               early_stopping_rounds=200,
                               splits=None, n_folds=3):
    """
    分类模型函数
    返回字典，包括： oof predictions, test predictions, scores and, if necessary, feature importances.
    :params: X - 训练数据， pd.DataFrame
    :params: X_test - 测试数据，pd.DataFrame
    :params: y - 目标
    :params: folds - folds to split data
    :params: model_type - 模型
    :params: eval_metric - 评价指标
    :params: columns - 特征列
    :params: plot_feature_importance - 是否展示特征重要性
    :params: model - sklearn model, works only for "sklearn" model type
    """
    start_time = time.time()
    global y_pred_valid, y_pred

    columns = X.columns if columns is None else columns
    X_test = X_test[columns]
    splits = folds.split(X, y) if splits is None else splits
    n_splits = folds.n_splits if splits is None else n_folds

    # to set up scoring parameters
    metrics_dict = {
        'logloss': {
            'lgb_metric_name': 'logloss',
            'xgb_metric_name': 'mlogloss',
            'catboost_metric_name': 'MultiClass',
            'sklearn_scoring_function': metrics.log_loss
        },
        'lb_score_method': {
            'sklearn_scoring_f1': metrics.f1_score,  # 线上评价指标
            'sklearn_scoring_accuracy': metrics.accuracy_score,  # 线上评价指标
            'sklearn_scoring_auc': metrics.roc_auc_score
        },
    }
    result_dict = {}

    # out-of-fold predictions on train data
    oof = np.zeros(shape=(len(X), num_classes))
    # averaged predictions on train data
    prediction = np.zeros(shape=(len(X_test), num_classes))
    # list of scores on folds
    acc_scores=[]
    scores = []
    # feature importance
    feature_importance = pd.DataFrame()

    # split and train on folds
    for fold_n, (train_index, valid_index) in enumerate(splits):
        if verbose:
            print(f'Fold {fold_n + 1} started at {time.ctime()}')
        if type(X) == np.ndarray:
            X_train, X_valid = X[train_index], X[valid_index]
            y_train, y_valid = y[train_index], y[valid_index]
        else:
            X_train, X_valid = X[columns].iloc[train_index], X[columns].iloc[valid_index]
            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        if model_type == 'lgb':
            model = lgb.LGBMClassifier(**params)
            model.fit(X_train, y_train,
                      eval_set=[(X_train, y_train), (X_valid, y_valid)],
                      eval_metric=metrics_dict[eval_metric]['lgb_metric_name'],
                      verbose=verbose,
                      early_stopping_rounds=early_stopping_rounds)

            y_pred_valid = model.predict_proba(X_valid)
            y_pred = model.predict_proba(X_test, num_iteration=model.best_iteration_)

        if model_type == 'xgb':
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train,
                      eval_set=[(X_train, y_train), (X_valid, y_valid)],
                      eval_metric=metrics_dict[eval_metric]['xgb_metric_name'],
                      verbose=bool(verbose),  # xgb verbose bool
                      early_stopping_rounds=early_stopping_rounds)
            y_pred_valid = model.predict_proba(X_valid)
            y_pred = model.predict_proba(X_test, ntree_limit=model.best_ntree_limit)
        if model_type == 'sklearn':
            model = model
            model.fit(X_train, y_train)
            y_pred_valid = model.predict_proba(X_valid)
            score = metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid)
            print(f'Fold {fold_n}. {eval_metric}: {score:.4f}.')
            y_pred = model.predict_proba(X_test)

        if model_type == 'cat':
            model = CatBoostClassifier(iterations=20000, eval_metric=metrics_dict[eval_metric]['catboost_metric_name'],
                                       **params,
                                       loss_function=metrics_dict[eval_metric]['catboost_metric_name'])
            model.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=[], use_best_model=True,
                      verbose=False)

            y_pred_valid = model.predict_proba(X_valid)
            y_pred = model.predict_proba(X_test)

        oof[valid_index] = y_pred_valid
        # 评价指标
        acc_scores.append(
            metrics_dict['lb_score_method']['sklearn_scoring_accuracy'](y_valid, np.argmax(y_pred_valid, axis=1)))
        scores.append(
            metrics_dict['lb_score_method']['sklearn_scoring_auc'](y_valid, y_pred_valid[:,1]))
        print(scores)
        prediction += y_pred

        if model_type == 'lgb' and plot_feature_importance:
            # feature importance
            fold_importance = pd.DataFrame()
            fold_importance["feature"] = columns
            fold_importance["importance"] = model.feature_importances_
            fold_importance["fold"] = fold_n + 1
            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)

        if model_type == 'xgb' and plot_feature_importance:
            # feature importance
            fold_importance = pd.DataFrame()
            fold_importance["feature"] = columns
            fold_importance["importance"] = model.feature_importances_
            fold_importance["fold"] = fold_n + 1
            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)
    prediction /= n_splits
    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))

    result_dict['oof'] = oof
    result_dict['prediction'] = prediction
    result_dict['acc_scores'] = acc_scores
    result_dict['scores'] = scores

    if model_type == 'lgb' or model_type == 'xgb':
        if plot_feature_importance:
            feature_importance["importance"] /= n_splits
            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
                by="importance", ascending=False)[:50].index

            best_features = feature_importance.loc[feature_importance.feature.isin(cols)]

            plt.figure(figsize=(16, 12))
            sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
            plt.title('LGB Features (avg over folds)')
            plt.show()
            result_dict['feature_importance'] = feature_importance
    end_time = time.time()

    print("train_model_classification cost time:{}".format(end_time - start_time))
    return result_dict


# cat_paras = {
#     # 'iterations':50,
#     'learning_rate': 0.1,
#     'od_type': 'Iter',
#     'l2_leaf_reg': 3,
#     'depth': 10,
#     # 'logging_level': True
# }
# cat_params = {
#     # 'n_estimators': 20000,
#     'learning_rate': 0.1,
#     'subsample': 0.7,
#     # 'custom_loss': 'AUC',
#     # 'eval_metric': 'AUC',
#     'l2_leaf_reg': 15.69352,
#     'max_depth': 9,
#     'bootstrap_type': 'Bernoulli',
#     'boosting_type': 'Ordered',
#     'use_best_model': True,
#     'verbose': -1,
#     'thread_count': 4,
#     # 'task_type':'GPU'
# }
cat_params = {'learning_rate': 0.1, 'depth': 9, 'l2_leaf_reg': 10, 'bootstrap_type': 'Bernoulli',
              'od_type': 'Iter', 'od_wait': 50, 'random_seed': 11, 'allow_writing_files': False}
n_fold = 5
num_classes = 2
print("分类个数num_classes:{}".format(num_classes))
folds = StratifiedKFold(n_splits=n_fold, random_state=1314)
train, y, test, features = load_data()

X = train[features]
print(y.value_counts())
X_test = test[features]

result_dict_lgb = train_model_classification(X=X,
                                             X_test=X_test,
                                             y=y,
                                             params=cat_params,
                                             num_classes=num_classes,
                                             folds=folds,
                                             model_type='cat',
                                             eval_metric='logloss',
                                             plot_feature_importance=False,
                                             verbose=1,
                                             early_stopping_rounds=200)

acc_score = np.mean(result_dict_lgb['acc_scores'])
score = np.mean(result_dict_lgb['scores'])
print(score)
test['isDefault'] = result_dict_lgb['prediction'][:, 1]
test[['id', 'isDefault']].to_csv('result/cat_acc{}auc{}.csv'.format(acc_score,score), index=False)

# 保存概率文件
pd.DataFrame(result_dict_lgb['oof']).to_csv('models/cat_acc{}auc{}trainoof.csv'.format(acc_score, score),
                                                index=False, header=False)
pd.DataFrame(result_dict_lgb['prediction']).to_csv(
        'models/cat_acc{}auc{}testoof.csv'.format(acc_score, score),
        index=False, header=False)