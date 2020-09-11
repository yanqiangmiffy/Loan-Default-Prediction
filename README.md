# 贷款违约预测

## 特征
![](others/features_importance.png)
## 模型

### lightgbm

- 基础特征 线下：`lgb_0.8072649999999999.csv` 线上`0.7342`

- 添加 5折cv转化率特征 `0.8074600000000001` 线上 `0.7368`

### xgboost
```text
[0.80745625, 0.8065875, 0.80711875, 0.8072125, 0.8070375]
CV mean score: 0.8071, std: 0.0003.
train_model_classification cost time:2882.082005262375
0.8070824999999999
```

### catboost
```text
[0.80724375, 0.8059, 0.80655625, 0.80650625, 0.8068625]
CV mean score: 0.8066, std: 0.0004.
train_model_classification cost time:972.1086599826813
0.80661375
```