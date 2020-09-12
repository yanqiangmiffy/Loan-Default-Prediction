# 贷款违约预测
天池：[零基础入门金融风控-贷款违约预测](https://tianchi.aliyun.com/competition/entrance/531830/forum)
学习资料：[FinancialRiskControl](https://github.com/datawhalechina/team-learning-data-mining/tree/master/FinancialRiskControl)
## 特征
![](others/features_importance.png)
## 模型

### lightgbm

- 基础特征 线下：`lgb_0.8072649999999999.csv` 线上`0.7342`

- 添加 5折cv转化率特征 `0.8074600000000001` 线上 `0.7368`

- 加入多组特征： 线下`0.8074874999999999` 线上`0.7385`
```text
[0.80766875, 0.80690625, 0.80741875, 0.80745, 0.80799375]
train_model_classification cost time:478.3310031890869
0.8074874999999999
```
- 提高类别转化率特征数量：线下`0.8075150000000001` 线上
```text
[0.80770625, 0.80708125, 0.8080625, 0.80725, 0.807475]
CV mean score: 0.8075, std: 0.0003.
train_model_classification cost time:436.97244095802307
0.8075150000000001
```

- 删除重复列 线上 `0.7384` 
del train['n2.1']
del test['n2.1'],test['n2.2'],test['n2.3']
```text
[0.7391476718117582, 0.7369324532976902, 0.7398767611322199, 0.7398380589746957, 0.7384306421534355]
CV mean score: 0.7388, std: 0.0011.
CV mean score: 0.8076, std: 0.0002.
train_model_classification cost time:426.4979546070099
0.7388451174739599
```
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
- 线下：catboost 0.80752375.csv 线上 0.7389

### 模型融合
```text
lgb = pd.read_csv('result/lgb_auc0.7388451174739599.csv')
xgb = pd.read_csv('result/xgb_0.8070824999999999.csv')
ctb = pd.read_csv('result/catboost0.80661375.csv')
sub = xgb.copy()
sub['isDefault'] = (lgb['isDefault'].rank()**(0.7)*xgb['isDefault'].rank()**(0.15) * ctb['isDefault'].rank()**(0.15))/200000
sub['isDefault'] = sub['isDefault'].round(2)
sub.to_csv("result/submission.csv",index=False)
```
线上：score:0.7384