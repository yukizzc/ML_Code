import xgboost as xgb
import pandas as pd
from sklearn.metrics import accuracy_score
data = pd.read_csv('./iris.csv', index_col=0)

feature_columns = data.columns[:-1]
target_column = data.columns[-1]

# https://www.knowledgedict.com/tutorial/ml-xgboost-objective-param-detail.html
# https://blog.csdn.net/iyuanshuo/article/details/80142730
#数据转换成Dmatrix格式，xgboost必须
xgtrain = xgb.DMatrix(data[feature_columns].values, data[target_column].values)
'''
1 101:1.2 102:0.03
0 1:2.1 10001:300 10002:400
0 0:1.3 1:0.3
1 0:0.01 1:0.3
0 0:0.2 1:0.3
每行代表一个实例，第一行'1'是实例标签，'101'和'102'是特征索引，'1.2'和'0.03'是特征值。
这种方式只记录有数值的特征的索引号，其他的都不记录达到节省内存效果，稀疏矩阵。
'''

#参数设置
params = {
    'booster': 'gbtree',
    'objective': 'multi:softmax',  # 多分类的问题
    'num_class': 3,               # 类别数，与 multisoftmax 并用
    'gamma': 0.1,                  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
    'max_depth': 20,               # 构建树的深度，越大越容易过拟合
    'lambda': 1,                   # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
    'subsample': 0.7,              # 随机采样训练样本
    'colsample_bytree': 0.7,       # 生成树时进行的列采样
    'min_child_weight': 3,
    'silent': 0,                   # 设置成1则没有运行信息输出，最好是设置为0.
    'eta': 0.002,                  # 如同学习率
    'seed': 1000,
    'nthread': 4,                  # cpu 线程数
}

#设定需要查看模型训练时的输出
watchlist = [(xgtrain,'tranin')]
num_round = 50
bst = xgb.train(params, xgtrain, num_round, watchlist)

#使用模型预测
preds = bst.predict(xgtrain)

#模型评估
print('准曲率为',accuracy_score(preds, data[target_column].values))
