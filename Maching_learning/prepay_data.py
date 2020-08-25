from sklearn import datasets
import pandas as pd
# 读取鸢尾花数据集
iris = datasets.load_iris()
iris_data=iris['data']
iris_feature_names = iris['feature_names']

iris_label=iris['target']
iris_target_names=iris['target_names']
# 合并特征和标签列
df = pd.DataFrame(iris_data,columns=iris_feature_names)
df['label'] = iris_label

if __name__ == "__main__":
    # generate file
    print('文件生成')
    df.to_csv('iris.csv')