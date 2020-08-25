import torch
import torch.nn as nn
# 损失函数的参数，第一个都是预测值，第二个是真实值label
# 参考https://blog.csdn.net/c2250645962/article/details/106014693/


def mse_test():
    # MSE均方误差
    x = torch.tensor([1., 1., 1., 1.])
    y1 = torch.tensor([1, 1, 1, 1])
    y2 = torch.tensor([2, 2, 2, 2])
    y3 = torch.tensor([3, 3, 3, 3])
    criterion = nn.MSELoss()
    y_li = [y1, y2, y3]
    for i in range(3):
        # 预测值必须只有一个维度，用这个函数扁平化values.squeeze(1)
        output = criterion(x, y_li[i])
        print('第{}个target和预测值损失值：{}'.format(i, output))


def softmax_test1():
    # NLLLoss多分类
    x = torch.tensor([[0.1, 0.2, 1., 0.3, 0.6],
                      [0.1, 0.2, 1., 0.3, 0.6],
                      [0.1, 0.2, 1., 0.3, 0.6]])
    # 在列向量上进行softmax处理， 最后就一个维度了
    m = nn.LogSoftmax(dim=1)
    y1 = torch.LongTensor([0, 0, 0])
    y2 = torch.LongTensor([1, 1, 1])
    y3 = torch.LongTensor([2, 3, 2])
    y4 = torch.LongTensor([3, 3, 3])
    y5 = torch.LongTensor([4, 4, 4])
    y_li = [y1, y2, y3, y4, y5]
    # 传入两个都是一维的
    criterion = nn.NLLLoss()
    for i in range(5):
        output = criterion(m(x), y_li[i])
        print('第{}个target和预测值损失值：{}'.format(i, output))


def softmax_test2():
    # CrossEntropyLoss多分类，用这种方便,这里3行代表3个数据对应下面y也有三个值
    x = torch.tensor([[0.1, 0.2, 1., 0.3, 0.6],
                      [0.1, 0.2, 1., 0.3, 0.6],
                      [0.1, 0.2, 1., 0.3, 0.6]])
    # 这里注意必须是LongTensor类型
    y1 = torch.LongTensor([0, 0, 0])
    y2 = torch.LongTensor([1, 1, 1])
    y3 = torch.LongTensor([2, 2, 2])
    y4 = torch.LongTensor([3, 3, 3])
    y5 = torch.LongTensor([4, 4, 4])
    y_li = [y1, y2, y3, y4, y5]
    # 传入预测值的维度就是类别数量（几种分类），第二个是真实值的维度就是一维用的是类别的索引（0表示第一个类别，3表示第四个类别）
    criterion = nn.CrossEntropyLoss()
    for i in range(5):
        output = criterion(x, y_li[i])
        print('第{}个target和预测值损失值：{}'.format(i, output))


def binary_test():
    # BCELoss, 输入和输出都是要数值，预测值是0~1之间，标签是0或者1
    x = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1], dtype=torch.float32)
    y = torch.ones_like(x)
    y2 = torch.zeros_like(y)
    criterion = nn.BCELoss()
    for i in range(len(x)):
        output = criterion(x[i], y[i])
        print('第{}个预测和标签的损失值：{}'.format(i, output))



if __name__ == "__main__":
    softmax_test2()
