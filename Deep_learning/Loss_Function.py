import torch
import torch.nn as nn
# 损失函数的参数，第一个都是预测值，第二个是真实值label
# 参考https://blog.csdn.net/c2250645962/article/details/106014693/


def mse_test():
    # MSE均方误差
    y = torch.tensor([1., 1., 1., 1.])
    x1 = torch.tensor([1, 1, 1, 1])
    x2 = torch.tensor([2, 2, 2, 2])
    x3 = torch.tensor([3, 3, 3, 3])
    criterion = nn.MSELoss()
    x_li = [x1, x2, x3]
    for i in range(3):
        # 预测值必须只有一个维度，用这个函数扁平化values.squeeze(1)
        output = criterion(x_li[i], y)
        print('第{}个预测值和实际值的损失值：{}'.format(i, output))


def softmax_test1():
    # NLLLoss多分类,行数就是数据个数，列数就是分类数目
    x1 = torch.tensor([[1, 0.2, 0.1],
                      [0.1, 1, 0.1],
                      [0.1, 0.2, 1],
                      [1, 0.2, 0.1]])

    x2 = torch.tensor([[1, 0.2, 0.1],
                      [0.1, 1, 0.1],
                      [0.1, 0.2, 1],
                      [0.9, 0.2, 0.1]])              
    # 每一个数据做softmax处理， 最后就一个维度了
    m = nn.LogSoftmax(dim=1)
    y = torch.LongTensor([0, 1, 2, 0])
    
    x_li = [x1, x2]
    # nn.NLLLoss的输入target是类别值
    criterion = nn.NLLLoss()
    for i in range(2):
        output = criterion(m(x_li[i]), y)
        print('第{}个预测值和实际值的损失值：{}'.format(i, output))


def softmax_test2():
    # CrossEntropyLoss多分类，用这种方便,这里4行代表4个数据对应下面y也有四个值
    x1 = torch.tensor([[1, 0.2, 0.1],
                      [0.1, 1, 0.1],
                      [0.1, 0.2, 1],
                      [1, 0.2, 0.1]])

    x2 = torch.tensor([[1, 0.2, 0.1],
                      [0.1, 1, 0.1],
                      [0.1, 0.2, 1],
                      [0.9, 0.2, 0.1]])  
    # 这里注意必须是LongTensor类型
    y = torch.LongTensor([0, 1, 2, 0])
    
    x_li = [x1, x2]
    # 传入预测值的维度就是类别数量（几种分类），第二个是真实值的维度就是一维用的是类别的索引（0表示第一个类别，3表示第四个类别）
    criterion = nn.CrossEntropyLoss()
    for i in range(2):
        output = criterion(x_li[i], y)
        print('第{}个预测值和实际值的损失值：{}'.format(i, output))


def binary_test():
    # BCELoss, 输入和输出都是要数值，预测值是0~1之间，标签是0或者1
    x = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1], dtype=torch.float32)
    y = torch.ones_like(x)
    y2 = torch.zeros_like(y)
    criterion = nn.BCELoss()
    for i in range(len(x)):
        output = criterion(x[i], y[i])
        print('第{}个预测值和实际值的损失值：{}'.format(i, output))



if __name__ == "__main__":
    softmax_test2()
