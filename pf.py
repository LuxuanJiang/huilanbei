# 逻辑回归二分类要用到的函数
import numpy as np
import scipy
import scipy.misc
import csv
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.image as mpimg
from scipy import ndimage
import random
import math
import json
import os
"""
    w -- 权重, 
    b -- 偏差
    X -- datas 
    Y -- labels 
    cost -- 逻辑回归的代价函数
    dw -- 相对于w的损失梯度
    db -- 相对于b的损失梯度
    grads—包含dw，db的字典
"""

# 生成测试集的地址文件csv
def convert_csv(pirture):
    if pirture == 'train_female' or pirture == 'train_male':
        p = 'train/'
    else:
        p = 'test/'
    path = p + pirture + '/'  # pirture's path.
    name = os.listdir(path)     # 返回指定的文件夹包含的文件或文件夹的名字的列表。
    with open(pirture+'.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        for n in name:
            if n[-4:] == '.jpg':
                writer.writerow([p + str(pirture) + '/' + str(n)])
            else:
                pass


# 打开图片并将图片像素以矩阵的形式保存到列表里
def import_dataset(csv_file):
    datas = []
    labels = []

    with open(csv_file, 'rt') as f:
        if csv_file == 'train_female.csv' or csv_file == "test_female.csv":
            labale = 1
        else:
            labale = 0
        list = csv.reader(f)
        for path in list:
            datas.append(np.array(Image.open(path[0], 'r')))
            labels.append(labale)
    return datas, labels


# 将待测试照片大小转化为182*182*3
def change_size(path):
    img = cv2.imread(path)
    width = 182
    height = 182
    dim = (width, height)

    # resize image
    resized = cv2.resize(img, dim)
    cv2.imwrite(path, resized)


# 初始化
def initialization(dim):
    w = np.zeros((dim, 1))  # 权重数组（182*182*3，1）
    b = 0   # 偏差
    return w, b


# 计算小批量样本的代价函数及其梯度
def propagate(w, b, X, Y):
    m = X.shape[1]
    # 前向传播
    Z = np.dot(w.T, X) + b  # （1，99372）*（99372，2194）
    C = 1 / (1 + np.exp(-Z))    # （1，2194）

    cost = np.sum(np.multiply(np.log(C + 1e-5), Y) + np.multiply(np.log(1-C + 1e-5), 1-Y)) / (-m)
    # （1，2194） 数据溢出
    # 反向传播
    dw = np.dot(X, (C-Y).T)/m   # （99372，2194）*（2194，1）
    db = np.sum(C-Y)/m
    # 代阶函数
    cost = np.squeeze(cost)

    grads = {"dw": dw,
             "db": db}

    return grads, cost

def optimize(batch,w, b, X, Y, num_iterations, learning_rate):
    """
    优化算法(梯度下降法）

   Input：

    learning_rate --学习率
    print_cost -- 是否每200步打印一次成本

    output：
    params -- 存储w和b的字典
    grads -- 存储权dw和db（偏导数）的字典
    costs -- 在优化期间计算的所有损失的列表，用于绘制学习曲线。
    """
    costs = []
    n_batch = math.ceil(len(X.T) / batch)   # 向上取整2194/

    for i in range(num_iterations):     #num_iterations -- 优化循环的迭代次数

        # 每次循环打乱顺序
        index = [i for i in range(len(X.T))]
        random.shuffle(index)
        X = X[:, index]
        Y = Y[:, index]

        # 用分层抽取n个样本构成一个小批量训练集
        V = []
        for l in range(0, np.squeeze(X[1].shape), batch):
            V.append(l)
        A = X[:, V]
        B = Y[:, V]

        # 小批量成本和梯度计算
        grads, cost = propagate(w, b, A, B)
        dw = grads["dw"]
        db = grads["db"]

        # 更新参数
        w = w - learning_rate * dw
        b = b - learning_rate * db

        # 记录成本
        if i % 100 == 0:
            costs.append(cost)

        # 每100次训练迭代打印成本
        if i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
    params = {"w": w,
              "b": b}
    grads = {"dw": dw,
             "db": db}
    return params, grads, costs


def predict(w, b, X):
    # 预测函数
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    # 包含X中示例的所有预测（0/1）的numpy数组（向量）,
    # zeros 返回一个给定形状和类型的用0填充的数组返回来一个给定形状和类型的用0填充的数组
    w = w.reshape(X.shape[0], 1)

    # 计算是女孩的概率
    A = 1 / (1 + np.exp(-(np.dot(w.T, X)+b)))
    for i in range(A.shape[1]):
        if A[0, i] <= 0.5:
            Y_prediction[0, i] = 0
        else:
            Y_prediction[0, i] = 1
        pass
    return Y_prediction


def model( X_train, Y_train, X_test, Y_test,batch=1, num_iterations=2000, learning_rate=0.001):
    """
    训练模型
    num_iterations -- 优化参数的迭代次数
    learning_rate -- 学习率
    print_cost -- 设置为true时，以每100次迭代打印成本
    """
    # 初始化wb
    w, b = initialization(X_train.shape[0])
    # 梯度下降
    parameters, grads, costs = optimize(batch, w, b, X_train, Y_train, num_iterations, learning_rate)

    # 从parameters中检索参数w和b
    w = parameters["w"]
    b = parameters["b"]

    # 预测测试/训练集
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)
    # 打印测试集预测准确率
    print("\n测试集预测准确率: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100)+"\n")

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test.tolist(),
         "Y_prediction_train": Y_prediction_train.tolist(),
         "w": w.tolist(),
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}
    return d


# 打开测试图片并判断男孩女孩
def open_test(d,myicture_path):
    change_size(myicture_path)
    img = cv2.imread(myicture_path)
    imge = cv2.imread(myicture_path)
    img = np.reshape(img, (1, img.shape[0], img.shape[1], img.shape[2]))

    img_vector = np.reshape(img, (img.shape[0], -1)).T
    img = img_vector / 255
    w = np.array(d["w"])
    y = predict(w, d["b"], img)
    if y == 1:
        cv2.putText(imge, "girl.", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, 2)
    else:
        cv2.putText(imge, "boy.", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, 2)
    cv2.imshow("Is this a boy or a girl?", imge)
    cv2.waitKey()
    cv2.destroyAllWindows()