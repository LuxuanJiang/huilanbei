# 主函数
import numpy as np
import matplotlib.pyplot as plt
from pf import *  # 把pf里的函数导入
import csv
from PIL import Image
import scipy.misc
import pandas as pd
import matplotlib.image as mpimg
import cv2
import scipy


def main():
    print("正在导入照片")
    # 将图片路径保存在csv文件中
    convert_csv('train_female')
    convert_csv('train_male')
    convert_csv('test_female')
    convert_csv('test_male')
    # 声明
    train_female = 'train_female.csv'
    train_male = 'train_male.csv'
    test_female = 'test_female.csv'
    test_male = 'test_male.csv'
    train_datas = []
    train_labels = []
    test_datas = []
    test_lables = []

    # 导入数据集（182，182，3，2194）
    train_datas, train_labels = import_dataset(train_female)
    a, b = import_dataset(train_male)
    train_datas.extend(a)
    train_labels.extend(b)

    test_datas, test_lables = import_dataset(test_male)
    a, b = import_dataset(test_female)
    test_datas.extend(a)
    test_lables.extend(b)

    # 对数据进行处理
    train_labels = np.array(train_labels)
    train_labels = train_labels.reshape((1, train_labels.shape[0]))

    train_datas = np.array(train_datas)
    train_datas_vector = train_datas.reshape(train_datas.shape[0], -1).T
    train_datas = train_datas_vector / 255

    test_lables = np.array(test_lables)
    test_lables = test_lables.reshape((test_lables.shape[0], -1)).T

    test_datas = np.array(test_datas)
    test_datas_vector = test_datas.reshape(test_datas.shape[0], -1).T
    test_datas = test_datas_vector / 255
    print("----------------")
    print('数据导入成功:')
    print("训练集有" + str(train_datas.shape[1]) + "个")
    print("测试集有"+str(test_datas.shape[1])+"个\n")
    print("正在训练模型")
    print("--------------------------------")
    d = model(train_datas, train_labels, train_datas, train_labels,
              batch=10, num_iterations=1000, learning_rate=0.002)

    # 保存模型信息到model.text文件里
    with open('model.text', 'w+') as f:
        js = json.dump(d, f)


    # 将模型的学习曲线与几种学习率选择进行比较
    learning_rates = [0.0003, 0.0002, 0.0001]
    models = {}
    for i in learning_rates:
        print("learning rate is: " + str(i))
        models[str(i)] = model(train_datas, train_labels, test_datas, test_lables, num_iterations=1500, learning_rate=i,
        )
        print('\n' + "-------------------------------------------------------" + '\n')

    for i in learning_rates:
        plt.plot(np.squeeze(models[str(i)]["costs"]), label=str(models[str(i)]["learning_rate"]))

    plt.ylabel('cost')
    plt.xlabel('iterations (hundreds)')

    legend = plt.legend(loc='upper center', shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    plt.show()


    # 画学习曲线
    costs = np.squeeze(d['costs'])
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(d["learning_rate"]))
    plt.show()

    # 预测测试集并使用图表将识别的结果显示出来
    with open(test_female, 'rt') as f:
        test_female_list = csv.reader(f)
        for test_female_path in test_female_list:
            open_test(d, test_female_path[0])

    with open(test_male, 'rt') as f:
        test_male_list = csv.reader(f)
        for test_male_path in test_male_list:
            open_test(d, test_male_path[0])


if __name__ == '__main__':
    main()