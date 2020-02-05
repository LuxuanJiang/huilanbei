# 导入模块
from sklearn.model_selection import train_test_split
from sklearn import datasets
from pf import*
from sklearn.neighbors import KNeighborsClassifier
import cv2

def main():
    # k近邻函数
    iris = datasets.load_iris()
    # 导入数据和标签
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


    train_datas = np.array(train_datas)
    train_datas_vector = train_datas.reshape(train_datas.shape[0], -1)
    train_datas = train_datas_vector / 255

    test_lables = np.array(test_lables)

    test_datas = np.array(test_datas)
    test_datas_vector = test_datas.reshape(test_datas.shape[0], -1)
    test_datas = test_datas_vector / 255
    print(test_lables.shape)
    print(test_datas.shape)
    print(train_labels.shape)
    print(train_datas.shape)
    print("----------------")
    print('数据导入成功:')
    print("训练集有" + str(train_datas.shape[0]) + "个")
    print("测试集有" + str(test_datas.shape[0]) + "个\n")
    print("正在训练模型")
    print("--------------------------------")
    # print(y_train)
    # 设置knn分类器
    knn = KNeighborsClassifier()
    # 进行训练
    knn.fit(train_datas, train_labels)
    # 使用训练好的knn进行数据预测

    print("\n测试集预测准确率: {} %".format(100 - np.mean(np.abs(knn.predict(train_datas) - train_labels)) * 100)+"\n")


if __name__ == '__main__':
    main()