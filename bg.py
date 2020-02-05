import json
from pf import *  # 把pf里的函数导入
def main():
    d = {}
    test_female = 'test_female.csv'
    test_male = 'test_male.csv'
    with open('model.text', 'r') as f:
        d = json.load(f)
        print(d)

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