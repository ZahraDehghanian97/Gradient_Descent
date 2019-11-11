import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
from matplotlib import style
import csv

style.use('ggplot')


class PolynomialRegression(object):
    def __init__(self, x, y):

        self.x = x
        self.y = y

    def fit(self, order=1):
        d = {}
        d['x' + str(0)] = np.ones([1, len(x_pts)])[0]
        for i in np.arange(1, order + 1):
            d['x' + str(i)] = self.x ** (i)
        d = OrderedDict(sorted(d.items(), key=lambda t: t[0]))
        X = np.column_stack(d.values())
        theta = np.matmul(np.matmul(np.linalg.pinv(np.matmul(np.transpose(X), X)), np.transpose(X)), self.y)
        self.theta = theta
        return self

    def plot_predictedPolyLine(self):
        plt.figure()
        plt.scatter(self.x, self.y, s=30, c='b')
        line = self.theta[0]
        label_holder = ['%.*f' % (2, self.theta[0])]
        for i in np.arange(1, len(self.theta)):
            line += self.theta[i] * self.x ** i
            label_holder.append(' + ' + '%.*f' % (2, self.theta[i]) + r'$x^' + str(i) + '$')
        plt.plot(self.x, line, label=''.join(label_holder))
        plt.title('Polynomial Fit: Order ' + str(len(self.theta) - 1))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend(loc='best')


def load_data():
    x_pts = []
    y_pts = []
    with open('Dataset.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                x_pts.append(float(row[0]))
                y_pts.append(float(row[1]))
                line_count += 1
    return x_pts, y_pts


x_pts, y_pts = load_data()
PR = PolynomialRegression(x_pts, y_pts)

order = 1
for i in range(4):
    theta = PR.fit(order=order)
    PR.plot_predictedPolyLine()
    order = order + (2)
plt.show()

## part c
# numIters = 10**2
# order = 5
# learningRate = 0.7
# for i in range (3):
#     numIters = numIters*10
#     theta = PR.fit( order=order, tol=10 ** -3, numIters=numIters, learningRate=learningRate,landa=0)
#     PR.plot_predictedPolyLine()
#     print("number of iteration= "+str(numIters)+" cost= "+str(PR.plotCost()))
# plt.show()


# #part d
# numIters = 10**4
# order = -1
# learningRate = 0.7
# for i in range (3):
#     order = order +2
#     theta = PR.fit( order=order, tol=10 ** -3, numIters=numIters, learningRate=learningRate,landa=0)
#     PR.plot_predictedPolyLine()
#     print("order= "+str(order)+" cost= "+str(PR.plotCost()))
# plt.show()
