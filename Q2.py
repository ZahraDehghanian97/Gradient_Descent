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

    def standardize(self, data):
        return (data - np.mean(data)) / (np.max(data) - np.min(data))

    def hypothesis(self, theta, x):
        h = theta[0]
        for i in np.arange(1, len(theta)):
            h += theta[i] * x ** i
        return h

    def computeCost(self, x, y, theta, landa):
        m = len(y)
        h = self.hypothesis(theta, x)
        errors = h - y
        regularizationTerm = landa * np.sum(theta ** 2)
        jTheta = (1 / (2 * m)) * np.sum(errors ** 2)
        MSE = jTheta + regularizationTerm
        return MSE

    def fit(self, order=1, tol=10 ** -3, numIters=20, learningRate=0.01, landa=0):
        d = {'x' + str(0): np.ones([1, len(x_pts)])[0]}
        for i in np.arange(1, order + 1):
            d['x' + str(i)] = self.standardize(self.x ** (i))
        d = OrderedDict(sorted(d.items(), key=lambda t: t[0]))
        X = np.column_stack(d.values())
        m = len(self.x)
        theta = np.zeros(order + 1)
        costs = []
        # global thetaR
        for i in range(numIters):
            h = self.hypothesis(theta, self.x)
            errors = h - self.y
            theta += -2 * landa * theta
            theta += -learningRate * (1 / m) * np.dot(errors, X)
            # thetaR.append(theta)
            cost = self.computeCost(self.x, self.y, theta, landa)
            costs.append(cost)
            if cost < tol:
                break
        self.costs = costs
        self.numIters = numIters
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

    def plotCost(self):
        plt.figure()
        plt.plot(np.arange(1, self.numIters + 1), self.costs, label=r'$J(\theta)$')
        plt.xlabel('Iterations')
        plt.ylabel(r'$J(\theta)$')
        plt.title('Cost vs Iterations of Gradient Descent')
        plt.legend(loc='best')
        return (self.costs[len(self.costs) - 1])


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
# #part 0
# PR.fit( order=5, tol=10 ** -3, numIters=50000, learningRate=0.7,landa=0.0001)
# PR.plot_predictedPolyLine()

# #part a
# thetaR = []
# PR.fit(order=5, tol=10 ** -3, numIters=10000, learningRate=0.7, landa=0.0001)
# PR.plot_predictedPolyLine()
# theta1 = thetaR
# thetaR = []
# PR.fit(order=5, tol=10 ** -3, numIters=10000, learningRate=0.7, landa=0)
# PR.plot_predictedPolyLine()
# theta2 = thetaR
# y = []
# for x in range(len(theta1)):
#     y.append(x)
# fig, myplt = plt.subplots(len(theta1[0]),2)
# for i in range(len(theta1[0])):
#     myplt[i,0].scatter(y, np.array(theta1)[:, i],s=2, c='b')
#     # myplt[0].title("regularization")
#     myplt[i,1].scatter(y, np.array(theta2)[:, i],s=2, c='r')
#     # myplt[1].title("without reguralization")
# plt.show()

## part b
# numIters = 10**4
# order = 5
# learningRate = 0.2
# for i in range (3):
#     lr = learningRate + (i*0.3)
#     theta = PR.fit( order=order, tol=10 ** -3, numIters=numIters, learningRate=lr,landa=0)
#     PR.plot_predictedPolyLine()
#     print("learning rate= "+str(lr)+" cost= "+str(PR.plotCost()))
# plt.show()

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