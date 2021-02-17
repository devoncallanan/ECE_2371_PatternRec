# Devon Callanan
# University of Pittsburgh
# ECE 2371 Pattern Rec

import numpy
from scipy import io
import matplotlib.pyplot as plt


def g(t):
    val = 1/(1+numpy.exp(-t))
    return val

def sse(dataset, theta):
    x_hats = [numpy.array([1,x[0],x[1]]).reshape((-1,1)) for x in points]
    sse = 0
    for x_hat, label in zip(x_hats,labels):
        t = numpy.matmul(theta.T, x_hat)
        sse += label*numpy.log(g(t)) + (1-label)*numpy.log(1-g(t))
    return sse


def gradient_descent(dataset):
    points, labels = dataset
    num_points, dims = points.shape
    # initialize decision rule guess
    theta = numpy.random.rand(dims + 1, 1)
    x_hats = [numpy.array([1,x[0],x[1]]).reshape((-1,1)) for x in points]
    alpha = .1
    gradient = numpy.ones((dims + 1, 1))

    iterations = 0

    while numpy.abs(gradient.sum()) > .1 and iterations < 10000:
        iterations += 1
        gradient = numpy.zeros((dims + 1, 1))
        for x_hat, label in zip(x_hats,labels):
            # sum gradient accross all inputs
            t = numpy.matmul(theta.T, x_hat)
            gradient = gradient + x_hat*(label-g(t))

        theta = theta + gradient*alpha
        # print(sse(dataset, theta))
    # print(theta)
    print("After " + str(iterations) + " iterations gradient_descent is complete")
    # theta = theta/(numpy.sqrt(theta[1]**2+ theta[2]**2))
    theta = theta/numpy.linalg.norm(theta)
    # print(theta)
    return theta

def newton(dataset):
    points, labels = dataset
    num_points, dims = points.shape
    # initialize decision rule guess
    theta = numpy.random.rand(dims + 1, 1)
    x_hats = [numpy.array([1,x[0],x[1]]).reshape((-1,1)) for x in points]
    gradient = numpy.ones((dims + 1, 1))

    iterations = 0

    while numpy.abs(gradient.sum()) > .01 and iterations < 10000:
        iterations += 1
        hessian = numpy.zeros((dims + 1, dims + 1))
        gradient = numpy.zeros((dims + 1, 1))
        for x_hat, label in zip(x_hats,labels):
            # sum gradient accross all inputs
            t = numpy.matmul(theta.T, x_hat)
            gradient += x_hat*(label-g(t))
            hessian += x_hat.dot(x_hat.T)*g(t)*(1-g(t))
            # print(hessian)
        # print(sse(dataset, theta))

        theta = theta + (numpy.linalg.inv(hessian)).dot(gradient)
    # print(theta)
    print("After " + str(iterations) + " iterations gradient_descent is complete")
    # theta = theta/(numpy.sqrt(theta[1]**2+ theta[2]**2))
    theta = theta/numpy.linalg.norm(theta)
    # print(theta)
    return theta

def stochastic(dataset):
    points, labels = dataset
    num_points, dims = points.shape
    # initialize decision rule guess
    theta = numpy.random.rand(dims + 1, 1)
    x_hats = [numpy.array([1,x[0],x[1]]).reshape((-1,1)) for x in points]
    alpha = .01
    gradient = numpy.ones((dims + 1, 1))

    iterations = 0

    while numpy.abs(gradient.sum()) > .01 and iterations < 10000:
        # gradient = numpy.zeros((dims + 1, 1))
        for x_hat, label in zip(x_hats,labels):
            iterations += 1
            # sum gradient accross all inputs
            t = numpy.matmul(theta.T, x_hat)
            gradient = x_hat*(label-g(t))
            # print(sse(dataset, theta))

            theta = theta + gradient*alpha
            if numpy.abs(gradient.sum()) <= .01 or iterations > 10000:
                break

    # print(theta)
    print("After " + str(iterations) + " iterations gradient_descent is complete")
    # theta = theta/(numpy.sqrt(theta[1]**2+ theta[2]**2))
    theta = theta/numpy.linalg.norm(theta)
    # print(theta)
    return theta


def test(dataset, rule):
    cor = 0
    total = 0
    points, labels = dataset
    x_hats = [numpy.array([1,x[0],x[1]]).reshape((-1,1)) for x in points]

    for x_hat, label in zip(x_hats, labels):
        guess = rule.T.dot(x_hat)
        # print(guess)
        # print(label)
        if guess >= 0 and label == 1:
            cor += 1
        elif guess < 0 and label ==0:
            cor += 1
        total += 1

    print("Accuracy for this model is " + str(cor/total))


# can't seem to get this to work
def plot(dataset, rule):
    points, labels = dataset
    # for plotting the data to file. allows visualization of lda and troubleshooting
    one_points = numpy.array([point for point, label in zip(points, labels) if label == 1])
    zero_points = numpy.array([point for point, label in zip(points, labels) if label == 0])
    plt.scatter(one_points.T[0], one_points.T[1], color='blue')
    plt.scatter(zero_points.T[0], zero_points.T[1], color='red')
    axes = plt.gca()
    axes.set_aspect('equal')
    x_vals = numpy.array(axes.get_xlim()).reshape(-1,1)
    # y_vals = rule[2]/rule[1] * x_vals# - rule[0]
    angle = numpy.arctan2(rule[2],rule[1])
    b_pointx = -rule[0]*numpy.cos(angle)
    b_pointy = -rule[0]*numpy.sin(angle)
    y_vals = -rule[2]/rule[1] *( x_vals - b_pointx) + b_pointy
    plt.scatter(b_pointx, b_pointy, color='black')
    plt.plot(x_vals, y_vals, '--')
    plt.ylabel("points")
    plt.show()

if __name__ == '__main__':
    # print("Select algorithm")

    matfile = io.loadmat("LRdata/synthetic1.mat")
    points = numpy.array(matfile["X"]).T
    labels = numpy.array(matfile["Y"]).T
    dataset = (points, labels)

    rule = gradient_descent(dataset)
    test(dataset, rule)
    rule = newton(dataset)
    test(dataset, rule)
    rule = stochastic(dataset)
    test(dataset, rule)
    # plot(dataset, rule)
