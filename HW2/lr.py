# Devon Callanan
# University of Pittsburgh
# ECE 2371 Pattern Rec

import random
import numpy
from scipy import io
import matplotlib.pyplot as plt


def g(t):
    # print(t)
    val = 1/(1+numpy.exp(-t))
    # print(val)
    return val

def log_loss(dataset, theta):
    x_hats = [numpy.array([1,x[0],x[1]]).reshape((-1,1)) for x in points]
    log_loss = 0
    for x_hat, label in zip(x_hats,labels):
        t = numpy.matmul(theta.T, x_hat)
        log_loss += label*numpy.log(g(t)) + (1-label)*numpy.log(1-g(t))
    # print(log_loss)
    return numpy.abs(log_loss[0][0])/len(dataset)


def gradient_descent(dataset):
    points, labels = dataset
    num_points, dims = points.shape
    # initialize decision rule guess
    theta = numpy.random.rand(dims + 1, 1)
    x_hats = [numpy.array([1,x[0],x[1]]).reshape((-1,1)) for x in points]
    alpha = .01
    gradient = numpy.ones((dims + 1, 1))

    iterations = 0
    plot_points = []
    last_log = log_loss(dataset, theta)
    delta = 1

    while delta > .1 and iterations < 10000:
        # print(log_loss(dataset, theta))
        iterations += 1
        gradient = numpy.zeros((dims + 1, 1))
        for x_hat, label in zip(x_hats,labels):
            # sum gradient accross all inputs
            t = numpy.matmul(theta.T, x_hat)
            gradient = gradient + x_hat*(label-g(t))

        # plot_points.append(log_loss(dataset, theta))
        theta = theta + gradient*alpha
        next_log = log_loss(dataset, theta)
        delta = numpy.abs(last_log - next_log)
        last_log = next_log
        # print(gradient)
        # print(delta)

    print("After " + str(iterations) + " iterations gradient_descent is complete")

    return theta, plot_points

def newton(dataset):
    points, labels = dataset
    num_points, dims = points.shape
    # initialize decision rule guess
    theta = numpy.random.rand(dims + 1, 1)
    x_hats = [numpy.array([1,x[0],x[1]]).reshape((-1,1)) for x in points]
    gradient = numpy.ones((dims + 1, 1))

    iterations = 0
    plot_points = []
    last_log = log_loss(dataset, theta)
    delta = 1

    while delta > .1 and iterations < 1000:
        iterations += 1
        hessian = numpy.zeros((dims + 1, dims + 1))
        gradient = numpy.zeros((dims + 1, 1))
        for x_hat, label in zip(x_hats,labels):
            # sum gradient accross all inputs
            t = numpy.matmul(theta.T, x_hat)
            gradient += x_hat*(label-g(t))
            hessian += x_hat.dot(x_hat.T)*g(t)*(1-g(t))
            # print(hessian)

        # theta_pre = theta
        # plot_points.append(log_loss(dataset, theta_pre))
        theta = theta + (numpy.linalg.inv(hessian)).dot(gradient)
        next_log = log_loss(dataset, theta)
        delta = numpy.abs(last_log - next_log)
        last_log = next_log
    # print(theta)
    print("After " + str(iterations) + " iterations gradient_descent is complete")
    # theta = theta/(numpy.sqrt(theta[1]**2+ theta[2]**2))
    # theta = theta/numpy.linalg.norm(theta)
    # print(theta)
    # print(iterations)
    return theta, plot_points

def stochastic(dataset):
    points, labels = dataset
    num_points, dims = points.shape
    # initialize decision rule guess
    theta = numpy.random.rand(dims + 1, 1)
    x_hats = [numpy.array([1,x[0],x[1]]).reshape((-1,1)) for x in points]
    alpha = .1
    gradient = numpy.ones((dims + 1, 1))

    plot_points = []

    iterations = 0
    grouped = [(x_hat, label) for x_hat, label in zip(x_hats, labels)]
    numpy.random.shuffle(grouped)

    delta = 1

    last_log = log_loss(dataset, theta)

    while delta > .001:
        # print(log_loss(dataset, theta))
        # gradient = numpy.zeros((dims + 1, 1))
        x_hat, label = random.choice(grouped)
        iterations += 1
        # sum gradient accross all inputs
        t = numpy.matmul(theta.T, x_hat)
        gradient = x_hat*(label-g(t))
        # plot_points.append(numpy.linalg.norm(gradient))
        plot_points.append(log_loss(dataset, theta))
        # print(numpy.linalg.norm(gradient))

        theta = theta + gradient*alpha
        next_log = log_loss(dataset, theta)
        delta = numpy.abs(last_log - next_log)
        last_log = next_log

    print("After " + str(iterations) + " iterations gradient_descent is complete")
    # theta = theta/(numpy.sqrt(theta[1]**2+ theta[2]**2))
    # theta = theta/numpy.linalg.norm(theta)
    # print(theta)
    # print(iterations)
    return theta, plot_points


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
def plot_rule(dataset, rule):
    points, labels = dataset
    # for plotting the data to file. allows visualization of lda and troubleshooting
    one_points = numpy.array([point for point, label in zip(points, labels) if label == 1])
    zero_points = numpy.array([point for point, label in zip(points, labels) if label == 0])
    plt.scatter(one_points.T[0], one_points.T[1], color='blue')
    plt.scatter(zero_points.T[0], zero_points.T[1], color='red')
    axes = plt.gca()
    axes.set_aspect('equal')
    axes.set_ylim(-2, 3)
    x_vals = numpy.array(axes.get_xlim()).reshape(-1,1)
    # y_vals = rule[2]/rule[1] * x_vals# - rule[0]
    angle = numpy.arctan2(rule[2],rule[1])
    b_pointx = -rule[0]*numpy.cos(angle)
    b_pointy = -rule[0]*numpy.sin(angle)
    y_vals = - (rule[1]*x_vals + rule[0])/rule[2] #+ b_pointy
    # plt.scatter(b_pointx, b_pointy, color='black')
    plt.plot(x_vals, y_vals, '--')
    # plt.ylabel("points")
    plt.show()

def plot_descent(log, stoch, iterations):
    axes = plt.gca()
    plt.plot(range(iterations), log, label="Regular")
    plt.plot(range(iterations), newt, label="Newtonian")
    plt.plot(range(iterations), stoch, label="Stochastic")
    axes.legend()
    # plt.xlim(left=-5)
    plt.xscale('log')
    plt.show()


if __name__ == '__main__':
    # print("Select algorithm")

    matfile = io.loadmat("LRdata/synthetic4.mat")
    points = numpy.array(matfile["X"]).T
    labels = numpy.array(matfile["Y"]).T
    dataset = (points, labels)

    # rule, log = gradient_descent(dataset)
    # test(dataset, rule)
    # rule, newt = newton(dataset)
    # test(dataset, rule)
    rule, stoch = stochastic(dataset)
    test(dataset, rule)
    plot_rule(dataset, rule)
    # plot_descent(log, newt, stoch, 1000)
