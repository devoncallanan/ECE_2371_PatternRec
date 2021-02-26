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

def pla(dataset):
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
    random.shuffle(grouped)
    mov_avg_delta = 1
    mov_avg = numpy.ones(50)
    mov_avg_ind = 0

    # last_log = log_loss(dataset, theta)
    # print(last_log)
    while test(dataset, theta) != 1. and numpy.abs(mov_avg_delta) > .00001:
        # x_hat, label = random.choice(grouped)
        x_hat, label = grouped[iterations%len(grouped)]
        iterations += 1
        # t = numpy.matmul(theta.T, x_hat)
        # plot_points.append(log_loss(dataset, theta))
        guess = theta.T.dot(x_hat)
        if (guess >= 0 and label == 0):
            theta = theta + -1*x_hat
            mov_avg_delta = mov_avg.sum()/len(mov_avg)
            mov_avg[mov_avg_ind%len(mov_avg)] = test(dataset, theta)
            mov_avg_delta = mov_avg_delta-mov_avg.sum()/len(mov_avg)
            mov_avg_ind += 1
            #
            # print(mov_avg)
        elif (guess < 0 and label == 1):
            theta = theta + 1*x_hat
            mov_avg_delta = mov_avg.sum()/len(mov_avg)
            mov_avg[mov_avg_ind%len(mov_avg)] = test(dataset, theta)
            mov_avg_delta = mov_avg_delta-mov_avg.sum()/len(mov_avg)
            mov_avg_ind += 1

            # print(mov_avg)

        # print(test(dataset, theta))
        # next_log = log_loss(dataset, theta)
        # delta = numpy.abs(last_log - next_log)
        # print(next_log)
        # last_log = next_log

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

    return cor/total



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

    matfile = io.loadmat("LRdata/synthetic1.mat")
    points = numpy.array(matfile["X"]).T
    labels = numpy.array(matfile["Y"]).T
    dataset = (points, labels)

    rule, stoch = pla(dataset)
    acc = test(dataset, rule)
    print("Accuracy for this model is " + str(acc))
    plot_rule(dataset, rule)

    matfile = io.loadmat("LRdata/synthetic2.mat")
    points = numpy.array(matfile["X"]).T
    labels = numpy.array(matfile["Y"]).T
    dataset = (points, labels)

    rule, stoch = pla(dataset)
    acc = test(dataset, rule)
    print("Accuracy for this model is " + str(acc))
    plot_rule(dataset, rule)

    matfile = io.loadmat("LRdata/synthetic3.mat")
    points = numpy.array(matfile["X"]).T
    labels = numpy.array(matfile["Y"]).T
    dataset = (points, labels)

    rule, stoch = pla(dataset)
    acc = test(dataset, rule)
    print("Accuracy for this model is " + str(acc))
    plot_rule(dataset, rule)

    matfile = io.loadmat("LRdata/synthetic4.mat")
    points = numpy.array(matfile["X"]).T
    labels = numpy.array(matfile["Y"]).T
    dataset = (points, labels)

    rule, stoch = pla(dataset)
    acc = test(dataset, rule)
    print("Accuracy for this model is " + str(acc))
    plot_rule(dataset, rule)
