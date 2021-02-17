import numpy
import scipy.io
import matplotlib
import matplotlib.pyplot as plt



# input matlab training set with (usually) X and Y keys for data and labels
# input mode (default 0): 0 is regular,  1 is modified cov_mat

def trainLDA(dataset_name, mode=0):
    matfile = scipy.io.loadmat(dataset_name)

    # load data into numpy arrays and transpose for ease of use
    try:
        points = numpy.array(matfile["X"]).T
        labels = numpy.array(matfile["Y"]).T
    except:
        points = numpy.array(matfile["Xtrain"]).T
        labels = numpy.array(matfile["Ytrain"]).T


    # assume binary classifier ( two classes )
    class_probs = [0]*2
    class_means = [0]*2
    for i in range(len(points)):
        class_means[int(labels[i])] += points[i]
        class_probs[int(labels[i])] += 1

    class_means[0] = class_means[0]/len(points)
    class_probs[0] = class_probs[0]/len(labels)
    class_means[1] = class_means[1]/len(points)
    class_probs[1] = class_probs[1]/len(labels)

    cov_mat = numpy.zeros((len(points[0]), len(points[0])))

    for i in range(len(points)):
        label = int(labels[i])
        # necessary reshape so matmul works as expected. shape is not changed
        point = points[i].reshape(-1,1)
        avg = class_means[label]
        elem = point - avg
        cov_mat += elem.dot(elem.T)

    cov_mat = cov_mat/len(points)
    # if mode is 1, do the alternate covariance method
    if mode == 1:
        cov_mat = 1/points[0].size * numpy.trace(cov_mat) * numpy.identity(points[0].size)

    # alias variables for easier writing of below eq
    mean0 = class_means[0].reshape(-1,1)
    mean1 = class_means[1].reshape(-1,1)
    inv_cov = numpy.linalg.inv(cov_mat)

    a = inv_cov.dot(mean0 - mean1)
    a = a/numpy.linalg.norm(a)
    b = -.5*mean0.T.dot(inv_cov).dot(mean0) + .5*mean1.T.dot(inv_cov).dot(mean1) \
        + numpy.log(class_probs[0]/class_probs[1])


    # for plotting the data to file. allows visualization of lda and troubleshooting
    one_points = numpy.array([point for point, label in zip(points, labels) if label == 1])
    zero_points = numpy.array([point for point, label in zip(points, labels) if label == 0])
    plt.scatter(one_points.T[0], one_points.T[1], color='blue')
    plt.scatter(zero_points.T[0], zero_points.T[1], color='red')
    axes = plt.gca()
    x_vals = numpy.array(axes.get_xlim()).reshape(-1,1)
    y_vals = a[1]/a[0] * x_vals
    angle = numpy.arctan(a[1]/a[0])
    b_pointx = b*numpy.cos(angle)
    b_pointy = b*numpy.sin(angle)
    plt.scatter(b_pointx, b_pointy, color='black')
    plt.plot(x_vals, y_vals, '--')
    plt.ylabel("points")
    plt.savefig(dataset_name + ".png")

    # return the decision rule parameters
    return (a, b)


def testLDA(dataset_name, a, b):
    matfile = scipy.io.loadmat(dataset_name)
    # load data into numpy arrays and transpose for ease of use
    try:
        points = numpy.array(matfile["X"]).T
        labels = numpy.array(matfile["Y"]).T
    except:
        # print(matfile)
        points = numpy.array(matfile["Xtest"]).T
        labels = numpy.array(matfile["Ytest"]).T

    cor = 0
    total = 0
    zero = 0
    for i in range(len(points)):
        total += 1
        zero += labels[i]
        test = a.T.dot(points[i].reshape(-1,1)) + b
        # print(test)
        # print(labels[i])
        if test > 0 and labels[i] == 0:
            cor += 1
        elif test < 0 and labels[i] == 1:
            cor += 1

    print(cor/total)
    # print(zero)
    # print(total)


a, b = trainLDA("./dataLDA/synthetic1.mat", 0)
testLDA("./dataLDA/synthetic1.mat", a, b)
a, b = trainLDA("./dataLDA/synthetic2.mat", 0)
testLDA("./dataLDA/synthetic2.mat", a, b)
a, b = trainLDA("./dataLDA/synthetic3.mat", 0)
testLDA("./dataLDA/synthetic3.mat", a, b)
a,b = trainLDA("./dataLDA/synthetic4.mat", 0)
testLDA("./dataLDA/synthetic4.mat", a, b)

a, b = trainLDA("./dataLDA/trainTrain.mat", 0)
testLDA("./dataLDA/testLDA.mat", a, b)
